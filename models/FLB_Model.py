import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from wavefront.wavefront_api import WavefrontConfig
from wavefront.wavefront_engine import WavefrontEngine
from kernels.additive_attention import fused_additive_attention

class FLB_Attention_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 1. Gating Weights (Learned gain control)
        self.linear_gate_L = nn.Linear(hidden_dim, hidden_dim)
        self.linear_gate_X = nn.Linear(hidden_dim, hidden_dim)

        # 2. Additive Attention Weights
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Parameter(torch.empty(num_heads, 1, self.head_dim))
        nn.init.normal_(self.v_a, mean=0.0, std=0.02)

        # 3. Value and Output Projections
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_token, lat_token, fdbk_token):
        B, _, D = x_token.shape
        H = self.num_heads
        d_h = self.head_dim

        # 1. BIOLOGICAL GATING
        lateral_gate = torch.sigmoid(self.linear_gate_L(fdbk_token))
        lat_refined = lat_token * lateral_gate

        sensory_gate = torch.sigmoid(self.linear_gate_X(lat_refined))
        x_refined = x_token * sensory_gate

        # Token Sequence Construction
        tokens = torch.cat([x_refined, lat_refined, fdbk_token], dim=1)

        # 2. PROJECTIONS
        q = self.W_q(tokens).view(B, 3, H, d_h).transpose(1, 2).contiguous() # [B, H, 3, d_h]
        k = self.W_k(tokens).view(B, 3, H, d_h).transpose(1, 2).contiguous()
        v = self.W_v(tokens).view(B, 3, H, d_h).transpose(1, 2).contiguous()

        # 3. THE KERNEL CALL
        out = fused_additive_attention(q, k, v, self.v_a)

        # 4. OUTPUT PROCESSING
        # Recombine heads and project
        out = out.transpose(1, 2).contiguous().view(B, 3, D)
        out = self.W_o(out)

        # Map back to original identities
        # Add residuals back to keep the identity of the signals clear
        return out[:, 0:1] + x_token, out[:, 1:2] + lat_token, out[:, 2:3] + fdbk_token

class FLB_Block(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm_lat = nn.LayerNorm(hidden_dim)
        self.norm_fdbk = nn.LayerNorm(hidden_dim)

        self.proj_lateral = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_feedback = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.flb_attention = FLB_Attention_Layer(hidden_dim=hidden_dim, num_heads=8, dropout=dropout)

    def forward(self, x_input, prev_lat, prev_fdbk):
        batch_size, num_cells, dim = x_input.shape
        B = batch_size * num_cells

        # Get the priors by projecting the contexts with linear layers and applying layer normalization
        lat_prior = self.proj_lateral(self.norm_lat(prev_lat))
        fdbk_prior = self.proj_feedback(self.norm_fdbk(prev_fdbk))

        # Reshape and concatenate for the transformer
        x_token = x_input.view(B, 1, dim)
        lat_token = lat_prior.view(B, 1, dim)
        fdbk_token = fdbk_prior.view(B, 1, dim)

        # Perform Attention
        h_new_fwd, h_new_lat, h_new_fdbk = self.flb_attention(x_token, lat_token, fdbk_token)

        # Joint Bayesian Predictive Loss
        loss_lat = (lat_prior - h_new_lat.detach()) ** 2
        loss_fdbk = (fdbk_prior - h_new_fdbk.detach()) ** 2
        aux_loss_grid = loss_lat + loss_fdbk

        return h_new_fwd, h_new_lat, h_new_fdbk, aux_loss_grid
    
class FLB_Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, seq_len=128, batch_size=32, dropout=0.1, aux_weight=0.05):
        super().__init__()
        self.aux_weight = aux_weight                                # Weight for the auxiliary loss in the final loss computation
        self.embedding = nn.Embedding(vocab_size, hidden_dim)       # Embedding layer to convert token indices to dense vectors
        self.emb_dropout = nn.Dropout(dropout)                      # Dropout layer applied to the embeddings for regularization
        self.out_dropout = nn.Dropout(dropout)                      # Dropout layer applied to the final output for regularization
        
        # Define the grid and the specific biological routing dependencies
        # Port 0 is the Feedforward output (y_text)
        # Port 1 is the Lateral memory output (h_new_lat)
        # Port 2 is the prediction error output (h_pred_error)
        # Port 3 is the auxiliary loss output (aux_loss)
        self.config = WavefrontConfig(
            grid_shape=(num_layers, seq_len),
            batch_size=batch_size,
            dim=hidden_dim,
            dependencies=[
                ((-1, 0), 0),   # Previous layer to current layer's feedforward input
                ((0, -1), 1),   # current layer from previous sequence step to current layer's prev_lat 
                ((1, -1), 2)    # Previous layer from previous time step to current layer's prev_fdbk 
            ],
            num_ports=4
        )
        self.layers = nn.ModuleList([
            FLB_Block(hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.engine = WavefrontEngine(self.config, self.layers)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_input, context=None):
        if context is not None:
            lat_context, fdbk_context = context
        else:
            lat_context, fdbk_context = None, None
            
        # Get embeddings and apply dropout
        x_emb = self.embedding(x_input)
        x_emb = self.emb_dropout(x_emb)

        # Build the dynamic dictionary
        initial_states = {}
        if lat_context is not None:
            initial_states[1] = lat_context  # Port 1 is Lateral
        if fdbk_context is not None:
            initial_states[2] = fdbk_context # Port 2 is Feedback

        # The engine handles the rest!
        output_grids = self.engine(x_emb, initial_states)

        # Get the outputs from the grids
        y_text_out = output_grids[0]        # The main feedforward output grid (y_text)
        ctx_out = output_grids[1]           # The context output grid (h_new_lat)
        fdbk_out = output_grids[2]          # The feedback output grid (h_pred_error)
        aux_loss_grid = output_grids[3]     # The auxiliary loss grid (aux_loss)

        # Average the distributed Bayesian errors AND scale them by our hyperparameter
        aux_loss = aux_loss_grid.mean() * self.aux_weight

        grid_shape = (self.config.grid_shape[0], self.config.grid_shape[1], self.config.batch_size, self.config.dim)
        
        y_text_out = y_text_out.view(*grid_shape)
        ctx_out = ctx_out.view(*grid_shape)
        fdbk_out = fdbk_out.view(*grid_shape)

        # --- Extract Final Predictions ---
        # Slice out the top layer for our final network prediction
        final_text_predictions = y_text_out[-1, :, :, :]                       
        final_text_predictions = final_text_predictions.transpose(0, 1)         
        final_text_predictions = self.out_dropout(final_text_predictions)
        logits = self.head(final_text_predictions)

        # --- Extract Context Boundaries for the Next Chunk ---
        # We slice index -1 on the sequence dimension to get the boundary states!
        # Shape becomes: [Layers, Batch, Dim]
        next_lat_context = ctx_out[:, -1, :, :]  
        next_fdbk_context = fdbk_out[:, -1, :, :]

        return logits, aux_loss, (next_lat_context, next_fdbk_context)