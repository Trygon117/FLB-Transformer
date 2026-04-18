import torch
import torch.nn as nn
from wavefront.wavefront_api import WavefrontConfig
from wavefront.wavefront_engine import WavefrontEngine

class FLB_Transformer_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward():
        pass
    

class FLB_Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm_lat = nn.LayerNorm(hidden_dim)
        self.norm_fdbk = nn.LayerNorm(hidden_dim)
        self.norm_engram = nn.LayerNorm(hidden_dim)

        self.proj_lateral = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_feedback = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_engram = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.transformer_block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.0)

    def forward(self, x_input, prev_lat, prev_fdbk, retrieved_engram):
        batch_size, num_cells, dim = x_input.shape
        B = batch_size * num_cells

        # 1. Linear Projection (The Priors)
        # PyTorch linear layers handle 3D tensors natively, so these remain (Batch, Cells, Dim)
        lat_prior = self.proj_lateral(self.norm_lat(prev_lat))
        fdbk_prior = self.proj_feedback(self.norm_fdbk(prev_fdbk))
        engram_token = self.proj_engram(self.norm_engram(retrieved_engram)).view(B, 1, dim)

        # 2. Reshape and Concatenate for the Transformer
        lat_token = lat_prior.view(B, 1, dim)
        fdbk_token = fdbk_prior.view(B, 1, dim)
        x_token = x_input.view(B, 1, dim)
        
        context_seq = torch.cat([lat_token, fdbk_token, engram_token, x_token], dim=1)

        # 3. Transformer Attention
        output_seq = self.transformer_block(context_seq)

        # 4. Output Splitting (The Posteriors)
        output_seq = output_seq.view(batch_size, num_cells, 4, dim)
        
        h_new_lat = output_seq[:, :, 0, :]
        h_new_fdbk = output_seq[:, :, 1, :]
        y_text = output_seq[:, :, 3, :]
        
        # 5. Joint Bayesian Predictive Loss
        # We use the original 3D priors we saved in Step 1, removing the need to reshape them again
        loss_lat = (lat_prior - h_new_lat.detach()) ** 2
        loss_fdbk = (fdbk_prior - h_new_fdbk.detach()) ** 2
        
        aux_loss_grid = loss_lat + loss_fdbk
        
        return y_text, h_new_lat, h_new_fdbk, aux_loss_grid
    
class FLB_Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, seq_len=128, batch_size=32, dropout=0.1, aux_weight=0.05):
        super().__init__()
        # Save the weight as a class variable
        self.aux_weight = aux_weight

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.emb_dropout = nn.Dropout(dropout)
        
        # Add a perimeter dropout for the final output
        self.out_dropout = nn.Dropout(dropout)
        
        # Define the grid and the specific biological routing dependencies
        # Port 0 is the Feedforward output (y_text)
        # Port 1 is the Lateral memory output (h_new_lat)
        # Port 2 is the prediction error output (h_pred_error)
        self.config = WavefrontConfig(
            grid_shape=(num_layers, seq_len),
            batch_size=batch_size,
            dim=hidden_dim,
            dependencies=[
                ((-1, 0), 0),  
                ((0, -1), 1),  
                ((1, -1), 2)   
            ],
            num_ports=4
        )
        
        self.layers = nn.ModuleList([FLB_Block(hidden_dim) for _ in range(num_layers)])
        self.engine = WavefrontEngine(self.config, self.layers)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Buffer to hold data for the offline "sleep" consolidation
        self.sleep_cycle_buffer = []
        self.surprise_threshold = 0.5  # Arbitrary threshold to define what constitutes a "surprise"
        
    def forward(self, x_input, retrieved_engram):
        # Convert (Batch, Seq_Len) -> (Batch, Seq_Len, Hidden_Dim)
        x_embeddings = self.embedding(x_input)
        
        # Apply dropout to the fresh embeddings
        x_embeddings = self.emb_dropout(x_embeddings)

        # The engine returns a tuple of our three grids
        output_grids = self.engine(x_embeddings, retrieved_engram)
        
        y_text_grid = output_grids[0]
        aux_loss_grid = output_grids[3]

        # Average the error across the hidden dimension to get a single score per cell
        cell_surprise_scores = aux_loss_grid.mean(dim=-1)
        
        # Find all cells where the model was highly surprised
        surprised_indices = torch.nonzero(cell_surprise_scores > self.surprise_threshold)
        
        if surprised_indices.numel() > 0:
            # Log these instances to our offline buffer (detach moves them safely out of the compute graph)
            self.sleep_cycle_buffer.append({
                'indices': surprised_indices.detach().cpu(),
                'input_tokens': x_input.detach().cpu()
            })
        
        # Average the distributed Bayesian errors AND scale them by our hyperparameter
        aux_loss = aux_loss_grid.mean() * self.aux_weight
        
        # 1. Swap the cell and batch dimensions
        y_text_batch_first = y_text_grid.transpose(0, 1)
        
        # 2. Unpack the flat cells back into layers and time steps
        batch_size = x_embeddings.shape[0]  # <--- THE FIX: Read dynamic batch size
        num_layers = self.config.grid_shape[0]
        seq_len = self.config.grid_shape[1]
        dim = self.config.dim
        
        # Shape becomes (batch_size, num_layers, seq_len, dim)
        y_text_reshaped = y_text_batch_first.view(batch_size, num_layers, seq_len, dim)
        
        # Slice out the very top layer for our final network predictions
        final_text_predictions = y_text_reshaped[:, -1, :, :]

        # Apply our perimeter dropout before the head makes its final guess
        final_text_predictions = self.out_dropout(final_text_predictions)

        # 3. Before returning, pass the final hidden states through the head
        # final_text_predictions shape: (Batch, Seq_Len, Hidden_Dim)
        logits = self.head(final_text_predictions)
        
        return logits, aux_loss