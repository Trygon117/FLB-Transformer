import torch
import torch.nn as nn
from wavefront.wavefront_api import WavefrontConfig
from wavefront.wavefront_engine import WavefrontEngine

class FLB_Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm_lat = nn.LayerNorm(hidden_dim)
        self.norm_fdbk = nn.LayerNorm(hidden_dim)

        self.proj_lateral = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_feedback = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.transformer_block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.0)

    def forward(self, x_input, prev_lat, prev_fdbk):
        batch_size, num_cells, dim = x_input.shape
        
        # Flatten batch and cells into a single dimension
        B = batch_size * num_cells

        # 1. Linear Projection (The Priors)
        lat_token = self.proj_lateral(self.norm_lat(prev_lat)).view(B, 1, dim)
        fdbk_token = self.proj_feedback(self.norm_fdbk(prev_fdbk)).view(B, 1, dim)
        x_input = x_input.view(B, 1, dim)

        # 2. Token Concatenation
        context_seq = torch.cat([lat_token, fdbk_token, x_input], dim=1)

        # 3. Transformer Attention
        output_seq = self.transformer_block(context_seq)

        # 4. Output Splitting (The Posteriors)
        h_new_lat = output_seq[:, 0, :].view(batch_size, num_cells, dim)
        h_new_fdbk = output_seq[:, 1, :].view(batch_size, num_cells, dim)
        y_text = output_seq[:, 2, :].view(batch_size, num_cells, dim)
        
        # 5. Joint Bayesian Predictive Loss
        # Reshape priors to match posteriors
        lat_prior = lat_token.view(batch_size, num_cells, dim)
        fdbk_prior = fdbk_token.view(batch_size, num_cells, dim)
        
        # Calculate squared error and use .detach() as the stop-gradient operator
        loss_lat = (lat_prior - h_new_lat.detach()) ** 2
        loss_fdbk = (fdbk_prior - h_new_fdbk.detach()) ** 2
        
        # This creates a grid of shape (Batch, Cells, Dim) representing the local errors
        aux_loss_grid = loss_lat + loss_fdbk
        
        # We return all three tensors to fill our four ports
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
        
    def forward(self, x_input):
        # Convert (Batch, Seq_Len) -> (Batch, Seq_Len, Hidden_Dim)
        x_embeddings = self.embedding(x_input)
        
        # Apply dropout to the fresh embeddings
        x_embeddings = self.emb_dropout(x_embeddings)

        # The engine returns a tuple of our three grids
        output_grids = self.engine(x_embeddings)
        
        y_text_grid = output_grids[0]
        aux_loss_grid = output_grids[3]
        
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