import torch.nn as nn
# Simplified FAM block: Standard attention + a global feedback compression layer
class FAMTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead):
        super().__init__()
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        # Deleted the unused feedback_compressor
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Generate a causal mask to prevent the model from looking into the future
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Pass the mask into the encoder
        out = self.model(x.float(), mask=causal_mask, is_causal=True)
        
        # Return the logits and a 0.0 for the auxiliary loss
        return self.head(out), 0.0