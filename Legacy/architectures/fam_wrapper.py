import torch.nn as nn
# Simplified FAM block: Standard attention + a global feedback compression layer
class FAMTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead):
        super().__init__()
        # Add the embedding dictionary to map word IDs to features
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Convert raw word IDs into dense 256-dimensional vectors
        x_emb = self.embedding(x)
        
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Pass the rich embedded vectors into the encoder
        out = self.model(x_emb, mask=causal_mask, is_causal=True)
        
        return self.head(out), 0.0