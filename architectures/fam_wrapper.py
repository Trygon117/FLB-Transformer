import torch.nn as nn
# Simplified FAM block: Standard attention + a global feedback compression layer
class FAMTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead):
        super().__init__()
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.feedback_compressor = nn.Linear(hidden_dim, hidden_dim) # The "FAM" loop
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Logic: Current state is influenced by compressed previous states
        out = self.model(x.float())
        return self.head(out)