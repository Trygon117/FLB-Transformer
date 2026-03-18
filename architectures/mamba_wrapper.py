from mamba_ssm import MambaLMHeadModel

class MambaWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.model = MambaLMHeadModel(
            d_model=hidden_dim, n_layer=num_layers, vocab_size=vocab_size, device='cuda'
        )

    def forward(self, x):
        return self.model(x).logits, 0.0