from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn as nn

class GPTWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead, seq_len):
        super().__init__()
        # We use the standard GPT-2 configuration
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=nhead,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, x):
        outputs = self.model(x)
        # Return a 0.0 for the auxiliary loss
        return outputs.logits, 0.0