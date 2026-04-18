from transformers import MambaConfig, MambaForCausalLM
import torch.nn as nn

class MambaWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        # Use Hugging Face's native, pure PyTorch Mamba implementation
        config = MambaConfig(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            n_layer=num_layers,
            pad_token_id=0,
        )
        self.model = MambaForCausalLM(config)

    def forward(self, x):
        # HF models return an object. We extract the logits.
        outputs = self.model(x)
        
        # Return a 0.0 for the auxiliary loss to keep the benchmarking loop happy
        return outputs.logits, 0.0