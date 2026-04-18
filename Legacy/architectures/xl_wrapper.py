from transformers import TransfoXLConfig, TransfoXLLMHeadModel
import torch.nn as nn

class XLWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead):
        super().__init__()
        config = TransfoXLConfig(
            vocab_size=vocab_size, 
            d_model=hidden_dim, 
            n_layer=num_layers, 
            n_head=nhead,
            cutoffs=[],  # Disable adaptive embedding chunks
            div_val=1    # Force standard embedding math
        )
        self.model = TransfoXLLMHeadModel(config)
        self.mems = None

    def forward(self, x):
        outputs = self.model(x, mems=self.mems)
        
        # Detach the memory to prevent an infinite backpropagation loop
        if outputs.mems is not None:
            self.mems = [mem.detach() for mem in outputs.mems]
            
        return outputs.logits, 0.0
        
    def reset_memory(self):
        self.mems = None