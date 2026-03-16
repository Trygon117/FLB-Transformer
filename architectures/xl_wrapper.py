from transformers import TransfoXLConfig, TransfoXLLMHeadModel

class XLWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, nhead):
        super().__init__()
        config = TransfoXLConfig(
            vocab_size=vocab_size, d_model=hidden_dim, 
            n_layer=num_layers, n_head=nhead
        )
        self.model = TransfoXLLMHeadModel(config)
        self.mems = None

    def forward(self, x):
        # We pass the memory from the previous step to the current one
        outputs = self.model(x, mems=self.mems)
        self.mems = outputs.mems 
        return outputs.logits
        
    def reset_memory(self):
        self.mems = None