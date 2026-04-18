import torch

class ContiguousDataLoader:
    def __init__(self, tokenized_data: torch.Tensor, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len

        # We need an extra token to act as the target (shifted by 1)
        tokens_per_batch = (len(tokenized_data) - 1) // batch_size
        
        # Slice and reshape inputs
        x_data = tokenized_data[:tokens_per_batch * batch_size]
        self.x_batches = x_data.view(batch_size, tokens_per_batch)
        
        # Slice and reshape targets (shifted one step into the future)
        y_data = tokenized_data[1 : tokens_per_batch * batch_size + 1]
        self.y_batches = y_data.view(batch_size, tokens_per_batch)
        
        self.num_chunks = tokens_per_batch // seq_len

    def __iter__(self):
        for i in range(self.num_chunks):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            
            x = self.x_batches[:, start_idx:end_idx]
            y = self.y_batches[:, start_idx:end_idx]
            
            yield x, y

    def __len__(self):
        return self.num_chunks