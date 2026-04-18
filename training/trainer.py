import torch
import torch.nn as nn
import pandas as pd
import time
import math
import os
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

def detach_states(state):
    if state is None: return None
    if isinstance(state, torch.Tensor): return state.detach()
    if isinstance(state, tuple): return tuple(detach_states(s) for s in state)
    if isinstance(state, list): return [detach_states(s) for s in state]
    if isinstance(state, dict): return {k: detach_states(v) for k, v in state.items()}
    return state

def train_universal_model(model, dataloader, optimizer, epochs, accumulation_steps=4, log_interval=50, save_interval=1000, device='cuda', log_file='training_log.csv', save_dir='checkpoints'):
    
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    log_data = []

    scaler = GradScaler('cuda')
    
    for epoch in range(epochs):
        contexts = [None] * accumulation_steps
        
        accumulated_loss = torch.tensor(0.0, device=device)
        accumulated_aux = torch.tensor(0.0, device=device)
        start_time = time.time()
        
        optimizer.zero_grad() 
        
        # --- WRAP THE DATALOADER IN TQDM ---
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            micro_step = step % accumulation_steps
            
            # Forward, Loss, Backward, State Management
            with autocast('cuda', dtype=torch.bfloat16): 
                logits, aux_loss, next_context = model(x, contexts[micro_step])
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss = loss + (aux_loss if aux_loss is not None else 0.0)
                scaled_loss = total_loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            contexts[micro_step] = detach_states(next_context)
            
            accumulated_loss += loss.detach()
            if aux_loss is not None:
                accumulated_aux += aux_loss.detach()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # --- TELEMETRY LOGGING ---
            if (step + 1) % log_interval == 0:
                avg_loss = (accumulated_loss / log_interval).item()
                avg_aux = (accumulated_aux / log_interval).item()
                perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf') 
                step_time = (time.time() - start_time) / log_interval
                
                log_data.append({
                    'epoch': epoch,
                    'step': step,
                    'loss': avg_loss,
                    'aux_loss': avg_aux,
                    'perplexity': perplexity,
                    'time_per_step': step_time
                })
                
                # --- UPDATE THE PROGRESS BAR INSTEAD OF PRINTING ---
                pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}", 
                    'PPL': f"{perplexity:.1f}",
                    's/step': f"{step_time:.2f}"
                })
                
                accumulated_loss.zero_()
                accumulated_aux.zero_()
                start_time = time.time()

            # --- CRITICAL CHECKPOINTING STEP ---
            if (step + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"model_ep{epoch}_step{step}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item() 
                }, checkpoint_path)
                # Use tqdm.write so it doesn't break the progress bar visual
                tqdm.write(f"---> Checkpoint saved to {checkpoint_path}")

        epoch_path = os.path.join(save_dir, f"model_ep{epoch}_complete.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_path)
        tqdm.write(f"===> Epoch {epoch} complete. Model saved.")

    df = pd.DataFrame(log_data)
    df.to_csv(log_file, index=False)
    return model