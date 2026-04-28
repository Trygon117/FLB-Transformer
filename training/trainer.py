import torch
import torch.nn as nn
import pandas as pd
import time
import math
import os
import glob
from tqdm.auto import tqdm
from torch.amp import autocast

def detach_states(states):
    """
    Recursively detaches the lateral and feedback context states from the 
    current computational graph. This prevents PyTorch from trying to 
    backpropagate through the entire history of the training run.
    """
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, (list, tuple)):
        return tuple(detach_states(s) for s in states)
    return states

def train_universal_model(model, dataloader, optimizer, epochs, accumulation_steps=4, log_interval=50, save_interval=1000, max_norm=1.0, device='cuda', log_file='artifacts/logs/flb_training_log.csv', save_dir='artifacts/checkpoints'):
    
    # 1. ENFORCED DIRECTORY CREATION
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    model.to(device)
    
    # 2. AUTO-RELOAD LOGIC
    start_epoch = 0
    # Look for the most recent "complete" epoch checkpoint
    existing_checkpoints = glob.glob(os.path.join(save_dir, "model_ep*_complete.pt"))
    if existing_checkpoints:
        # Sort by epoch number to get the latest
        latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
        print(f"--- Found existing checkpoint: {latest_checkpoint} ---")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"--- Successfully resumed from Epoch {start_epoch} ---")
        
        # ADD THIS: Load existing log data if it exists
        if os.path.exists(log_file):
            try:
                log_data = pd.read_csv(log_file).to_dict('records')
                print(f"--- Loaded {len(log_data)} existing telemetry entries ---")
            except Exception:
                log_data = []
    else:
        log_data = []
        print("--- No existing checkpoints found. Starting from scratch. ---")

    model.train()
    criterion = nn.CrossEntropyLoss()
    log_data = []

    for epoch in range(start_epoch, epochs):
        contexts = [None] * accumulation_steps
        accumulated_loss = torch.tensor(0.0, device=device)
        accumulated_aux = torch.tensor(0.0, device=device)
        start_time = time.time()
        
        optimizer.zero_grad(set_to_none=True) 
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (x, y) in enumerate(pbar):
            # Your data is already on GPU if you followed the pre-load optimization!
            micro_step = step % accumulation_steps
            
            with autocast('cuda', dtype=torch.bfloat16): 
                logits, aux_loss, next_context = model(x, contexts[micro_step])
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss = loss + (aux_loss if aux_loss is not None else 0.0)
                scaled_loss = total_loss / accumulation_steps

            scaled_loss.backward()
            contexts[micro_step] = detach_states(next_context)
            
            accumulated_loss += loss.detach()
            if aux_loss is not None:
                accumulated_aux += aux_loss.detach()
            
            # Initialize a variable to store the norm for this accumulation cycle
            current_grad_norm = 0.0 

            if (step + 1) % accumulation_steps == 0:
                # CAPTURE THE NORM HERE while gradients still exist!
                # We assign the result of clip_grad_norm_ to our variable
                current_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=max_norm
                ).item()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # --- TELEMETRY ---
            if (step + 1) % log_interval == 0:
                # Calculate metrics separately
                # Note: 'loss' here is the pure CrossEntropy (Language) loss
                avg_lang_loss = (accumulated_loss / log_interval).item()
                avg_aux_loss = (accumulated_aux / log_interval).item()
                
                # Perplexity MUST be calculated from the Language Loss only!
                perplexity = math.exp(avg_lang_loss) if avg_lang_loss < 20 else float('inf') 
                
                elapsed = time.time() - start_time
                step_time = elapsed / log_interval
                tokens_per_sec = (x.numel() * log_interval) / elapsed
                
                # Get current VRAM usage
                vram_usage = torch.cuda.memory_reserved() / (1024 ** 2) # In MB
                
                log_entry = {
                    'epoch': epoch,
                    'step': step,
                    'lang_loss': avg_lang_loss,
                    'aux_loss': avg_aux_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                    'grad_norm': current_grad_norm,
                    'vram_mb': vram_usage,
                    'perplexity': perplexity,
                    'tok_s': tokens_per_sec
                }
                log_data.append(log_entry)
                
                # Update the Progress Bar to show BOTH
                # This lets you see if the model is prioritizing language or memory
                pbar.set_postfix({
                    'Loss_lang': f"{avg_lang_loss:.3f}", 
                    'Loss_aux': f"{avg_aux_loss:.3f}",
                    'perplexity': f"{perplexity:.3f}",
                    'gnorm': f"{current_grad_norm:.2f}",
                    'tok/s': f"{tokens_per_sec:.0f}"
                })
                
                accumulated_loss.zero_()
                accumulated_aux.zero_()
                start_time = time.time()

            if (step + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"model_ep{epoch}_step{step}.pt")
                torch.save({
                    'epoch': epoch, 'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        # EPOCH COMPLETE SAVE
        epoch_path = os.path.join(save_dir, f"model_ep{epoch}_complete.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_path)
        
        # Save logs progressively after each epoch so we don't lose them if it crashes later
        pd.DataFrame(log_data).to_csv(log_file, index=False)
        tqdm.write(f"===> Epoch {epoch+1} complete. Model and Logs saved.")

    return model