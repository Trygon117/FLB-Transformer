import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_metrics(log_file='training_log.csv', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(log_file)
    df['global_step'] = df.index  # Continuous x-axis across epochs
    
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Main Loss & Aux Loss ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='global_step', y='lang_loss', label='Language Loss')
    sns.lineplot(data=df, x='global_step', y='aux_loss', label='Auxiliary Loss')
    plt.title('Training Loss over Time')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300)
    plt.close()
    
    # --- Plot 2: Perplexity ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='global_step', y='perplexity', color='purple')
    plt.title('Model Perplexity (Language Modeling Quality)')
    plt.xlabel('Global Step')
    plt.ylabel('Perplexity')
    # Cap the Y-axis to handle massive initial spikes
    plt.ylim(0, df['perplexity'].quantile(0.95) * 1.5) 
    plt.savefig(f"{output_dir}/perplexity_curve.png", dpi=300)
    plt.close()
    
    # --- Plot 3: Hardware Efficiency ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='global_step', y='step_time', color='green')
    plt.title('Wall-clock Time per Step')
    plt.xlabel('Global Step')
    plt.ylabel('Seconds')
    plt.savefig(f"{output_dir}/step_time.png", dpi=300)
    plt.close()
    
    print(f"Artifacts successfully generated and saved to ./{output_dir}/")