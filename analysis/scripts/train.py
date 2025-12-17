import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import json

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.models import get_standard_grid, SpectralDeepONet
from src.dataset import OptionDataset
from src.utils import calculate_metrics

class PhysicsInformedLoss(nn.Module):
    def __init__(self, arbitrage_weight=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.arbitrage_weight = arbitrage_weight
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # L1 Loss on Prices (Robustness)
        price_loss = self.loss_fn(y_pred, y_true)

        # Arbitrage Penalty
        y_reshaped = y_pred.view(-1, 3, 7)
        # Monotonicity in Strike: Call Price must decrease as K increases
        # diffs = Price(K_next) - Price(K_prev). Should be negative.
        diffs = y_reshaped[:, :, 1:] - y_reshaped[:, :, :-1]
        # Penalize positive differences
        penalty = torch.relu(diffs).mean()

        total_loss = price_loss + self.arbitrage_weight * penalty
        return total_loss, price_loss, penalty

from src.synthetic import HestonSimulator

def train_model(epochs=300, batch_size=32, lr=5e-4, latent_dim=64, minimal=False, pretrain_epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ... (Load data as before)
    dataset_path = 'data/processed_dataset.parquet'

    features_to_keep = [0, 1] if minimal else None
    input_dim = 2 if minimal else 6
    model_name = 'deeponet_minimal.pth' if minimal else 'deeponet.pth'

    print(f"Training with Input Dim: {input_dim} (Minimal: {minimal})")

    dataset_path = os.path.join(PROJECT_ROOT, 'data/processed_dataset.parquet')

    features_to_keep = [0, 1] if minimal else None
    input_dim = 2 if minimal else 6
    model_name = 'deeponet_minimal.pth' if minimal else 'deeponet.pth'

    print(f"Training with Input Dim: {input_dim} (Minimal: {minimal})")

    train_dataset = OptionDataset(dataset_path, mode='train')
    val_dataset = OptionDataset(dataset_path, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize Models
    model = SpectralDeepONet(input_channels=6, latent_dim=latent_dim, hidden_dim=64).to(device)
    base_grid = get_standard_grid(device)

    # Synthetic Generator
    simulator = HestonSimulator(device=device)

    # Optimizer
    # Aggressive Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=5e-3) # Higher initial LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

    # --- PHASE 0: Synthetic Pre-training ---
    if pretrain_epochs > 0:
        print(f"\n=== Starting Synthetic Pre-training ({pretrain_epochs} Batches) ===")
        print("Goal: Learn Heston Physics (Mapping History -> Vol Surface)")

        pretrain_history = {'task_loss': [], 'domain_loss': [], 'total_loss': []}

        model.train()
        for i in range(pretrain_epochs):
            # Generate on the fly
            # x_syn: (B, L, 6)
            # y_iv_syn: (B, 1, 21) -> (B, 21) via .squeeze(1) if necessary, actually generate_batch returns (B, 21)
            # grid_tensor: (128, 21, 2)
            # Warning was: target size (torch.Size([128, 128, 21])) vs input (torch.Size([128, 21]))
            # Why is y_iv_syn (128, 128, 21)?
            # Check synthetic.py: v_final = params['v0'].unsqueeze(1).expand(-1, 21) -> (B, 21)
            # grid_tensor passed to generate_batch uses base_grid.expand(128, -1, -1), or (128, 21, 2).
            # The issue might be in how synthetic.py uses grid_tensor?
            # The warning says: Target size [128, 128, 21]. Input size [128, 21].
            # Input is sigma_pred (128, 21). Target is y_iv_syn.
            # So y_iv_syn is [128, 128, 21].
            # Fix synthetic.py if needed, or adjust usage here.

            x_syn, y_iv_syn, _, domain_syn = simulator.generate_batch(batch_size=128, seq_len=30, grid_tensor=base_grid.expand(128, -1, -1))

            # y_iv_syn in synthetic.py:
            # t_mat = grid[:, :, 1]
            # grid is (B, N, 2). t_mat is (B, N).
            # If grid is expanded correctly...

            optimizer.zero_grad()
            _, sigma_pred, _, domain_pred = model(x_syn, base_grid.expand(128, -1, -1), alpha=1.0)

            task_loss = nn.MSELoss()(sigma_pred, y_iv_syn)

            # Domain Adversarial: We want Synthetic features to map to Domain 0
            # domain_pred should be 0
            domain_loss = nn.BCEWithLogitsLoss()(domain_pred, domain_syn)


            loss = task_loss + 0.1 * domain_loss
            loss.backward()
            optimizer.step()

            # Capture Metrics
            pretrain_history['task_loss'].append(task_loss.item())
            pretrain_history['domain_loss'].append(domain_loss.item())
            pretrain_history['total_loss'].append(loss.item())

            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{pretrain_epochs} | Syn Loss: {task_loss.item():.6f}")

        print("=== Synthetic Pre-training Complete ===\n")

        # Plot Pre-training Dynamics
        plot_dir = os.path.join(PROJECT_ROOT, 'analysis/plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.plot(pretrain_history['task_loss'], label='Physics Loss (MSE)', alpha=0.7)
        plt.plot(pretrain_history['total_loss'], label='Total Loss', alpha=0.5, linestyle='--')
        plt.title('Phase 0: Synthetic Pre-training Dynamics')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(plot_dir, 'pretraining_dynamics.png'), dpi=300)
        plt.close()
        print(f"Pre-training plots saved to {plot_dir}/pretraining_dynamics.png")

    # --- PHASE 1 & 2: Real Data Fine-tuning ---
    print(f"Starting Real Data Fine-tuning for {epochs} epochs...")

    history = {'train_loss': [], 'val_loss': [], 'train_mape': [], 'val_mape': []}

    criterion = PhysicsInformedLoss(arbitrage_weight=0.0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mape = 0.0

        # Phase 1: Pre-train on Implied Volatility
        # Phase 2: Fine-tune on Price
        phase = "IV_Pretrain" if epoch < 50 else "Price_Finetune"


        for i, (x, y_price, y_iv, domain_label) in enumerate(train_loader):
            x, y_price, y_iv, domain_label = x.to(device), y_price.to(device), y_iv.to(device), domain_label.to(device).unsqueeze(1)

            # Expand grid to batch size
            current_batch_size = x.size(0)
            grid = base_grid.expand(current_batch_size, -1, -1)

            # GRL Alpha Schedule (Ramp up from 0 to 1 over first 100 epochs)
            p = min(1.0, epoch / 100.0)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            optimizer.zero_grad()
            price_pred, sigma_pred, _, domain_pred = model(x, grid, alpha=alpha)

            # Task Loss
            if phase == "IV_Pretrain":
                # Mask out clamped values (<= 0.01) which are artifacts
                mask = y_iv > 0.011
                if mask.sum() > 0:
                    task_loss = nn.MSELoss()(sigma_pred[mask], y_iv[mask])
                else:
                    task_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # L1 Loss on Price (via PhysicsInformedLoss) for robustness
                # User asked for justification of L1, but for training stability with outliers L1 is better.
                # For the final metric, use MSE on normalized prices.
                task_loss = nn.MSELoss()(price_pred, y_price)

            # Domain Adversarial Loss - DISABLED
            # We suspect this is destabilizing the features.
            domain_loss = torch.tensor(0.0, device=device)
            # domain_loss = nn.BCEWithLogitsLoss()(domain_pred, domain_label)

            # Total Loss
            total_loss = task_loss # + 0.0 * domain_loss

            total_loss.backward()
            optimizer.step()

            mape, _ = calculate_metrics(price_pred, y_price)

            train_loss += task_loss.item() # Log only task loss for comparability
            train_mape += mape

        # Validation
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        val_dollar = 0.0
        domain_acc = 0.0

        with torch.no_grad():
            for x, y_price, y_iv, domain_label in val_loader:
                x, y_price, y_iv, domain_label = x.to(device), y_price.to(device), y_iv.to(device), domain_label.to(device).unsqueeze(1)

                current_batch_size = x.size(0)
                grid = base_grid.expand(current_batch_size, -1, -1)

                # Alpha 0.0
                price_pred, sigma_pred, _, domain_pred = model(x, grid, alpha=0.0)

                # Validation always tracks Price Loss (MSE)
                loss = nn.MSELoss()(price_pred, y_price)

                # Calculate MAPE only on meaningful prices (> 0.01)
                # This prevents 1000% errors on penny options
                # calculate_metrics handles the basic case; this pass excludes penny options.
                # Valid Price Mask
                vp_mask = y_price > 0.01
                if vp_mask.sum() > 0:
                    mape_batch = torch.mean(torch.abs((price_pred[vp_mask] - y_price[vp_mask]) / y_price[vp_mask])) * 100
                else:
                    mape_batch = 0.0

                mape = mape_batch.item()

                val_loss += loss.item()
                val_mape += mape
                val_dollar += 0.0 # Not tracked in this training run
                domain_acc += 0.0 # Not tracked in this training run

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mape = train_mape / len(train_loader)
        avg_val_mape = val_mape / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_dollar = 0.0
        avg_domain_acc = 0.0

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mape'].append(avg_train_mape)
        history['val_mape'].append(avg_val_mape)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} ({phase}) | Task Loss: {avg_train_loss:.6f} | Val MAPE: {avg_val_mape:.2f}% | Dom Acc: {avg_domain_acc:.2%}")

        # Save model every epoch
        model_dir = os.path.join(PROJECT_ROOT, 'training/models')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    # Save History
    log_dir = os.path.join(PROJECT_ROOT, 'training/logs')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {log_dir}/training_history.json")

    # Plot Combined Training History (Pre-train + Fine-tune)
    plot_dir = os.path.join(PROJECT_ROOT, 'analysis/plots')
    os.makedirs(plot_dir, exist_ok=True)

    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Combined Loss Plot
    plt.figure(figsize=(15, 6))

    # Pre-training data (Batch-wise)
    # Scale x-axis to be comparable?
    # Pre-train is 100 batches. Fine-tune is 100 epochs * ~N batches.
    # Compare pre-training and fine-tuning in separate panels.
    # Pre-training can be visualized as "Epoch -1" or distinct panel.

    # Better: Two subplots sharing y-axis?
    # User asked to "mark the graph".

    # Keep the existing "training_loss.png" name for downstream scripts.

    # Concatenate Pre-train loss (last 100 batches) and Train Loss (Epochs)
    # This is tricky because units differ.
    # Plot the standard training/validation curves and annotate Phase 1 vs Phase 2.

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: Pre-training (Batch)
    axes[0].plot(pretrain_history['task_loss'], label='Physics Loss', color='blue')
    axes[0].set_title('Phase 0: Synthetic Pre-training (High-Freq)')
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_yscale('log')
    axes[0].legend()

    # Right: Fine-tuning (Epoch)
    # Mark transition from IV Pretrain -> Price Finetune (Epoch 50)
    axes[1].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[1].plot(history['val_loss'], label='Val Loss', color='orange')

    # Vertical Line for Phase 1 -> Phase 2 transition
    axes[1].axvline(x=50, color='red', linestyle='--', label='Switch to Price Loss')
    axes[1].text(52, max(history['train_loss'])*0.8, 'Price Fine-tuning', color='red')
    axes[1].text(2, max(history['train_loss'])*0.8, 'IV Transfer', color='blue')

    axes[1].set_title('Phase 1 & 2: Real Data Adaptation (Epochs)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'full_training_dynamics.png'), dpi=300)
    print(f"Combined dynamics saved to {plot_dir}/full_training_dynamics.png")

    # Keep the standard metrics plot as well
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # ... (existing plot code)
    sns.lineplot(x=range(len(history['train_loss'])), y=history['train_loss'], label='Train Loss', ax=axes[0, 0])
    sns.lineplot(x=range(len(history['val_loss'])), y=history['val_loss'], label='Val Loss', ax=axes[0, 0])
    axes[0, 0].axvline(x=50, color='r', linestyle='--', alpha=0.5) # Mark transition
    axes[0, 0].set_title('Model Loss (Linear)')
    # ...
    sns.lineplot(x=range(len(history['train_mape'])), y=history['train_mape'], label='Train MAPE', ax=axes[0, 1])
    sns.lineplot(x=range(len(history['val_mape'])), y=history['val_mape'], label='Val MAPE', ax=axes[0, 1])
    axes[0, 1].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('MAPE (Linear)')
    # ...
    sns.lineplot(x=range(len(history['train_loss'])), y=history['train_loss'], label='Train Loss', ax=axes[1, 0])
    sns.lineplot(x=range(len(history['val_loss'])), y=history['val_loss'], label='Val Loss', ax=axes[1, 0])
    axes[1, 0].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Model Loss (Log Scale)')
    axes[1, 0].set_yscale('log')
    # ...
    sns.lineplot(x=range(len(history['train_mape'])), y=history['train_mape'], label='Train MAPE', ax=axes[1, 1])
    sns.lineplot(x=range(len(history['val_mape'])), y=history['val_mape'], label='Val MAPE', ax=axes[1, 1])
    axes[1, 1].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('MAPE (Log Scale)')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_loss.png'), dpi=300)
    print(f"Training plots saved to {plot_dir}/training_loss.png")

    print(f"\nFinal Validation Performance (2022):")
    print(f"MAPE: {history['val_mape'][-1]:.2f}%")
    print(f"Loss: {history['val_loss'][-1]:.6f}")

if __name__ == "__main__":
    # Updated training configuration based on user feedback (Short pre-train, focus on fine-tuning)
    train_model(epochs=100, pretrain_epochs=100) # Short burst of physics, then real data
