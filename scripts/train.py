import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import json

from src.models import NeuralOperator
from src.dataset import OptionDataset
from src.utils import calculate_metrics

class PhysicsInformedLoss(nn.Module):
    def __init__(self, arbitrage_weight=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.arbitrage_weight = arbitrage_weight

    def forward(self, y_pred, y_true):
        # MSLE Loss (Mean Squared Log Error)
        log_pred = torch.log(y_pred + 1e-6)
        log_true = torch.log(y_true + 1e-6)

        msle_loss = torch.mean((log_pred - log_true) ** 2)

        # Arbitrage Penalty
        y_reshaped = y_pred.view(-1, 3, 7)
        diffs = y_reshaped[:, :, 1:] - y_reshaped[:, :, :-1]
        penalty = torch.relu(diffs).mean()

        total_loss = msle_loss + self.arbitrage_weight * penalty
        return total_loss, msle_loss, penalty

def train_model(epochs=200, batch_size=32, lr=1e-3, latent_dim=3, minimal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = 'data/processed_dataset.parquet'

    features_to_keep = [0, 1] if minimal else None
    input_dim = 2 if minimal else 6
    model_name = 'neural_operator_minimal.pth' if minimal else 'neural_operator.pth'

    print(f"Training with Input Dim: {input_dim} (Minimal: {minimal})")

    train_dataset = OptionDataset(dataset_path, mode='train', features_to_keep=features_to_keep)
    val_dataset = OptionDataset(dataset_path, mode='val', features_to_keep=features_to_keep)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Explicitly pass input_dim
    model = NeuralOperator(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = PhysicsInformedLoss(arbitrage_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training for {epochs} epochs...")

    history = {'train_loss': [], 'val_loss': [], 'train_mape': [], 'val_mape': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_msle = 0.0
        train_arb = 0.0
        train_mape = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred, _ = model(x)

            loss, msle, arb = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            mape, _ = calculate_metrics(y_pred, y)

            train_loss += loss.item()
            train_msle += msle.item()
            train_arb += arb.item()
            train_mape += mape

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        val_dollar = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = model(x)
                loss, msle, arb = criterion(y_pred, y)
                mape, dollar = calculate_metrics(y_pred, y)

                val_loss += loss.item()
                val_mape += mape
                val_dollar += dollar

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mape = train_mape / len(train_loader)
        avg_val_mape = val_mape / len(val_loader)
        avg_val_dollar = val_dollar / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mape'].append(avg_train_mape)
        history['val_mape'].append(avg_val_mape)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val MAPE: {avg_val_mape:.2f}% (${avg_val_dollar:.2f})")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/neural_operator.pth')
    print("Model saved to models/neural_operator.pth")

    # Save History
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to models/training_history.json")

    # Plot Training History
    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Linear Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss (Linear)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Linear MAPE
    axes[0, 1].plot(history['train_mape'], label='Train MAPE')
    axes[0, 1].plot(history['val_mape'], label='Val MAPE')
    axes[0, 1].set_title('MAPE (Linear)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Log Loss
    axes[1, 0].plot(history['train_loss'], label='Train Loss')
    axes[1, 0].plot(history['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Model Loss (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Log Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Log MAPE
    axes[1, 1].plot(history['train_mape'], label='Train MAPE')
    axes[1, 1].plot(history['val_mape'], label='Val MAPE')
    axes[1, 1].set_title('MAPE (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Log MAPE (%)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('plots/training_loss.png', dpi=300)
    print("Training plots saved to plots/training_loss.png")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import json

from src.models import NeuralOperator
from src.dataset import OptionDataset
from src.utils import calculate_metrics

class PhysicsInformedLoss(nn.Module):
    def __init__(self, arbitrage_weight=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.arbitrage_weight = arbitrage_weight

    def forward(self, y_pred, y_true):
        # MSLE Loss (Mean Squared Log Error)
        log_pred = torch.log(y_pred + 1e-6)
        log_true = torch.log(y_true + 1e-6)

        msle_loss = torch.mean((log_pred - log_true) ** 2)

        # Arbitrage Penalty
        y_reshaped = y_pred.view(-1, 3, 7)
        diffs = y_reshaped[:, :, 1:] - y_reshaped[:, :, :-1]
        penalty = torch.relu(diffs).mean()

        total_loss = msle_loss + self.arbitrage_weight * penalty
        return total_loss, msle_loss, penalty

def train_model(epochs=200, batch_size=32, lr=1e-3, latent_dim=3, minimal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = 'data/processed_dataset.parquet'

    features_to_keep = [0, 1] if minimal else None
    input_dim = 2 if minimal else 6
    model_name = 'neural_operator_minimal.pth' if minimal else 'neural_operator.pth'

    print(f"Training with Input Dim: {input_dim} (Minimal: {minimal})")

    train_dataset = OptionDataset(dataset_path, mode='train', features_to_keep=features_to_keep)
    val_dataset = OptionDataset(dataset_path, mode='val', features_to_keep=features_to_keep)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Explicitly pass input_dim
    model = NeuralOperator(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = PhysicsInformedLoss(arbitrage_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training for {epochs} epochs...")

    history = {'train_loss': [], 'val_loss': [], 'train_mape': [], 'val_mape': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_msle = 0.0
        train_arb = 0.0
        train_mape = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred, _ = model(x)

            loss, msle, arb = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            mape, _ = calculate_metrics(y_pred, y)

            train_loss += loss.item()
            train_msle += msle.item()
            train_arb += arb.item()
            train_mape += mape

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        val_dollar = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = model(x)
                loss, msle, arb = criterion(y_pred, y)
                mape, dollar = calculate_metrics(y_pred, y)

                val_loss += loss.item()
                val_mape += mape
                val_dollar += dollar

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mape = train_mape / len(train_loader)
        avg_val_mape = val_mape / len(val_loader)
        avg_val_dollar = val_dollar / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mape'].append(avg_train_mape)
        history['val_mape'].append(avg_val_mape)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val MAPE: {avg_val_mape:.2f}% (${avg_val_dollar:.2f})")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}')
    print(f"Model saved to models/{model_name}")

    # Save History
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to models/training_history.json")

    # Plot Training History
    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Linear Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss (Linear)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Linear MAPE
    axes[0, 1].plot(history['train_mape'], label='Train MAPE')
    axes[0, 1].plot(history['val_mape'], label='Val MAPE')
    axes[0, 1].set_title('MAPE (Linear)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Log Loss
    axes[1, 0].plot(history['train_loss'], label='Train Loss')
    axes[1, 0].plot(history['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Model Loss (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Log Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Log MAPE
    axes[1, 1].plot(history['train_mape'], label='Train MAPE')
    axes[1, 1].plot(history['val_mape'], label='Val MAPE')
    axes[1, 1].set_title('MAPE (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Log MAPE (%)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('plots/training_loss.png', dpi=300)
    print("Training plots saved to plots/training_loss.png")

    print(f"\nFinal Test Set Performance:")
    print(f"MAPE: {history['val_mape'][-1]:.2f}%")
    print(f"Loss: {history['val_loss'][-1]:.6f}")

if __name__ == "__main__":
    train_model()
