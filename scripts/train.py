import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..src.models import NeuralOperator
from ..src.dataset import OptionDataset
from ..src.utils import calculate_metrics

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

def train_model(epochs=100, batch_size=32, lr=1e-3, latent_dim=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = 'data/processed_dataset.parquet'
    train_dataset = OptionDataset(dataset_path, mode='train')
    val_dataset = OptionDataset(dataset_path, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Explicitly pass input_dim=6
    model = NeuralOperator(input_dim=6, latent_dim=latent_dim).to(device)
    criterion = PhysicsInformedLoss(arbitrage_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training for {epochs} epochs...")

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

        avg_train_mape = train_mape / len(train_loader)
        avg_val_mape = val_mape / len(val_loader)
        avg_val_dollar = val_dollar / len(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train MSLE: {train_msle/len(train_loader):.6f} | MAPE: {avg_train_mape:.2f}% | Val MAPE: {avg_val_mape:.2f}% (${avg_val_dollar:.2f})")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/neural_operator.pth')
    print("Model saved to models/neural_operator.pth")

if __name__ == "__main__":
    train_model()
