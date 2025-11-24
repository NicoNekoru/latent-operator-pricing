import torch
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import NeuralOperator
from src.dataset import OptionDataset
from src.utils import calculate_metrics

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (Test Set: 2024+)
    dataset_path = 'data/processed_dataset.parquet'
    test_dataset = OptionDataset(dataset_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test Set Size: {len(test_dataset)} samples")

    # Load Model
    model = NeuralOperator(latent_dim=3).to(device)
    try:
        model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    except FileNotFoundError:
        print("Model not found. Please train first.")
        return
    model.eval()

    total_mape = 0.0
    total_dollar = 0.0
    count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)

            mape, dollar = calculate_metrics(y_pred, y)

            batch_size = x.size(0)
            total_mape += mape * batch_size
            total_dollar += dollar * batch_size
            count += batch_size

    avg_mape = total_mape / count
    avg_dollar = total_dollar / count

    print(f"\nFinal Test Results (2024-2025):")
    print(f"MAPE: {avg_mape:.4f}%")
    print(f"Dollar Error (Index=4000): ${avg_dollar:.4f}")

    if avg_mape < 1.0:
        print("SUCCESS: Model meets high-precision criteria.")
    else:
        print("WARNING: Model precision is still low.")

if __name__ == "__main__":
    evaluate()
