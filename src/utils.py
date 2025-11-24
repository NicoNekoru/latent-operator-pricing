import torch

def calculate_metrics(y_pred, y_true):
    # MAPE: Mean Absolute Percentage Error
    # Mask out very small values to avoid division by zero
    mask = y_true > 1e-4
    if mask.sum() == 0:
        return 0.0, 0.0

    diff = torch.abs(y_pred[mask] - y_true[mask])
    mape = torch.mean(diff / y_true[mask]) * 100.0

    # Dollar Error (assuming Index ~ 4000)
    dollar_err = torch.mean(diff) * 4000.0

    return mape.item(), dollar_err.item()
