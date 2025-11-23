
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import NeuralOperator
from src.train import OptionDataset

def visualize_latent_space():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (Test Set)
    dataset_path = 'data/processed_dataset.parquet'
    test_dataset = OptionDataset(dataset_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load Model
    model = NeuralOperator(latent_dim=3).to(device)
    model.load_state_dict(torch.load('models/neural_operator.pth', map_location=device))
    model.eval()

    print("Extracting latent vectors...")
    all_z = []
    all_vols = []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            z = model.encoder(x)
            all_z.append(z.cpu().numpy())

            # Extract realized vol (last time step)
            # x is (Batch, 30, 2) before reshape in MLP
            # But in MLP encoder we pass flattened.
            # Wait, the dataset returns (30, 2). The model handles reshape.
            # So here x is (30, 2).
            current_vol = x[:, -1, 1].cpu().numpy()
            all_vols.append(current_vol)

    all_z = np.concatenate(all_z, axis=0)
    all_vols = np.concatenate(all_vols, axis=0)

    print(f"Extracted {len(all_z)} vectors.")

    # Create DataFrame for Plotly
    df_plot = pd.DataFrame(all_z, columns=['Latent 1', 'Latent 2', 'Latent 3'])
    df_plot['Volatility'] = all_vols

    # 1. 3D Scatter Plot (Plotly)
    print("Generating Latent Space Scatter Plot (Plotly)...")
    fig = px.scatter_3d(
        df_plot, x='Latent 1', y='Latent 2', z='Latent 3',
        color='Volatility',
        color_continuous_scale='RdBu_r', # Red=High Vol, Blue=Low
        title='Latent Space of Market Regimes',
        opacity=0.7
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Latent Dim 3'
        )
    )

    os.makedirs('plots', exist_ok=True)
    # Save as PNG (requires kaleido)
    try:
        fig.write_image("plots/latent_space_3d.png", scale=2)
    except Exception as e:
        print(f"Could not save Plotly PNG (missing kaleido?): {e}")
        fig.write_html("plots/latent_space_3d.html")
        print("Saved as HTML instead.")

    # 2. Interpolation Test
    print("Running Interpolation Test...")
    low_vol_idx = np.argmin(all_vols)
    high_vol_idx = np.argmax(all_vols)

    z_low = torch.tensor(all_z[low_vol_idx]).unsqueeze(0).to(device)
    z_high = torch.tensor(all_z[high_vol_idx]).unsqueeze(0).to(device)

    price_low_decoded = model.decoder(z_low).cpu().detach().numpy().flatten()
    price_high_decoded = model.decoder(z_high).cpu().detach().numpy().flatten()

    alphas = np.linspace(0, 1, 5)
    moneyness_levels = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    # Use Plotly for Interpolation?
    # It's a bit complex to subplot 2D with Plotly in one image easily for static export.
    # Let's stick to Matplotlib but make it nicer.
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    # Latent
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z_low + alpha * z_high
        price_interp = model.decoder(z_interp).cpu().detach().numpy().flatten()
        # Plot only first maturity (indices 0-6)
        axes[0].plot(moneyness_levels, price_interp[:7], label=f'α={alpha:.2f}', color=colors[i], linewidth=2)

    axes[0].set_title('Latent Space Interpolation (Physics-Consistent)', fontsize=12)
    axes[0].set_xlabel('Moneyness (K/S)')
    axes[0].set_ylabel('Option Price (Normalized)')
    axes[0].legend()

    # Naive
    for i, alpha in enumerate(alphas):
        price_naive = (1 - alpha) * price_low_decoded + alpha * price_high_decoded
        axes[1].plot(moneyness_levels, price_naive[:7], label=f'α={alpha:.2f}', color=colors[i], linestyle='--', linewidth=2)

    axes[1].set_title('Naive Data Space Interpolation (Linear Average)', fontsize=12)
    axes[1].set_xlabel('Moneyness (K/S)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('plots/interpolation_test.png', dpi=150)
    plt.close()

    # 3. Trajectory Analysis (Plotly)
    print("Generating Trajectory Plot (Plotly)...")
    traj_len = 200
    df_traj = df_plot.iloc[:traj_len].copy()
    df_traj['Time'] = np.arange(traj_len)

    fig = px.line_3d(
        df_traj, x='Latent 1', y='Latent 2', z='Latent 3',
        color='Time',
        title='Market Trajectory (First 200 Days)',
    )
    # Add markers
    fig.add_trace(go.Scatter3d(
        x=df_traj['Latent 1'], y=df_traj['Latent 2'], z=df_traj['Latent 3'],
        mode='markers',
        marker=dict(size=3, color=df_traj['Time'], colorscale='Viridis'),
        showlegend=False
    ))

    try:
        fig.write_image("plots/trajectory_2023.png", scale=2)
    except:
        fig.write_html("plots/trajectory_2023.html")

    # 4. Latent Physics Landscape (Heatmap)
    print("Generating Latent Physics Landscape...")
    # Grid over Latent 1 and Latent 2
    x_range = np.linspace(all_z[:, 0].min(), all_z[:, 0].max(), 50)
    y_range = np.linspace(all_z[:, 1].min(), all_z[:, 1].max(), 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Fix Latent 3 to mean
    z3_mean = all_z[:, 2].mean()

    # Decode grid
    grid_points = np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), z3_mean)], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        decoded_prices = model.decoder(grid_tensor).cpu().numpy()

    # Extract ATM Price (Moneyness 1.0, Maturity 1m -> Index 3)
    atm_prices = decoded_prices[:, 3].reshape(50, 50)

    fig = go.Figure(data=go.Contour(
        z=atm_prices,
        x=x_range,
        y=y_range,
        colorscale='Viridis',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
        )
    ))
    fig.update_layout(
        title='Latent Physics Landscape (ATM Call Price)',
        xaxis_title='Latent Dim 1',
        yaxis_title='Latent Dim 2',
    )
    try:
        fig.write_image("plots/latent_landscape.png", scale=2)
    except:
        fig.write_html("plots/latent_landscape.html")

    # 5. Latent Velocity Field (Quiver)
    print("Generating Latent Velocity Field...")
    # Calculate velocity vectors
    dz = all_z[1:] - all_z[:-1]
    z_curr = all_z[:-1]

    # Bin the space to get average flow
    # Use 20x20 bins
    x_bins = np.linspace(all_z[:, 0].min(), all_z[:, 0].max(), 20)
    y_bins = np.linspace(all_z[:, 1].min(), all_z[:, 1].max(), 20)

    u_grid = np.zeros((19, 19))
    v_grid = np.zeros((19, 19))
    count_grid = np.zeros((19, 19))

    for i in range(len(z_curr)):
        x_idx = np.digitize(z_curr[i, 0], x_bins) - 1
        y_idx = np.digitize(z_curr[i, 1], y_bins) - 1

        if 0 <= x_idx < 19 and 0 <= y_idx < 19:
            u_grid[y_idx, x_idx] += dz[i, 0]
            v_grid[y_idx, x_idx] += dz[i, 1]
            count_grid[y_idx, x_idx] += 1

    # Average
    mask = count_grid > 0
    u_grid[mask] /= count_grid[mask]
    v_grid[mask] /= count_grid[mask]

    # Plot Quiver using Plotly Figure Factory
    import plotly.figure_factory as ff

    # Meshgrid for centers
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    X_q, Y_q = np.meshgrid(x_centers, y_centers)

    fig = ff.create_quiver(X_q.flatten(), Y_q.flatten(), u_grid.flatten(), v_grid.flatten(),
                           scale=0.5,
                           arrow_scale=0.3,
                           name='Flow',
                           line=dict(width=1))

    fig.update_layout(
        title='Latent Velocity Field (Market Flow Dynamics)',
        xaxis_title='Latent Dim 1',
        yaxis_title='Latent Dim 2',
    )

    try:
        fig.write_image("plots/latent_velocity.png", scale=2)
    except:
        fig.write_html("plots/latent_velocity.html")

    print("Analysis Complete. Plots saved to plots/")

if __name__ == "__main__":
    visualize_latent_space()
