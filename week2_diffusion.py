# week2_diffusion.py
"""
Train a simple DDPM on Oja synthetic trajectories.
Then run the inference pipeline on diffusion-generated data.
Compare recovery quality: clean vs diffusion-noisy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_ojas_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Simple DDPM (denoising diffusion probabilistic model)
# ─────────────────────────────────────────────────────────────────────────────

class NoisePredictor(nn.Module):
    """
    Small MLP noise predictor for DDPM.
    Input:  [noisy_trajectory, timestep_embedding]  
    Output: predicted noise (same shape as trajectory)
    
    Trajectory shape: (T * n_observed,) flattened
    """
    def __init__(self, traj_dim, hidden=256, n_timesteps=1000):
        super().__init__()
        self.n_timesteps = n_timesteps
        
        # Sinusoidal timestep embedding
        self.t_embed = nn.Embedding(n_timesteps, 32)
        
        self.net = nn.Sequential(
            nn.Linear(traj_dim + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, traj_dim)
        )
    
    def forward(self, x_noisy, t):
        """
        x_noisy: (batch, traj_dim)
        t:       (batch,) integer timestep indices
        """
        t_emb = self.t_embed(t)            # (batch, 32)
        inp = torch.cat([x_noisy, t_emb], dim=-1)
        return self.net(inp)


class DDPM:
    """
    Lightweight DDPM wrapper.
    Follows Ho et al. 2020 with linear noise schedule.
    """
    def __init__(self, n_timesteps=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = n_timesteps
        self.device = device

        self.betas     = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas    = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bar[t].sqrt().view(-1, 1)
        sqrt_1ab = (1 - self.alpha_bar[t]).sqrt().view(-1, 1)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Reverse diffusion: denoise from pure noise → trajectory."""
        x = torch.randn(shape, device=self.device)
        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t_idx,
                                 dtype=torch.long, device=self.device)
            pred_noise = model(x, t_batch)
            
            beta_t = self.betas[t_idx]
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]
            
            # DDPM reverse step
            x = (1 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
            )
            if t_idx > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)
        return x


def train_ddpm(trajectories_flat, n_epochs=500, lr=3e-4, n_timesteps=200, device=DEVICE):
    """
    Train DDPM on flattened neural trajectories.

    Args:
        trajectories_flat: (n_traj, T*n_obs) — flattened Oja trajectories
    Returns:
        trained noise_predictor model
    """
    traj_dim = trajectories_flat.shape[1]
    ddpm  = DDPM(n_timesteps=n_timesteps, device=device)
    model = NoisePredictor(traj_dim, hidden=256, n_timesteps=n_timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trajectories_flat = trajectories_flat.to(device)
    
    n_traj = len(trajectories_flat)
    loss_history = []
    
    print(f"Training DDPM on {n_traj} trajectories, dim={traj_dim}")
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_traj)
        epoch_loss = 0.0
        
        # Mini-batches of 16
        for i in range(0, n_traj, 16):
            batch = trajectories_flat[perm[i:i+16]]
            t = torch.randint(0, n_timesteps, (len(batch),), device=device)
            
            optimizer.zero_grad()
            x_noisy, noise = ddpm.q_sample(batch, t)
            pred = model(x_noisy, t)
            
            loss = ((noise - pred) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss)
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}/{n_epochs} | Loss: {epoch_loss:.4f}")
    
    return model, ddpm, loss_history


def run_diffusion_experiment():
    """
    Full diffusion extension experiment (Figure 5):
    1. Generate Oja data
    2. Train DDPM on observation trajectories
    3. Sample new synthetic trajectories from DDPM
    4. Run Oja rule recovery on DDPM-generated observations (same X, synthetic O)
    5. Run Oja rule recovery on real observations for comparison
    6. Compare R² (weights and activity) between real vs DDPM
    """
    from week2_ojas_recovery import run_recovery_on_data, compute_r2

    torch.manual_seed(0)
    print(f"Using device: {DEVICE}")

    # Generate clean data (CPU) then move to device
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=100, n_output=50, T=50,
        n_trajectories=50, noise_std=0.0
    )
    X, O, W_gt, obs_idx = X.to(DEVICE), O.to(DEVICE), W_gt.to(DEVICE), obs_idx.to(DEVICE)
    n_obs = O.shape[2]  # = n_output when sparsity=1.0

    # Flatten observation trajectories for DDPM: (50, T*n_obs)
    O_flat = O.reshape(len(O), -1)

    # ── Step 2: Train DDPM ────────────────────────────────────────────────
    print("\n=== Training DDPM ===")
    ddpm_model, ddpm, _ = train_ddpm(O_flat, n_epochs=500, device=DEVICE)

    # ── Step 3: Sample from DDPM ──────────────────────────────────────────
    print("\n=== Sampling from DDPM ===")
    n_samples = len(O)  # same number of trajectories as real data
    with torch.no_grad():
        O_synthetic_flat = ddpm.p_sample_loop(ddpm_model, shape=(n_samples, O_flat.shape[1]))
    O_synthetic = O_synthetic_flat.clamp(0, 1).reshape(n_samples, 50, n_obs)

    # ── Visual check: real vs generated trajectories ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Real vs DDPM-Generated Neural Trajectories (Figure 5)", fontsize=12)
    for i in range(3):
        axes[0, i].plot(O[i, :, :5].cpu().numpy(), alpha=0.7)
        axes[0, i].set_title(f'Real trajectory {i}')
        axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].plot(O_synthetic[i, :, :5].cpu().numpy(), alpha=0.7)
        axes[1, i].set_title(f'DDPM-generated {i}')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('Real\nNeural activity')
    axes[1, 0].set_ylabel('DDPM\nNeural activity')
    for ax in axes[1]:
        ax.set_xlabel('Timestep t')
    plt.tight_layout()
    plt.savefig('week2_diffusion_trajectories.png', dpi=150)
    plt.show()
    print("Saved: week2_diffusion_trajectories.png")

    # ── Step 4: Recovery on DDPM-generated observations ───────────────────
    # Use real X (inputs) paired with synthetic O (observations from DDPM).
    # This tests whether DDPM-generated trajectories preserve enough statistical
    # structure to allow plasticity rule recovery.
    print("\n=== Step 4: Recovery on DDPM-generated data ===")
    r2w_ddpm, r2a_ddpm = run_recovery_on_data(
        X, O_synthetic, W_gt, obs_idx,
        n_epochs=400, circuit_lr=1/100, seed=1, verbose=True, device=DEVICE
    )
    print(f"  DDPM  → R²(weights)={r2w_ddpm:.3f}, R²(activity)={r2a_ddpm:.3f}")

    # ── Step 5: Recovery on real observations (baseline) ─────────────────
    print("\n=== Step 5: Recovery on real data (baseline) ===")
    r2w_real, r2a_real = run_recovery_on_data(
        X, O, W_gt, obs_idx,
        n_epochs=400, circuit_lr=1/100, seed=1, verbose=True, device=DEVICE
    )
    print(f"  Real  → R²(weights)={r2w_real:.3f}, R²(activity)={r2a_real:.3f}")

    # ── Step 6: Comparison bar chart (Figure 5) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Plasticity Recovery: Real vs DDPM Data (Figure 5)", fontsize=12)

    labels = ['Real data', 'DDPM data']
    for ax, vals, metric in zip(
        axes,
        [[r2w_real, r2w_ddpm], [r2a_real, r2a_ddpm]],
        ['R² (weights)', 'R² (neural activity)']
    ):
        bars = ax.bar(labels, vals, color=['steelblue', 'darkorange'], width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('week2_diffusion_recovery_comparison.png', dpi=150)
    plt.show()
    print("Saved: week2_diffusion_recovery_comparison.png")

    return O_synthetic, r2w_real, r2a_real, r2w_ddpm, r2a_ddpm


if __name__ == "__main__":
    O_synthetic, r2w_real, r2a_real, r2w_ddpm, r2a_ddpm = run_diffusion_experiment()
    print("\n=== Summary ===")
    print(f"Real data:  R²(weights)={r2w_real:.3f}, R²(activity)={r2a_real:.3f}")
    print(f"DDPM data:  R²(weights)={r2w_ddpm:.3f}, R²(activity)={r2a_ddpm:.3f}")