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
    def __init__(self, n_timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.T = n_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bar[t].sqrt().view(-1, 1)
        sqrt_1ab = (1 - self.alpha_bar[t]).sqrt().view(-1, 1)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, device='cpu'):
        """Reverse diffusion: denoise from pure noise → trajectory."""
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t_idx,
                                 dtype=torch.long, device=device)
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


def train_ddpm(trajectories_flat, n_epochs=500, lr=3e-4, n_timesteps=200):
    """
    Train DDPM on flattened neural trajectories.
    
    Args:
        trajectories_flat: (n_traj, T*n_obs) — flattened Oja trajectories
    Returns:
        trained noise_predictor model
    """
    traj_dim = trajectories_flat.shape[1]
    ddpm = DDPM(n_timesteps=n_timesteps)
    model = NoisePredictor(traj_dim, hidden=256, n_timesteps=n_timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    n_traj = len(trajectories_flat)
    loss_history = []
    
    print(f"Training DDPM on {n_traj} trajectories, dim={traj_dim}")
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_traj)
        epoch_loss = 0.0
        
        # Mini-batches of 16
        for i in range(0, n_traj, 16):
            batch = trajectories_flat[perm[i:i+16]]
            t = torch.randint(0, n_timesteps, (len(batch),))
            
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
    Full diffusion extension experiment:
    1. Generate Oja data
    2. Train DDPM on trajectories
    3. Sample new synthetic trajectories from DDPM
    4. Run Oja recovery on DDPM-generated data
    5. Compare R² to clean data recovery
    """
    from week2_ojas_recovery import run_ojas_recovery, compute_r2
    
    torch.manual_seed(0)
    
    # Generate clean data
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=100, n_output=50, T=50,
        n_trajectories=50, noise_std=0.0
    )
    
    # Flatten trajectories for DDPM: (50, 50*50) = (50, 2500)
    O_flat = O.reshape(len(O), -1)
    
    # ── Train DDPM ────────────────────────────────────────────────────────
    print("\n=== Training DDPM ===")
    ddpm_model, ddpm, _ = train_ddpm(O_flat, n_epochs=500)
    
    # ── Sample from DDPM ──────────────────────────────────────────────────
    print("\n=== Sampling from DDPM ===")
    n_samples = 20
    with torch.no_grad():
        O_synthetic_flat = ddpm.p_sample_loop(
            ddpm_model, shape=(n_samples, O_flat.shape[1])
        )
    # Clip to [0,1] since trajectories are sigmoid outputs
    O_synthetic = O_synthetic_flat.clamp(0, 1).reshape(n_samples, 50, -1)
    
    # ── Visual check: real vs generated trajectories ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Real vs DDPM-Generated Neural Trajectories (Figure 5)", fontsize=12)
    
    for i in range(3):
        # Real
        axes[0, i].plot(O[i, :, :5].numpy(), alpha=0.7)
        axes[0, i].set_title(f'Real trajectory {i}')
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_ylabel('Neural activity')
        axes[0, i].grid(True, alpha=0.3)
        
        # Generated
        axes[1, i].plot(O_synthetic[i, :, :5].numpy(), alpha=0.7)
        axes[1, i].set_title(f'DDPM-generated {i}')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_ylabel('Neural activity')
        axes[1, i].grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel('Real\nNeural activity')
    axes[1, 0].set_ylabel('DDPM\nNeural activity')
    for ax in axes[1]:
        ax.set_xlabel('Timestep t')
    
    plt.tight_layout()
    plt.savefig('week2_diffusion_trajectories.png', dpi=150)
    plt.show()
    print("Saved: week2_diffusion_trajectories.png")
    
    return O_synthetic


if __name__ == "__main__":
    O_synthetic = run_diffusion_experiment()