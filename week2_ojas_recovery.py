# week2_ojas_recovery.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_generation import generate_ojas_data
from plasticity_rules import TaylorPlasticityRule
from circuit_model import CircuitModel
from training import mse_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Full Oja's Recovery Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ojas_recovery(
    n_input=100, n_output=1000, T=50,
    n_trajectories=50, n_epochs=400,
    noise_std=0.0, sparsity=1.0,
    lr_optimizer=1e-3, circuit_lr=1/100,
    grad_clip=0.2, l1_lambda=0.0,
    seed=42, verbose=True, device=DEVICE
):
    """
    Full Oja's rule recovery experiment.
    Reproduces Figure 2 from Mehta et al. 2024.
    
    Returns loss history, theta history, and R² on held-out test weights.
    """
    torch.manual_seed(seed)
    
    # ── Generate data (CPU) then move to device ────────────────────────────
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=n_input, n_output=n_output, T=T,
        n_trajectories=n_trajectories,
        noise_std=noise_std, sparsity=sparsity,
        lr=circuit_lr, seed=seed
    )
    X, O, W_gt, obs_idx = X.to(device), O.to(device), W_gt.to(device), obs_idx.to(device)

    # Train/test split (use last 10 trajectories as test set)
    n_test = 10
    X_train, O_train = X[:-n_test], O[:-n_test]
    X_test,  O_test  = X[-n_test:], O[-n_test:]
    W_inits_train = W_gt[:-n_test, 0]   # initial weights for training trajs
    W_inits_test  = W_gt[-n_test:, 0]   # initial weights for test trajs
    W_gt_train    = W_gt[:-n_test]       # full weight trajectories for training set
    W_gt_test     = W_gt[-n_test:]      # full weight trajectories for R²

    # Weight error tracking for Fig 2B heatmap (sampled every ~n_epochs/20 epochs)
    checkpoint_freq = max(1, n_epochs // 20)
    weight_error_grid = []
    weight_error_epochs = []

    # ── Train ─────────────────────────────────────────────────────────────
    rule = TaylorPlasticityRule(max_order=2, include_reward=False).to(device)
    circuit = CircuitModel(n_input, n_output, rule, lr=circuit_lr).to(device)
    optimizer = optim.Adam(rule.parameters(), lr=lr_optimizer)
    
    loss_history = []
    theta_history = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(len(X_train))
        
        for i in perm:
            optimizer.zero_grad()
            m = circuit.forward(X_train[i], W_init=W_inits_train[i],
                                observed_idx=obs_idx)
            loss = mse_loss(m, O_train[i])
            if l1_lambda > 0:
                loss = loss + l1_lambda * rule.theta.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rule.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss / len(X_train))
        theta_history.append(rule.theta.detach().clone())
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss_history[-1]:.6f}")

        # Track mean weight error per timestep (for Fig 2B heatmap)
        if epoch % checkpoint_freq == 0:
            with torch.no_grad():
                w_err = torch.zeros(T + 1, device=device)
                for i in range(len(X_train)):
                    _, W_pred = circuit_forward_with_weights(
                        circuit, X_train[i], W_inits_train[i]
                    )
                    w_err += ((W_pred - W_gt_train[i]) ** 2).mean(dim=(1, 2))
                weight_error_grid.append((w_err / len(X_train)).cpu().numpy())
                weight_error_epochs.append(epoch)
    
    # ── Evaluate: R² on test weight trajectories AND neural activity ───────
    rule.eval()
    with torch.no_grad():
        r2_weight_scores = []
        r2_activity_scores = []
        for i in range(n_test):
            m_pred, W_pred = circuit_forward_with_weights(
                circuit, X_test[i], W_inits_test[i]
            )
            # R² on weights
            r2_weight_scores.append(compute_r2(W_pred, W_gt_test[i]))
            # R² on neural activity (apply obs_idx to match O_test shape)
            m_obs = m_pred[:, obs_idx]  # (T, n_observed)
            r2_activity_scores.append(compute_r2(m_obs, O_test[i]))

    mean_r2_weights   = np.mean(r2_weight_scores)
    mean_r2_activity  = np.mean(r2_activity_scores)
    if verbose:
        print(f"\nMean R² on test weight trajectories: {mean_r2_weights:.4f}")
        print(f"Mean R² on test neural activity:     {mean_r2_activity:.4f}")

    weight_error_grid = np.stack(weight_error_grid)  # (n_checkpoints, T+1)
    return (loss_history, theta_history, mean_r2_weights,
            rule, weight_error_grid, weight_error_epochs, mean_r2_activity)


def run_recovery_on_data(X, O, W_gt, obs_idx,
                         n_epochs=400, circuit_lr=1/100,
                         lr_optimizer=1e-3, grad_clip=0.2,
                         seed=42, verbose=False, device=DEVICE):
    """
    Run plasticity rule recovery on pre-supplied data tensors.
    Used by the DDPM comparison experiment (week2_diffusion.py).

    Returns:
        mean_r2_weights:  float — R² on held-out weight trajectories
        mean_r2_activity: float — R² on held-out neural activity
    """
    torch.manual_seed(seed)
    n_traj   = X.shape[0]
    n_input  = X.shape[2]
    n_output = W_gt.shape[2]     # (n_traj, T+1, n_output, n_input)

    # Move pre-supplied tensors to device
    X, O, W_gt = X.to(device), O.to(device), W_gt.to(device)
    obs_idx = obs_idx.to(device)

    n_test = max(1, n_traj // 5)
    X_train, O_train = X[:-n_test], O[:-n_test]
    X_test,  O_test  = X[-n_test:], O[-n_test:]
    W_inits_train = W_gt[:-n_test, 0]
    W_inits_test  = W_gt[-n_test:, 0]
    W_gt_test     = W_gt[-n_test:]

    rule    = TaylorPlasticityRule(max_order=2, include_reward=False).to(device)
    circuit = CircuitModel(n_input, n_output, rule, lr=circuit_lr).to(device)
    optimizer = optim.Adam(rule.parameters(), lr=lr_optimizer)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_train))
        for i in perm:
            optimizer.zero_grad()
            m = circuit.forward(X_train[i], W_init=W_inits_train[i],
                                observed_idx=obs_idx)
            loss = mse_loss(m, O_train[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rule.parameters(), grad_clip)
            optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch}/{n_epochs}")

    rule.eval()
    with torch.no_grad():
        r2_w_scores, r2_a_scores = [], []
        for i in range(n_test):
            m_pred, W_pred = circuit_forward_with_weights(circuit, X_test[i], W_inits_test[i])
            r2_w_scores.append(compute_r2(W_pred, W_gt_test[i]))
            m_obs = m_pred[:, obs_idx]
            r2_a_scores.append(compute_r2(m_obs, O_test[i]))

    return np.mean(r2_w_scores), np.mean(r2_a_scores)


def circuit_forward_with_weights(circuit, X, W_init):
    """Run circuit and also return full weight trajectory (for R² computation)."""
    T = X.shape[0]
    W = W_init.clone()
    m_traj, W_traj = [], [W.clone()]
    
    for t in range(T):
        x = X[t]
        y = torch.sigmoid(W @ x)
        m_traj.append(y)
        dW = circuit.plasticity_rule(x, y, W, r=None)
        W = W + circuit.lr * dW
        W_traj.append(W.clone())
    
    return torch.stack(m_traj), torch.stack(W_traj)


def compute_r2(W_pred, W_true):
    """R² between predicted and ground-truth weight trajectories."""
    W_pred_np = W_pred.cpu().numpy().flatten()
    W_true_np = W_true.cpu().numpy().flatten()
    ss_res = np.sum((W_true_np - W_pred_np) ** 2)
    ss_tot = np.sum((W_true_np - W_true_np.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2B+C: Weight error heatmap + coefficient convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_recovery_figure(weight_error_grid, weight_error_epochs, theta_history, rule, title=""):
    """Reproduces Figure 2B and 2C from the paper."""
    thetas = torch.stack(theta_history).cpu().numpy()  # (epochs, 27)

    idx_110 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(1,1,0))
    idx_021 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(0,2,1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or "Oja's Rule Recovery", fontsize=13)

    # Left: Weight error heatmap (Figure 2B)
    # Rows = training epoch checkpoints, Cols = timestep t
    im = axes[0].imshow(
        weight_error_grid, aspect='auto', cmap='RdYlGn',
        origin='upper', vmin=0, vmax=weight_error_grid[0].max()
    )
    plt.colorbar(im, ax=axes[0], label='Mean Squared Weight Error')
    n_ticks = min(5, len(weight_error_epochs))
    tick_pos = np.linspace(0, len(weight_error_epochs) - 1, n_ticks, dtype=int)
    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels([weight_error_epochs[i] for i in tick_pos])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Training Epoch')
    axes[0].set_title('Weight Error over Training (Figure 2B)')
    
    # Right: θ coefficient convergence (Figure 2C)
    n_epochs = len(theta_history)
    epochs = np.arange(n_epochs)
    
    # Plot all coefficients grey, then highlight the two Oja ones
    for k in range(thetas.shape[1]):
        if k not in (idx_110, idx_021):
            axes[1].plot(epochs, thetas[:, k], color='grey', alpha=0.2, lw=0.8)
    
    axes[1].plot(epochs, thetas[:, idx_110], color='orange', lw=2,
                 label=f'θ₁₁₀ (Hebbian, target +1)')
    axes[1].plot(epochs, thetas[:, idx_021], color='royalblue', lw=2,
                 label=f'θ₀₂₁ (decay, target -1)')
    axes[1].axhline(y=1.0,  color='orange',    linestyle='--', alpha=0.6)
    axes[1].axhline(y=-1.0, color='royalblue', linestyle='--', alpha=0.6)
    axes[1].axhline(y=0.0,  color='black',     linestyle='-',  alpha=0.2)
    
    axes[1].set_xlabel('Training Epoch')
    axes[1].set_ylabel('θ value')
    axes[1].set_title("Taylor Coefficient Convergence (Figure 2C)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f'week2_ojas_recovery{"_" + title.replace(" ","_") if title else ""}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2D: R² grid over noise × sparsity (the main robustness result)
# ─────────────────────────────────────────────────────────────────────────────

def robustness_experiment():
    """
    Reproduces Figure 2D from the paper.
    Sweeps noise_std in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    and sparsity  in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    Records R² for each combination.
    
    WARNING: This takes a while. Run on GPU or reduce n_epochs to 200 for a quick version.
    """
    noise_levels   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sparsity_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    r2_grid = np.zeros((len(noise_levels), len(sparsity_levels)))
    
    total = len(noise_levels) * len(sparsity_levels)
    done = 0
    
    for i, noise in enumerate(noise_levels):
        for j, sparsity in enumerate(sparsity_levels):
            print(f"\n[{done+1}/{total}] noise={noise:.1f}, sparsity={sparsity:.1f}")
            _, _, r2, _, _, _, _ = run_ojas_recovery(
                noise_std=noise,
                sparsity=sparsity,
                n_epochs=400,       # reduce for speed; use 400 for final results
                n_trajectories=50,
                verbose=False
            )
            r2_grid[i, j] = r2
            print(f"  R² = {r2:.3f}")
            done += 1
    
    # ── Plot heatmap (Figure 2D) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    
    im = ax.imshow(r2_grid, aspect='auto', cmap='YlGn', vmin=0, vmax=1,
                   origin='upper')
    plt.colorbar(im, ax=ax, label='R² score')
    
    ax.set_xticks(range(len(sparsity_levels)))
    ax.set_xticklabels([f'{s:.1f}' for s in sparsity_levels])
    ax.set_yticks(range(len(noise_levels)))
    ax.set_yticklabels([f'{n:.1f}' for n in noise_levels])
    ax.set_xlabel('Sparsity (fraction of neurons observed)')
    ax.set_ylabel('Noise Scale (σ)')
    ax.set_title('R² over weights — Noise vs Sparsity\n(Figure 2D reproduction)')
    
    # Annotate cells
    for i in range(len(noise_levels)):
        for j in range(len(sparsity_levels)):
            ax.text(j, i, f'{r2_grid[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if r2_grid[i,j] > 0.4 else 'grey')
    
    plt.tight_layout()
    plt.savefig('week2_robustness_heatmap.png', dpi=150)
    plt.show()
    print("Saved: week2_robustness_heatmap.png")
    print("\nR² grid:")
    print(np.round(r2_grid, 3))
    
    return r2_grid


def robustness_ef_experiment(n_seeds=50):
    """
    Reproduces Figure 2E and 2F from the paper.
    Fig 2E: R² distribution (50 seeds) varying noise,    sparsity=1.0 (first column of Fig 2D)
    Fig 2F: R² distribution (50 seeds) varying sparsity, noise=0.0  (first row  of Fig 2D)

    WARNING: Runs 50 * 6 * 2 = 600 training runs. Use GPU or reduce n_seeds for speed.
    """
    noise_levels   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sparsity_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    # Fig 2E: vary noise, sparsity fixed at 1.0
    print("=== Fig 2E: noise sweep (50 seeds each) ===")
    r2_noise = []
    for noise in noise_levels:
        scores = []
        for s in range(n_seeds):
            _, _, r2, _, _, _, _ = run_ojas_recovery(
                noise_std=noise, sparsity=1.0, n_epochs=400, seed=s, verbose=False
            )
            scores.append(r2)
        r2_noise.append(scores)
        print(f"  noise={noise:.1f}: median R²={np.median(scores):.3f}")

    # Fig 2F: vary sparsity, noise fixed at 0.0
    print("=== Fig 2F: sparsity sweep (50 seeds each) ===")
    r2_sparsity = []
    for sparsity in sparsity_levels:
        scores = []
        for s in range(n_seeds):
            _, _, r2, _, _, _, _ = run_ojas_recovery(
                noise_std=0.0, sparsity=sparsity, n_epochs=400, seed=s, verbose=False
            )
            scores.append(r2)
        r2_sparsity.append(scores)
        print(f"  sparsity={sparsity:.1f}: median R²={np.median(scores):.3f}")

    # Plot Fig 2E/F
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot(r2_noise, labels=[f'{n:.1f}' for n in noise_levels])
    axes[0].set_xlabel('Noise Scale (σ)')
    axes[0].set_ylabel('R² score')
    axes[0].set_title('R² vs Noise Scale (Figure 2E)')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(r2_sparsity, labels=[f'{s:.1f}' for s in sparsity_levels])
    axes[1].set_xlabel('Sparsity (fraction of neurons observed)')
    axes[1].set_ylabel('R² score')
    axes[1].set_title('R² vs Sparsity (Figure 2F)')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week2_robustness_ef.png', dpi=150)
    plt.show()
    print("Saved: week2_robustness_ef.png")
    return r2_noise, r2_sparsity


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    # Step 1: Basic recovery (fast, ~2 min)
    print("=== Step 1: Basic Oja's recovery ===")
    loss_h, theta_h, r2_w, rule, w_err_grid, w_err_epochs, r2_a = run_ojas_recovery(n_epochs=400, verbose=True)
    plot_recovery_figure(w_err_grid, w_err_epochs, theta_h, rule)
    print(f"Final R² (weights):  {r2_w:.4f}  (paper reports ~0.9+)")
    print(f"Final R² (activity): {r2_a:.4f}")

    # Step 2: Noise/sparsity robustness grid (slow, ~20 min — run overnight)
    print("\n=== Step 2: Robustness grid (noise × sparsity) — Figure 2D ===")
    r2_grid = robustness_experiment()

    # Step 3: Boxplots with 50 seeds for Fig 2E/F (very slow — run on GPU)
    print("\n=== Step 3: Boxplots over 50 seeds — Figure 2E/F ===")
    robustness_ef_experiment(n_seeds=50)