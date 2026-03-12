# week2_ojas_recovery.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_generation import generate_ojas_data
from plasticity_rules import TaylorPlasticityRule
from circuit_model import CircuitModel
from training import mse_loss

# ─────────────────────────────────────────────────────────────────────────────
# Full Oja's Recovery Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ojas_recovery(
    n_input=100, n_output=50, T=50,
    n_trajectories=50, n_epochs=400,
    noise_std=0.0, sparsity=1.0,
    lr_optimizer=1e-3, circuit_lr=0.01,
    grad_clip=1.0, l1_lambda=0.0,
    seed=42, verbose=True
):
    """
    Full Oja's rule recovery experiment.
    Reproduces Figure 2 from Mehta et al. 2024.
    
    Returns loss history, theta history, and R² on held-out test weights.
    """
    torch.manual_seed(seed)
    
    # ── Generate data ─────────────────────────────────────────────────────
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=n_input, n_output=n_output, T=T,
        n_trajectories=n_trajectories,
        noise_std=noise_std, sparsity=sparsity,
        lr=circuit_lr, seed=seed
    )
    
    # Train/test split (use last 10 trajectories as test set)
    n_test = 10
    X_train, O_train = X[:-n_test], O[:-n_test]
    X_test,  O_test  = X[-n_test:], O[-n_test:]
    W_inits_train = W_gt[:-n_test, 0]   # initial weights for training trajs
    W_inits_test  = W_gt[-n_test:, 0]   # initial weights for test trajs
    W_gt_test     = W_gt[-n_test:]      # full weight trajectories for R²
    
    # ── Train ─────────────────────────────────────────────────────────────
    rule = TaylorPlasticityRule(max_order=2, include_reward=False)
    circuit = CircuitModel(n_input, n_output, rule, lr=circuit_lr)
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
    
    # ── Evaluate: R² on test weight trajectories ──────────────────────────
    rule.eval()
    with torch.no_grad():
        r2_scores = []
        for i in range(n_test):
            # Run model forward to get predicted weight trajectory
            _, W_pred = circuit_forward_with_weights(
                circuit, X_test[i], W_inits_test[i]
            )
            W_true = W_gt_test[i]  # (T+1, n_output, n_input)
            r2 = compute_r2(W_pred, W_true)
            r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    if verbose:
        print(f"\nMean R² on test weight trajectories: {mean_r2:.4f}")
    
    return loss_history, theta_history, mean_r2, rule


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
    W_pred_np = W_pred.numpy().flatten()
    W_true_np = W_true.numpy().flatten()
    ss_res = np.sum((W_true_np - W_pred_np) ** 2)
    ss_tot = np.sum((W_true_np - W_true_np.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2B+C: Weight error heatmap + coefficient convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_recovery_figure(loss_history, theta_history, rule, title=""):
    """Reproduces Figure 2B and 2C from the paper."""
    thetas = torch.stack(theta_history).numpy()  # (epochs, 27)
    
    idx_110 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(1,1,0))
    idx_021 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(0,2,1))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or "Oja's Rule Recovery", fontsize=13)
    
    # Left: Loss over training (proxy for Figure 2B weight error)
    axes[0].semilogy(loss_history, color='teal')
    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss (Figure 2B proxy)')
    axes[0].grid(True, alpha=0.3)
    
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
            _, _, r2, _ = run_ojas_recovery(
                noise_std=noise,
                sparsity=sparsity,
                n_epochs=200,       # reduce for speed; use 400 for final results
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


if __name__ == "__main__":
    # Step 1: Basic recovery (fast, ~2 min)
    print("=== Step 1: Basic Oja's recovery ===")
    loss_h, theta_h, r2, rule = run_ojas_recovery(n_epochs=400, verbose=True)
    plot_recovery_figure(loss_h, theta_h, rule)
    print(f"Final R²: {r2:.4f}  (paper reports ~0.9+ for clean data)")
    
    # Step 2: Noise/sparsity robustness grid (slow, ~20 min — run overnight)
    print("\n=== Step 2: Robustness grid (noise × sparsity) ===")
    r2_grid = robustness_experiment()