# training.py 
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from data_generation import generate_ojas_data
from plasticity_rules import TaylorPlasticityRule
from circuit_model import CircuitModel


def mse_loss(m_traj, o_traj):
    return ((o_traj - m_traj) ** 2).mean()


def train_inference_model(
    X_train,
    O_train,
    W_inits,            # FIX: pass fixed W_init per trajectory
    plasticity_rule,
    n_epochs=400,
    lr_optimizer=1e-3,
    observed_idx=None,
    n_input=100,
    n_output=50,
    grad_clip=1.0,      # FIX: was 0.2, too aggressive for small nets
    l1_lambda=0.0,
    verbose=True
):
    circuit = CircuitModel(
        n_input=n_input,
        n_output=n_output,
        plasticity_rule=plasticity_rule
    )
    
    optimizer = optim.Adam(plasticity_rule.parameters(), lr=lr_optimizer)
    
    n_traj = X_train.shape[0]
    loss_history = []
    theta_history = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n_traj)
        
        for traj_idx in perm:
            X = X_train[traj_idx].unsqueeze(0)
            O = O_train[traj_idx].unsqueeze(0)
            W_init = W_inits[traj_idx].unsqueeze(0)    # FIX: use stored init
            
            optimizer.zero_grad()
            
            # BPTT: runs full T-step loop, keeps graph alive
            m_traj = circuit.forward(X, W_init=W_init, observed_idx=observed_idx)
            
            loss = mse_loss(m_traj, O)
            
            if l1_lambda > 0 and hasattr(plasticity_rule, 'theta'):
                loss = loss + l1_lambda * plasticity_rule.theta.abs().sum()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                plasticity_rule.parameters(), grad_clip
            )
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_traj
        loss_history.append(avg_loss)
        
        if hasattr(plasticity_rule, 'theta'):
            theta_history.append(plasticity_rule.theta.detach().clone())
        
        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d}/{n_epochs}  |  Loss: {avg_loss:.6f}")
    
    return loss_history, theta_history


def toy_overfit_experiment():
    print("\n" + "="*60)
    print("TOY EXPERIMENT: Overfitting to 1 trajectory")
    print("="*60)
    
    torch.manual_seed(42)
    
    n_input, n_output, T = 100, 500, 50
    
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=n_input, n_output=n_output,
        T=T, n_trajectories=1
    )
    
    # FIX: store the SAME W_init we'll use every forward call.
    # Use the ground-truth initial weight (W_gt[:, 0, :, :] = W at t=0)
    # Shape: (n_trajectories, n_output, n_input)
    W_inits = W_gt[:, 0, :, :]  # (1, n_output, n_input)
    
    print(f"X: {X.shape}, O: {O.shape}, W_inits: {W_inits.shape}")
    
    rule = TaylorPlasticityRule(max_order=2, include_reward=False)
    
    loss_history, theta_history = train_inference_model(
        X_train=X,
        O_train=O,
        W_inits=W_inits,
        plasticity_rule=rule,
        n_epochs=1000,          # FIX: was 500, use 1000
        lr_optimizer=5e-3,      # FIX: was 1e-3, use 5e-3
        observed_idx=obs_idx,
        n_input=n_input,
        n_output=n_output,
        grad_clip=1.0,          # FIX: was 0.2
        l1_lambda=0.0,          # FIX: no regularization for toy test
        verbose=True
    )
    
    # ── Diagnostic: check gradient flow manually ──────────────────────────
    print("\n--- Gradient check ---")
    rule2 = TaylorPlasticityRule(max_order=2, include_reward=False)
    circuit2 = CircuitModel(n_input, n_output, rule2)
    m = circuit2.forward(X, W_init=W_inits, observed_idx=obs_idx)
    loss = mse_loss(m, O)
    loss.backward()
    grad_norm = rule2.theta.grad.norm().item()
    print(f"Gradient norm on theta after 1 backward: {grad_norm:.6f}")
    if grad_norm > 1e-8:
        print("✓ Gradients are flowing to theta!")
    else:
        print("✗ Gradients are ZERO — computation graph is broken!")
    
    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].semilogy(loss_history)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss (log scale)')
    axes[0].set_title('Toy Overfit: Loss over training')
    axes[0].grid(True, alpha=0.3)
    
    if theta_history:
        thetas = torch.stack(theta_history).numpy()
        idx_110 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(1,1,0))
        idx_021 = next(k for k,(a,b,g,_) in enumerate(rule.indices) if (a,b,g)==(0,2,1))
        
        axes[1].plot(thetas[:, idx_110], label='θ₁₁₀ (target: +1)', color='orange', lw=2)
        axes[1].plot(thetas[:, idx_021], label='θ₀₂₁ (target: -1)', color='blue', lw=2)
        axes[1].axhline(y=1.0,  color='orange', linestyle='--', alpha=0.5, label='Oja target +1')
        axes[1].axhline(y=-1.0, color='blue',   linestyle='--', alpha=0.5, label='Oja target -1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('θ value')
        axes[1].set_title("Taylor coefficients — should converge to Oja's values")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_toy_overfit_fixed.png', dpi=150)
    plt.show()
    
    # ── Pass/Fail ──────────────────────────────────────────────────────────
    final_loss = loss_history[-1]
    initial_loss = loss_history[0]
    reduction = 100 * (1 - final_loss / initial_loss)
    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Loss reduction: {reduction:.1f}%")
    
    if reduction > 90:
        print("✓ PASS: Successfully overfit 1 trajectory (>90% reduction)")
    elif reduction > 50:
        print("~ PARTIAL: Gradient flows but needs more epochs — try n_epochs=3000")
    else:
        print("✗ FAIL: Check gradient diagnostic above")


if __name__ == "__main__":
    toy_overfit_experiment()