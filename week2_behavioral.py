# week2_behavioral.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from plasticity_rules import TaylorPlasticityRule, MLPPlasticityRule
from circuit_model import CircuitModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bce_loss(m_traj, o_traj):
    eps = 1e-7
    m = m_traj.clamp(eps, 1 - eps)
    return -(o_traj * torch.log(m) + (1 - o_traj) * torch.log(1 - m)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral simulation: mushroom-body style 2AFC task
# ─────────────────────────────────────────────────────────────────────────────

def generate_behavioral_data(
    n_trials=240,       # 3 blocks × 80 trials
    input_firing=0.75,  # from paper Appendix A.4
    noise_std=0.05,
    moving_avg_window=10,
    n_hidden=10,
    seed=42
):
    """
    Simulate the Drosophila mushroom-body 2AFC task.
    Architecture: 2 input → 10 hidden → 1 output (as in paper)
    Ground-truth rule: Δw_ij = x_j * r  where r = R - E[R]
    
    Reward blocks (from Appendix A.4):
        Block 1 (trials 0-79):   Odor A=0.2, Odor B=0.8
        Block 2 (trials 80-159): Odor A=0.9, Odor B=0.1
        Block 3 (trials 160-239):Odor A=0.2, Odor B=0.8
    
    Returns:
        X_traj:      (n_trials, 2) — odor stimuli
        choices:     (n_trials,)   — binary 0/1
        rewards:     (n_trials,)   — binary reward received
        W_traj:      (n_trials+1, n_hidden, 2) — weight trajectory
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Reward probabilities per block
    reward_probs = {
        'A': [0.2, 0.9, 0.2],
        'B': [0.8, 0.1, 0.8]
    }
    
    # Network: 2-10-1 (Appendix A.4)
    # W_out: fixed (1, n_hidden) readout weights — not updated by plasticity
    # W_plastic: (n_hidden, 2) input→hidden plastic weights
    W_out = torch.randn(1, n_hidden) * 0.1        # fixed readout
    W_plastic = torch.randn(n_hidden, 2) * 0.1    # plastic input→hidden weights
    
    X_all, choices_all, rewards_all = [], [], []
    W_traj = [W_plastic.clone()]
    
    recent_rewards = []  # for computing E[R]
    
    for trial in range(n_trials):
        block = trial // 80
        
        # Encode odors as 2D vectors with noise (Appendix A.4)
        # Odor A = [input_firing, 0], Odor B = [0, input_firing]
        x_A = torch.tensor([input_firing, 0.0]) + torch.randn(2) * noise_std
        x_B = torch.tensor([0.0, input_firing]) + torch.randn(2) * noise_std
        
        # Softmax choice probability: hidden activations → readout → sigmoid
        h_A = torch.sigmoid(W_plastic @ x_A)          # (n_hidden,)
        h_B = torch.sigmoid(W_plastic @ x_B)          # (n_hidden,)
        act_A = torch.sigmoid(W_out @ h_A).squeeze().item()
        act_B = torch.sigmoid(W_out @ h_B).squeeze().item()
        
        # Softmax over [act_A, act_B] → probability of choosing A
        temp = 5.0  # temperature
        prob_A = np.exp(temp * act_A) / (np.exp(temp * act_A) + np.exp(temp * act_B))
        
        # Make choice
        choice = 0 if np.random.rand() < prob_A else 1   # 0=A, 1=B
        chosen_odor = x_A if choice == 0 else x_B
        
        # Get reward
        p_reward = reward_probs['A' if choice == 0 else 'B'][block]
        R = 1.0 if np.random.rand() < p_reward else 0.0
        
        # Compute E[R] as moving average
        recent_rewards.append(R)
        if len(recent_rewards) > moving_avg_window:
            recent_rewards.pop(0)
        E_R = np.mean(recent_rewards)
        
        # Reward prediction error (RPE)
        r = R - E_R
        
        # Ground-truth plasticity: Δw_ij = x_j * r
        dW = torch.outer(torch.ones(n_hidden) * r, chosen_odor)
        W_plastic = W_plastic + 0.1 * dW
        
        X_all.append(chosen_odor)
        choices_all.append(float(choice))
        rewards_all.append(R)
        W_traj.append(W_plastic.clone())
    
    return (
        torch.stack(X_all),           # (T, 2)
        torch.tensor(choices_all),    # (T,)
        torch.tensor(rewards_all),    # (T,)
        torch.stack(W_traj)           # (T+1, n_hidden, 2)
    )


class BehavioralCircuit(torch.nn.Module):
    """
    3-layer circuit for behavioral inference.
    2 input → n_hidden → 1 output
    Plastic weights: input → hidden layer only.
    """
    def __init__(self, n_hidden, plasticity_rule, lr=0.1, moving_avg_window=10):
        super().__init__()
        self.n_hidden = n_hidden
        self.plasticity_rule = plasticity_rule
        self.lr = lr
        self.window = moving_avg_window
        # Fixed (non-plastic) output readout weights — paper Fig 3A architecture
        self.register_buffer('W_out', torch.randn(1, n_hidden) * 0.1)
    
    def forward(self, X, rewards, W_plastic_init=None):
        """
        Args:
            X:       (T, 2) — stimuli for chosen odors
            rewards: (T,)   — actual rewards received
        Returns:
            m_traj:  (T,) — predicted choice probability
        """
        T = X.shape[0]
        
        if W_plastic_init is None:
            W = torch.randn(self.n_hidden, 2) * 0.1
        else:
            W = W_plastic_init.clone()
        
        m_traj = []
        recent_rewards = []
        
        for t in range(T):
            x = X[t]          # (2,)
            R = rewards[t]
            
            # Compute E[R]
            recent_rewards.append(R.item())
            if len(recent_rewards) > self.window:
                recent_rewards.pop(0)
            E_R = torch.tensor(np.mean(recent_rewards))
            r = R - E_R       # scalar reward signal
            
            # Hidden layer
            h = torch.sigmoid(W @ x)    # (n_hidden,)

            # Output: fixed readout W_out → choice probability (paper Fig 3A)
            m = torch.sigmoid(self.W_out @ h)  # (1,)
            m_traj.append(m)
            
            # Weight update via learned rule
            dW = self.plasticity_rule(x, h, W, r=r)
            W = W + self.lr * dW
        
        return torch.stack(m_traj).squeeze()  # (T,)


def behavioral_forward_with_weights(circuit, X, rewards, W_init):
    """
    Run BehavioralCircuit and return (m_traj, W_traj).
    Mirrors circuit_forward_with_weights from week2_ojas_recovery.
    """
    T = X.shape[0]
    W = W_init.clone()
    m_traj, W_traj = [], [W.clone()]
    recent_rewards = []

    for t in range(T):
        x = X[t]
        R = rewards[t]
        recent_rewards.append(R.item())
        if len(recent_rewards) > circuit.window:
            recent_rewards.pop(0)
        E_R = float(np.mean(recent_rewards))
        r = R - E_R

        h = torch.sigmoid(W @ x)
        m = torch.sigmoid(circuit.W_out @ h)  # (1,)
        m_traj.append(m)

        dW = circuit.plasticity_rule(x, h, W, r=r)
        W = W + circuit.lr * dW
        W_traj.append(W.clone())

    return torch.stack(m_traj).squeeze(), torch.stack(W_traj)


def compute_r2(W_pred, W_true):
    """R² between predicted and ground-truth weight trajectories."""
    W_pred_np = W_pred.cpu().numpy().flatten()
    W_true_np = W_true.cpu().numpy().flatten()
    ss_res = np.sum((W_true_np - W_pred_np) ** 2)
    ss_tot = np.sum((W_true_np - W_true_np.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def percent_deviance_explained(model_probs, choices, null_probs=None):
    """
    From paper Appendix A.4 (Equation 11).
    Null model = weights frozen at random initialization.
    """
    eps = 1e-7
    m = model_probs.clamp(eps, 1 - eps)
    
    dev_model = -2 * (choices * torch.log(m) + 
                      (1 - choices) * torch.log(1 - m)).sum()
    
    # Null deviance: model always predicts 0.5
    null = torch.full_like(m, 0.5)
    dev_null = -2 * (choices * torch.log(null) + 
                     (1 - choices) * torch.log(1 - null)).sum()
    
    return 100 * (1 - dev_model / dev_null).item()


def run_behavioral_recovery(rule_type='taylor', n_seeds=10, n_epochs=350,
                             n_train=18, n_eval=7, device=DEVICE):
    """
    Reproduces Figure 3 from the paper.
    Recovers the reward-based rule Δw = x_j * r from binary choices only.

    Per Appendix A.4:
      - 18 trajectories for training, 7 for evaluation per seed.
      - Model weights initialized randomly (not from ground truth).
      - Updates performed on a single trajectory at a time (no batching).
      - L1 regularization 1e-2 applied to Taylor coefficients only.
      - Results reported as median across eval trajectories per seed.
    """
    print(f"\n=== Behavioral Recovery ({rule_type}) ===")

    r2_scores, pde_scores = [], []
    final_thetas = []            # final rule.theta per seed  (for Fig 3E)
    theta_history_seed0 = []    # theta per epoch for seed 0 (for Fig 3D)
    sample_W_traj = None        # W trajectory example from seed 0 (for Fig 3B)

    for main_seed in range(n_seeds):
        torch.manual_seed(main_seed)
        np.random.seed(main_seed)

        # Generate 18 training + 7 eval trajectories, move to device
        train_data = [
            tuple(t.to(device) for t in generate_behavioral_data(seed=main_seed * 100 + i))
            for i in range(n_train)
        ]
        eval_data = [
            tuple(t.to(device) for t in generate_behavioral_data(seed=main_seed * 100 + 50 + i))
            for i in range(n_eval)
        ]

        # Model uses RANDOM initial weights (not ground-truth) — Appendix A.4
        torch.manual_seed(main_seed * 999)
        W_model_inits_train = [torch.randn(10, 2, device=device) * 0.1 for _ in range(n_train)]
        W_model_inits_eval  = [torch.randn(10, 2, device=device) * 0.1 for _ in range(n_eval)]

        # Create rule and circuit
        if rule_type == 'taylor':
            rule = TaylorPlasticityRule(max_order=2, include_reward=True).to(device)
        else:
            rule = MLPPlasticityRule(hidden_size=10, include_reward=True).to(device)

        circuit = BehavioralCircuit(n_hidden=10, plasticity_rule=rule, lr=0.1).to(device)
        optimizer = optim.Adam(rule.parameters(), lr=1e-3)
        l1 = 1e-2 if rule_type == 'taylor' else 0.0

        # Train: one trajectory at a time, no batching (Appendix A.4)
        for epoch in range(n_epochs):
            for k, (X_k, choices_k, rewards_k, _) in enumerate(train_data):
                optimizer.zero_grad()
                m = circuit.forward(X_k, rewards_k,
                                    W_plastic_init=W_model_inits_train[k])
                loss = bce_loss(m, choices_k)
                if l1 > 0:
                    loss = loss + l1 * rule.theta.abs().sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
                optimizer.step()

            # Track theta evolution for seed 0 (Fig 3D)
            if main_seed == 0 and hasattr(rule, 'theta'):
                theta_history_seed0.append(rule.theta.detach().clone())

        # Store final theta for Fig 3E
        if hasattr(rule, 'theta'):
            final_thetas.append(rule.theta.detach().clone())

        # Evaluate on 7 held-out trajectories; report median per seed
        r2_eval, pde_eval = [], []
        with torch.no_grad():
            for k, (X_k, choices_k, rewards_k, W_gt_k) in enumerate(eval_data):
                m_final, W_pred = behavioral_forward_with_weights(
                    circuit, X_k, rewards_k, W_model_inits_eval[k]
                )
                r2_eval.append(compute_r2(W_pred, W_gt_k))
                pde_eval.append(percent_deviance_explained(m_final, choices_k))

            # Save one example W_traj from seed 0 for Fig 3B
            if main_seed == 0:
                _, W_ex = behavioral_forward_with_weights(
                    circuit, eval_data[0][0], eval_data[0][2], W_model_inits_eval[0]
                )
                W_gt_ex = eval_data[0][3]
                sample_W_traj = (W_ex, W_gt_ex)

        r2_scores.append(np.median(r2_eval))
        pde_scores.append(np.median(pde_eval))
        print(f"  Seed {main_seed}: R²={r2_scores[-1]:.3f}, "
              f"%DevExp={pde_scores[-1]:.1f}%")

    print(f"\nMean R²   ({rule_type}): {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"Mean %Dev ({rule_type}): {np.mean(pde_scores):.1f}% ± {np.std(pde_scores):.1f}%")
    return r2_scores, pde_scores, theta_history_seed0, final_thetas, sample_W_traj


def plot_behavioral_figures(rule_type, r2_scores, pde_scores,
                             theta_history_seed0, final_thetas, sample_W_traj):
    """
    Produces Figures 3B, 3C, 3D, 3E, 3F from the paper.
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Behavioral Recovery — {rule_type.upper()} rule", fontsize=14)

    # ── Fig 3B: Single synapse weight vs trial ────────────────────────────
    ax_b = fig.add_subplot(2, 3, 1)
    if sample_W_traj is not None:
        W_pred_ex, W_gt_ex = sample_W_traj
        trials = np.arange(W_pred_ex.shape[0])
        # Plot first 3 synapses (hidden unit 0, 1, 2 → input 0)
        for s in range(min(3, W_pred_ex.shape[1])):
            ax_b.plot(trials, W_pred_ex[:, s, 0].cpu().numpy(),
                      label=f'Predicted synapse {s}', lw=1.5)
            ax_b.plot(trials, W_gt_ex[:, s, 0].cpu().numpy(),
                      linestyle='--', alpha=0.6, label=f'Ground truth {s}')
    ax_b.set_xlabel('Trial')
    ax_b.set_ylabel('Synaptic weight w')
    ax_b.set_title('Fig 3B: Weight vs Trial')
    ax_b.legend(fontsize=6, ncol=2)
    ax_b.grid(True, alpha=0.3)

    # ── Fig 3C: R² on weights boxplot ─────────────────────────────────────
    ax_c = fig.add_subplot(2, 3, 2)
    ax_c.boxplot([r2_scores], labels=[rule_type.upper()])
    ax_c.set_ylabel('R² score (weights)')
    ax_c.set_title('Fig 3C: R² on Weight Trajectories')
    ax_c.set_ylim(0, 1)
    ax_c.grid(True, alpha=0.3)

    # ── Fig 3D: θ evolution during training (seed 0) ─────────────────────
    ax_d = fig.add_subplot(2, 3, 3)
    if theta_history_seed0 and rule_type == 'taylor':
        thetas = torch.stack(theta_history_seed0).cpu().numpy()  # (epochs, n_terms)
        epochs = np.arange(len(theta_history_seed0))
        # Plot all grey, then highlight dominant terms
        for k in range(thetas.shape[1]):
            ax_d.plot(epochs, thetas[:, k], color='grey', alpha=0.15, lw=0.8)
        # Highlight the term with largest final absolute value
        top_k = np.argsort(np.abs(thetas[-1]))[-3:]
        colors = ['tab:orange', 'tab:blue', 'tab:green']
        for idx, c in zip(top_k, colors):
            ax_d.plot(epochs, thetas[:, idx], color=c, lw=2,
                      label=f'θ[{idx}]={thetas[-1, idx]:+.2f}')
        ax_d.axhline(0, color='black', alpha=0.3)
        ax_d.legend(fontsize=7)
    elif theta_history_seed0:
        ax_d.text(0.5, 0.5, 'MLP rule:\nno θ coefficients',
                  ha='center', va='center', transform=ax_d.transAxes)
    ax_d.set_xlabel('Epoch')
    ax_d.set_ylabel('θ value')
    ax_d.set_title('Fig 3D: θ Evolution (seed 0)')
    ax_d.grid(True, alpha=0.3)

    # ── Fig 3E: Final θ distribution across seeds ─────────────────────────
    ax_e = fig.add_subplot(2, 3, 4)
    if final_thetas and rule_type == 'taylor':
        final_arr = torch.stack(final_thetas).cpu().numpy()  # (n_seeds, n_terms)
        ax_e.boxplot(final_arr, vert=True, showfliers=False)
        ax_e.axhline(0, color='black', alpha=0.3)
        ax_e.set_xlabel('θ index')
        ax_e.set_ylabel('Final θ value')
    elif final_thetas:
        ax_e.text(0.5, 0.5, 'MLP rule:\nno θ coefficients',
                  ha='center', va='center', transform=ax_e.transAxes)
    ax_e.set_title('Fig 3E: Final θ Distribution')
    ax_e.grid(True, alpha=0.3)

    # ── Fig 3F: % Deviance Explained boxplot ─────────────────────────────
    ax_f = fig.add_subplot(2, 3, 5)
    ax_f.boxplot([pde_scores], labels=[rule_type.upper()])
    ax_f.set_ylabel('% Deviance Explained')
    ax_f.set_title('Fig 3F: Behavioral Recovery')
    ax_f.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'week2_behavioral_recovery_{rule_type}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    # Quick data check
    X, choices, rewards, W_traj = generate_behavioral_data(seed=42)
    print(f"X: {X.shape}, choices: {choices.shape}, W_traj: {W_traj.shape}")
    print(f"Choice distribution: {choices.mean():.2f} (should be ~0.5 early, shifting later)")

    # Run recovery — Taylor rule
    (r2_taylor, pde_taylor,
     theta_h_taylor, final_thetas_taylor,
     W_traj_ex_taylor) = run_behavioral_recovery('taylor', n_seeds=5)

    plot_behavioral_figures('taylor', r2_taylor, pde_taylor,
                            theta_h_taylor, final_thetas_taylor, W_traj_ex_taylor)

    # Run recovery — MLP rule
    (r2_mlp, pde_mlp,
     theta_h_mlp, final_thetas_mlp,
     W_traj_ex_mlp) = run_behavioral_recovery('mlp', n_seeds=5)

    plot_behavioral_figures('mlp', r2_mlp, pde_mlp,
                            theta_h_mlp, final_thetas_mlp, W_traj_ex_mlp)

    # Combined comparison (Fig 3C + 3F side by side)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].boxplot([r2_taylor, r2_mlp], labels=['Taylor', 'MLP'])
    axes[0].set_ylabel('R² score (weights)')
    axes[0].set_title('R² on Weight Trajectories (Figure 3C)')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[1].boxplot([pde_taylor, pde_mlp], labels=['Taylor', 'MLP'])
    axes[1].set_ylabel('% Deviance Explained')
    axes[1].set_title('Behavioral Recovery (Figure 3F)')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('week2_behavioral_recovery_combined.png', dpi=150)
    plt.show()