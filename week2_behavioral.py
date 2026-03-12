# week2_behavioral.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from plasticity_rules import TaylorPlasticityRule, MLPPlasticityRule
from circuit_model import CircuitModel


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
    
    # Network: 2-10-1, plastic weights only at input→output layer
    # (following paper's description)
    W = torch.randn(1, n_hidden) * 0.1       # output weights (fixed proxy)
    W_plastic = torch.randn(n_hidden, 2) * 0.1  # plastic input→hidden weights
    
    X_all, choices_all, rewards_all = [], [], []
    W_traj = [W_plastic.clone()]
    
    recent_rewards = []  # for computing E[R]
    
    for trial in range(n_trials):
        block = trial // 80
        
        # Encode odors as 2D vectors with noise (Appendix A.4)
        # Odor A = [input_firing, 0], Odor B = [0, input_firing]
        x_A = torch.tensor([input_firing, 0.0]) + torch.randn(2) * noise_std
        x_B = torch.tensor([0.0, input_firing]) + torch.randn(2) * noise_std
        
        # Softmax choice probability based on current weights
        act_A = torch.sigmoid(W_plastic @ x_A).mean().item()
        act_B = torch.sigmoid(W_plastic @ x_B).mean().item()
        
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
            
            # Output: mean activity → choice probability
            m = h.mean().unsqueeze(0)   # scalar in [0,1]
            m_traj.append(m)
            
            # Weight update via learned rule
            dW = self.plasticity_rule(x, h, W, r=r)
            W = W + self.lr * dW
        
        return torch.stack(m_traj).squeeze()  # (T,)


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


def run_behavioral_recovery(rule_type='taylor', n_seeds=10, n_epochs=350):
    """
    Reproduces Figure 3 from the paper.
    Recovers the reward-based rule Δw = x_j * r from binary choices only.
    """
    print(f"\n=== Behavioral Recovery ({rule_type}) ===")
    
    r2_scores, pde_scores = [], []
    
    for seed in range(n_seeds):
        X, choices, rewards, W_gt = generate_behavioral_data(seed=seed)
        W_init = W_gt[0]  # initial weights
        
        # Create rule
        if rule_type == 'taylor':
            rule = TaylorPlasticityRule(max_order=2, include_reward=True)
        else:
            rule = MLPPlasticityRule(hidden_size=10, include_reward=True)
        
        circuit = BehavioralCircuit(
            n_hidden=10, plasticity_rule=rule, lr=0.1
        )
        optimizer = optim.Adam(
            rule.parameters(), lr=1e-3,
            weight_decay=0 if rule_type == 'mlp' else 0
        )
        
        # L1 only for Taylor (paper uses 1e-2)
        l1 = 1e-2 if rule_type == 'taylor' else 0.0
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            m = circuit.forward(X, rewards, W_plastic_init=W_init)
            loss = bce_loss(m, choices)
            if l1 > 0:
                loss = loss + l1 * rule.theta.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            m_final = circuit.forward(X, rewards, W_plastic_init=W_init)
            pde = percent_deviance_explained(m_final, choices)
            pde_scores.append(pde)
        
        print(f"  Seed {seed}: %DevExp = {pde:.1f}%")
    
    print(f"\nMean %DevExp ({rule_type}): {np.mean(pde_scores):.1f}% ± {np.std(pde_scores):.1f}%")
    return pde_scores


if __name__ == "__main__":
    # Quick data check
    X, choices, rewards, W_traj = generate_behavioral_data(seed=42)
    print(f"X: {X.shape}, choices: {choices.shape}, W_traj: {W_traj.shape}")
    print(f"Choice distribution: {choices.mean():.2f} (should be ~0.5 early, shifting later)")
    
    # Run recovery
    pde_taylor = run_behavioral_recovery('taylor', n_seeds=5)
    pde_mlp    = run_behavioral_recovery('mlp',    n_seeds=5)
    
    # Compare (Figure 3F)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot([pde_taylor, pde_mlp], labels=['Taylor', 'MLP'])
    ax.set_ylabel('% Deviance Explained')
    ax.set_title('Behavioral Recovery (Figure 3F)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('week2_behavioral_recovery.png', dpi=150)
    plt.show()