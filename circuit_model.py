import torch
import torch.nn as nn

class CircuitModel(nn.Module):
    def __init__(self, n_input, n_output, plasticity_rule):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.plasticity_rule = plasticity_rule
        self.lr = 1.0

    # circuit_model.py — corrected forward()
    def forward(self, X, W_init, observed_idx=None):
        B, T, _ = X.shape
        W = W_init.clone()
        m_traj = []

        for t in range(T):
            x_t = X[:, t, :]
            pre = torch.einsum('boi,bi->bo', W, x_t)
            y_t = torch.sigmoid(pre)

            # Sparsity = only RECORD observed neurons, not limit plasticity
            m = y_t[:, observed_idx] if observed_idx is not None else y_t
            m_traj.append(m)

            # ALL neurons update — consistent with data_generation.py
            dW = self.plasticity_rule(x_t, y_t, W)
            W  = W + self.lr * dW

        return torch.stack(m_traj, dim=1)

        return torch.stack(m_traj, dim=1)