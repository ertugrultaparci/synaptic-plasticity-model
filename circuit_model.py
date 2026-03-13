import torch
import torch.nn as nn

class CircuitModel(nn.Module):
    def __init__(self, n_input, n_output, plasticity_rule):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.plasticity_rule = plasticity_rule
        self.lr = 1 / n_input

    def forward(self, X, W_init, observed_idx=None):
        B, T, _ = X.shape
        W = W_init.clone()
        m_traj = []

        for t in range(T):
            x_t = X[:, t, :]  
            
            pre = torch.bmm(W, x_t.unsqueeze(-1)).squeeze(-1)
            y_t = torch.sigmoid(pre) 

            if observed_idx is not None:
                m = y_t[:, observed_idx]
            else:
                m = y_t
            m_traj.append(m)

            # PASS DIRECTLY: No unsqueeze, no x_j, no y_i
            dW = self.plasticity_rule(x_t, y_t, W)
            W = W + self.lr * dW

        return torch.stack(m_traj, dim=1)