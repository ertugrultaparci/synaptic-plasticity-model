# circuit_model.py
import torch
import torch.nn as nn

class CircuitModel(nn.Module):
    def __init__(self, n_input, n_output, plasticity_rule, lr=None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.plasticity_rule = plasticity_rule
        self.lr = lr if lr is not None else 1.0 / n_input
    
    def forward(self, X, W_init=None, observed_idx=None):
        """
        FIX: Accept W_init as argument so caller controls initialization.
        When W_init is passed, the same starting weights are used every
        forward call — critical for the gradient to learn consistently.
        """
        T = X.shape[0]
        
        if W_init is None:
            # Only random-init if no W_init provided (not recommended for toy test)
            device = next(self.plasticity_rule.parameters()).device
            W = torch.randn(self.n_output, self.n_input,
                            device=device) / (self.n_input ** 0.5)
        else:
            W = W_init.clone()  # clone so we don't modify the stored init

        # CRITICAL: W must NOT require grad itself.
        # Gradients flow to theta via: theta → dW → W_new → y → loss
        # NOT via W_init → anything
        
        m_traj = []
        
        for t in range(T):
            x = X[t]
            
            # Forward pass — y depends on W which depends on theta (after t=0)
            y = torch.sigmoid(W @ x)
            
            if observed_idx is not None:
                m = y[observed_idx]
            else:
                m = y
            m_traj.append(m)
            
            # Weight update — this is where theta enters the graph
            dW = self.plasticity_rule(x, y, W, r=None)
            
            # FIX: Do NOT call .detach() here — ever.
            # This line keeps the computation graph alive across timesteps:
            W = W + self.lr * dW
        
        return torch.stack(m_traj)  # (T, n_observed)