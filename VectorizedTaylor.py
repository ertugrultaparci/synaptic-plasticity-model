import torch
import torch.nn as nn
from itertools import product

class TaylorPlasticityRule(nn.Module):
    """
    Vectorized Taylor series parameterization of the plasticity rule.
    """
    def __init__(self, max_order=2, include_reward=False):
        super().__init__()
        self.max_order = max_order
        self.include_reward = include_reward
        
        r_range = range(max_order + 1) if include_reward else [0]
        self.indices = [
            (a, b, g, r) for a, b, g, r in product(
                range(max_order + 1), range(max_order + 1),
                range(max_order + 1), r_range
            )
        ]
        
        n_terms = len(self.indices)
        self.theta = nn.Parameter(torch.randn(n_terms) * 1e-2)
        
        # SPEED FIX: Create tensors for exponents so we can compute them all at once
        # Shape is (1, 27, 1, 1) so it broadcasts perfectly against (Batch, Terms, N_out, N_in)
        a_tensor = torch.tensor([idx[0] for idx in self.indices], dtype=torch.float32).view(1, -1, 1, 1)
        b_tensor = torch.tensor([idx[1] for idx in self.indices], dtype=torch.float32).view(1, -1, 1, 1)
        g_tensor = torch.tensor([idx[2] for idx in self.indices], dtype=torch.float32).view(1, -1, 1, 1)
        r_tensor = torch.tensor([idx[3] for idx in self.indices], dtype=torch.float32).view(1, -1, 1, 1)
        
        # register_buffer ensures these tensors are moved to the GPU automatically with the model
        self.register_buffer('a_tensor', a_tensor)
        self.register_buffer('b_tensor', b_tensor)
        self.register_buffer('g_tensor', g_tensor)
        self.register_buffer('r_tensor', r_tensor)
        
        print(f"TaylorPlasticityRule (Vectorized): {n_terms} terms")

    def forward(self, x_t, y_t, w, r=None):
        # We expect raw batched inputs directly from the CircuitModel
        # x_t: (Batch, N_in)
        # y_t: (Batch, N_out)
        # w:   (Batch, N_out, N_in)
        
        # Reshape inputs to broadcast against the 27 terms:
        # Shapes become (Batch, 1, N_out, N_in)
        x = x_t.view(x_t.shape[0], 1, 1, x_t.shape[1])
        y = y_t.view(y_t.shape[0], 1, y_t.shape[1], 1)
        w_expanded = w.unsqueeze(1)
        
        theta_view = self.theta.view(1, -1, 1, 1)
        
        # INSTANT MATH: No for-loop required! 
        # Computes all 27 terms across the entire batch simultaneously
        terms = (x ** self.a_tensor) * (y ** self.b_tensor) * (w_expanded ** self.g_tensor)
        
        if self.include_reward and r is not None:
            r_val = r.view(r.shape[0], 1, 1, 1)
            terms = terms * (r_val ** self.r_tensor)
            
        # Multiply by weights and sum across the 27 terms (dim=1)
        dW = (theta_view * terms).sum(dim=1)
        
        # Final output shape: (Batch, N_out, N_in)
        return dW

    def get_named_coefficients(self):
        print("\n=== Taylor Coefficients ===")
        for k, (a, b, g, rd) in enumerate(self.indices):
            val = self.theta[k].item()
            label = f"θ_{a}{b}{g}" + (f"{rd}" if self.include_reward else "")
            if abs(val) > 1e-3: 
                print(f"  {label:12s} = {val:+.4f}")

    @classmethod
    def with_ojas_coefficients(cls):
        rule = cls(max_order=2, include_reward=False)
        with torch.no_grad():
            rule.theta.fill_(0.0)
            for k, (a, b, g, _) in enumerate(rule.indices):
                if (a, b, g) == (1, 1, 0):
                    rule.theta[k] = 1.0    
                elif (a, b, g) == (0, 2, 1):
                    rule.theta[k] = -1.0   
        return rule