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
        """
        self.indices = [
            (a, b, g, r) for a, b, g, r in product(
                range(max_order + 1), range(max_order + 1),
                range(max_order + 1), r_range
            )
            if (a + b + g) <= max_order + 1
        ]
        """

        
        self.indices = [
            #(1,0,0,0),   # x
            #(0,1,0,0),   # y
            #(0,0,1,0),   # w
            (1,1,0,0),   # xy
            (1,0,1,0),   # xw
            #(0,1,1,0),   # yw
            (2,0,0,0),   # x²
            (0,2,0,0),   # y²
            (0,2,1,0),   # y²w  ← Oja decay
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
    
class TaylorRule3Var(nn.Module):
    """Taylor series plasticity rule with 3 variables (x, y, w)."""
    def __init__(self, init_scale=1e-4):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(3, 3, 3) * init_scale)
    
    def forward(self, x, y, w):
        # 1. Create stacks ONLY for the small vectors x and y.
        # We explicitly use dim=1 to map to the 'a' and 'b' indices.
        X = torch.stack([torch.ones_like(x), x, x**2], dim=1)  # Shape: (B, 3, N_in)
        Y = torch.stack([torch.ones_like(y), y, y**2], dim=1)  # Shape: (B, 3, N_out)
        
        # 2. Compute the interaction between x, y, and theta FIRST.
        # This is a much lighter 3-tensor contraction.
        # H Shape: (Batch, c_poly_degree, N_out, N_in)
        H = torch.einsum('abc, kaj, kbi -> kcij', self.coeffs, X, Y)
        
        # 3. Apply w as a simple broadcasted polynomial at the very end.
        # This completely avoids allocating a (3, B, N_out, N_in) intermediate tensor!
        dW = H[:, 0, :, :] + H[:, 1, :, :] * w + H[:, 2, :, :] * (w ** 2)
        
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