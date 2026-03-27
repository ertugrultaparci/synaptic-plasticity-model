import torch
import torch.nn as nn

    
class TaylorRule3Var(nn.Module):
    """Taylor series plasticity rule with 3 variables (x, y, w)."""
    def __init__(self, init_scale=1e-2):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(3, 3, 3) * init_scale)
    
    def forward(self, x, y, w, observed_idx=None):
        B, N_in = x.shape
        _, N_out = y.shape

        # 🔥 Apply sparsity to weights
        if observed_idx is not None:
            mask = torch.zeros_like(w)
            mask[:, observed_idx, :] = 1.0
            w = w * mask

        X = torch.stack([torch.ones_like(x), x, x**2], dim=1)
        Y = torch.stack([torch.ones_like(y), y, y**2], dim=1)
        Wp = torch.stack([torch.ones_like(w), w, w**2], dim=1)

        dW = torch.einsum(
            'abc, kaj, kbi, kcij -> kij',
            self.coeffs, X, Y, Wp
        )

        return dW