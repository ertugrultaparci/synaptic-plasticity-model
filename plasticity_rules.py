# plasticity_rules.py
import torch
import torch.nn as nn
from itertools import product

class TaylorPlasticityRule(nn.Module):
    """
    Truncated Taylor series parameterization of the plasticity rule.
    
    g_θ(x_j, y_i, w_ij) = Σ_{α,β,γ=0}^{max_order} θ_{αβγ} * x^α * y^β * w^γ
    
    For Oja's rule: θ_110=1, θ_021=-1, all others=0
    """
    
    def __init__(self, max_order=2, include_reward=False):
        super().__init__()
        self.max_order = max_order
        self.include_reward = include_reward
        
        # Build list of (α, β, γ) index tuples
        r_range = range(max_order + 1) if include_reward else [0]
        self.indices = [
            (a, b, g, r)
            for a, b, g, r in product(
                range(max_order + 1),
                range(max_order + 1),
                range(max_order + 1),
                r_range
            )
        ]
        
        n_terms = len(self.indices)
        
        # θ parameters — initialized near zero (as in paper Appendix A.3)
        self.theta = nn.Parameter(
            torch.randn(n_terms) * 1e-2
        )
        
        print(f"TaylorPlasticityRule: {n_terms} terms "
              f"({'with' if include_reward else 'without'} reward)")
    
    def forward(self, x_j, y_i, w_ij, r=None):
        """
        Compute Δw_ij for a single synapse (i, j).
        
        In practice we vectorize over all (i,j) pairs:
        
        Args:
            x_j: presynaptic activity, shape (n_input,)
            y_i: postsynaptic activity, shape (n_output,)
            w_ij: weight matrix, shape (n_output, n_input)
            r: reward signal (scalar), used if include_reward=True
        
        Returns:
            dW: weight update matrix, shape (n_output, n_input)
        """
        # Expand to (n_output, n_input) for vectorized computation
        # x_j: broadcast along output dim → (1, n_input)
        # y_i: broadcast along input dim  → (n_output, 1)
        x = x_j.unsqueeze(0)   # (1, n_input)
        y = y_i.unsqueeze(1)   # (n_output, 1)
        w = w_ij               # (n_output, n_input)
        r_val = r if r is not None else torch.tensor(0.0)
        
        dW = torch.zeros_like(w)
        
        for k, (a, b, g, rd) in enumerate(self.indices):
            # Each term: θ_{αβγ} * x^α * y^β * w^γ * r^δ
            term = (x ** a) * (y ** b) * (w ** g) * (r_val ** rd)
            dW = dW + self.theta[k] * term
        
        return dW
    
    def get_named_coefficients(self):
        """Print coefficients with their (α,β,γ) labels — useful for tracking recovery."""
        print("\n=== Taylor Coefficients ===")
        for k, (a, b, g, rd) in enumerate(self.indices):
            val = self.theta[k].item()
            label = f"θ_{a}{b}{g}" + (f"{rd}" if self.include_reward else "")
            if abs(val) > 1e-3:  # only print non-negligible ones
                print(f"  {label:12s} = {val:+.4f}")
    
    @classmethod
    def with_ojas_coefficients(cls):
        """Create a Taylor rule pre-set to Oja's rule (for unit testing)."""
        rule = cls(max_order=2, include_reward=False)
        with torch.no_grad():
            rule.theta.fill_(0.0)
            # Find indices for (1,1,0) and (0,2,1)
            for k, (a, b, g, _) in enumerate(rule.indices):
                if (a, b, g) == (1, 1, 0):
                    rule.theta[k] = 1.0    # θ_110 = 1 (Hebbian term)
                elif (a, b, g) == (0, 2, 1):
                    rule.theta[k] = -1.0   # θ_021 = -1 (decay term)
        return rule


class MLPPlasticityRule(nn.Module):
    """
    MLP parameterization of the plasticity rule.
    
    g_θ^MLP(x_j, y_i, w_ij, r) = MLP_θ([x_j, y_i, w_ij, r])
    
    Input: 4 scalars per synapse → Output: 1 scalar (Δw_ij)
    Architecture: 4 → 10 → 1 (as in paper Appendix A.4)
    """
    
    def __init__(self, hidden_size=10, include_reward=False):
        super().__init__()
        self.include_reward = include_reward
        input_size = 4 if include_reward else 3
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights small so updates start near zero
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1e-3)
                nn.init.zeros_(layer.bias)
        
        print(f"MLPPlasticityRule: {input_size}-{hidden_size}-1 architecture")
    
    def forward(self, x_j, y_i, w_ij, r=None):
        """
        Args:
            x_j: shape (n_input,)
            y_i: shape (n_output,)
            w_ij: shape (n_output, n_input)
            r: scalar reward (optional)
        
        Returns:
            dW: shape (n_output, n_input)
        """
        n_out, n_in = w_ij.shape
        
        # Build per-synapse input vectors
        x_expanded = x_j.unsqueeze(0).expand(n_out, -1)    # (n_out, n_in)
        y_expanded = y_i.unsqueeze(1).expand(-1, n_in)      # (n_out, n_in)
        
        if self.include_reward and r is not None:
            r_expanded = torch.full_like(w_ij, r.item() if hasattr(r, 'item') else float(r))
            inputs = torch.stack([x_expanded, y_expanded, w_ij, r_expanded], dim=-1)
        else:
            inputs = torch.stack([x_expanded, y_expanded, w_ij], dim=-1)
        
        # inputs: (n_out, n_in, input_size)
        flat = inputs.reshape(-1, inputs.shape[-1])    # (n_out*n_in, input_size)
        dW_flat = self.net(flat).squeeze(-1)            # (n_out*n_in,)
        return dW_flat.reshape(n_out, n_in)


# ── Unit Tests ─────────────────────────────────────────────────────────────────

def unit_test_taylor_with_ojas():
    """Given ground-truth Oja θ, verify weight updates match Oja's rule analytically."""
    print("\n=== Unit Test: Taylor with Oja's coefficients ===")
    
    torch.manual_seed(0)
    rule = TaylorPlasticityRule.with_ojas_coefficients()
    
    n_in, n_out = 5, 3
    x = torch.randn(n_in)
    W = torch.randn(n_out, n_in) * 0.1
    y = torch.sigmoid(W @ x)
    
    # Our rule's output
    dW_rule = rule(x, y, W, r=None)
    
    # Analytic Oja's rule
    dW_true = torch.outer(y, x) - (y**2).unsqueeze(1) * W
    
    max_err = (dW_rule - dW_true).abs().max().item()
    print(f"  Max error between Taylor(Oja θ) and analytic Oja: {max_err:.2e}")
    assert max_err < 1e-5, f"Unit test FAILED: max error = {max_err}"
    print("  ✓ Taylor rule with Oja θ matches analytic Oja's rule exactly!")

def unit_test_mlp_updates():
    """Verify MLP rule produces correct output shapes."""
    print("\n=== Unit Test: MLP plasticity rule ===")
    
    rule = MLPPlasticityRule(hidden_size=10, include_reward=False)
    n_in, n_out = 100, 50
    x = torch.randn(n_in)
    W = torch.randn(n_out, n_in) * 0.1
    y = torch.sigmoid(W @ x)
    
    dW = rule(x, y, W)
    print(f"  dW shape: {dW.shape}  (expected: ({n_out}, {n_in}))")
    assert dW.shape == (n_out, n_in), "Wrong shape!"
    assert not torch.isnan(dW).any(), "NaNs in MLP output!"
    print("  ✓ MLP rule produces correct shape, no NaNs")

if __name__ == "__main__":
    unit_test_taylor_with_ojas()
    unit_test_mlp_updates()