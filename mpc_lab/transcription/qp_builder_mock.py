import numpy as np
from typing import Dict

class MockOCPProblem(dict):
    pass

class MockQPBuilder:
    """Builds a trivial convex QP placeholder used for smoke testing."""
    def build(self, A_seq, B_seq, d_seq, Q, R, P, x_bounds, u_bounds, linear_ineq=None, soften=None):
        N, nx, _ = A_seq.shape
        nu = B_seq.shape[2]
        prob = MockOCPProblem()
        prob['dims'] = {'N': N, 'nx': nx, 'nu': nu}
        return prob

class MockQPFacade:
    def __init__(self): self.prob = None
    def compile(self, problem: dict, opts=None): self.prob = problem
    def solve(self, params: Dict[str, np.ndarray], warmstart=None):
        N = self.prob['dims']['N']; nx = self.prob['dims']['nx']; nu = self.prob['dims']['nu']
        X = np.zeros((N+1, nx))
        U = np.zeros((N, nu))
        return {'x': X, 'u': U, 'obj': 0.0, 'status': 'Solve_Success', 'iters': 1}
