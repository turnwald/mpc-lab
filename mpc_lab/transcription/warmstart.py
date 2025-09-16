import numpy as np

def shift_trajectory(X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-shift warm start for receding horizon.
    X: (N+1,nx), U: (N,nu) -> shift left by 1, repeat last.
    """
    Xs = np.concatenate([X[1:], X[-1:]], axis=0)
    Us = np.concatenate([U[1:], U[-1:]], axis=0)
    return Xs, Us
