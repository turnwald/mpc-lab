import csv
from pathlib import Path
import casadi as ca

class RolloutLogger:
    """Lightweight CSV logger for MPC rollouts."""
    def __init__(self, path: str, header: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        self.rows = []

    def log(self, values: list[float]):
        assert len(values) == len(self.header)
        self.rows.append([float(v) for v in values])

    def save_csv(self):
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)
        return str(self.path)
