class SolveStatus:
    SUCCESS = "Solve_Success"
    MAX_ITER = "MaxIter"
    INFEASIBLE = "Infeasible"
    ERROR = "Error"

TELEMETRY_KEYS = ("obj","status","iters","solve_ms","feasible","num_active","slack_sum")
