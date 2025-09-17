"""Quick environment verification: solver availability and two smoke demos.

This file ensures the package root is on sys.path so imports like
`mpc_lab.examples.nmpc_attitude_demo` work even when invoked as a script:
    python mpc_lab/verify_env.py
"""
import os
import sys
import importlib
import casadi as ca

# --- Ensure package root on sys.path ---
# __file__ = .../mpc_lab/verify_env.py  → add parent dir to sys.path
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent  = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

def main():
    print("== Verifying CasADi nlpsol plugins (probing) ==")
    for plugin in ["ipopt", "fatrop", "sqpmethod"]:
        try:
            s = ca.nlpsol("s", plugin, {"x": ca.MX.sym("x"), "f": ca.MX(0)})
            print(f"{plugin:<9}: True")
        except Exception:
            print(f"{plugin:<9}: False")

    # Run demos if available
    try:
        print("\n== Attitude NMPC demo ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_attitude_demo")
        mod.main()
    except Exception as e:
        print("Attitude demo failed:", e)

    try:
        print("\n== Rover NMPC demo ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_rover_demo")
        mod.main()
    except Exception as e:
        print("Rover demo failed:", e)


    try:
        print("\n== Attitude NMPC closed-loop rollout ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_attitude_rollout")
        mod.main()
    except Exception as e:
        print("Attitude rollout failed:", e)

    try:
        print("\n== Rover NMPC closed-loop rollout ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_rover_rollout")
        mod.main()
    except Exception as e:
        print("Rover rollout failed:", e)



    try:
        print("\n== Attitude NMPC TV rollout (with disturbance & mismatch) ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_attitude_tv_rollout")
        mod.main()
    except Exception as e:
        print("Attitude TV rollout failed:", e)

    try:
        print("\n== Rover NMPC TV rollout (with disturbance & mismatch) ==")
        mod = importlib.import_module("mpc_lab.examples.nmpc_rover_tv_rollout")
        mod.main()
    except Exception as e:
        print("Rover TV rollout failed:", e)


if __name__ == "__main__":
    main()
