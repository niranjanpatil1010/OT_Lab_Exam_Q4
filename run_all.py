"""Runs every part and stores the console output in results/."""
import subprocess, sys
for name, out in [("part_a_ml.py", "part_a"), ("part_b_qubo.py", "part_b"),
                  ("part_c_grover.py", "part_c"), ("part_d_parallel.py", "part_d")]:
    r = subprocess.run([sys.executable, name], capture_output=True, text=True)
    open(f"results/{out}_output.txt", "w").write(r.stdout + r.stderr)
    print("=" * 30, name, "=" * 30); print(r.stdout + r.stderr)
