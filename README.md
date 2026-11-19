# Q.4 — ML + Quantum + Parallel Computing (Pump Inspection)

**Problem:** 8 pumps, budget B = 6 technician-hours, each inspection takes 1–3 h. Choose which pumps to inspect to cover the most failure risk.

**Run:** `pip install numpy pandas scikit-learn matplotlib` then `python run_all.py` (or run `part_a_ml.py` … `part_d_parallel.py` individually). Seed = 055 everywhere.

| File | Part |
|---|---|
| common.py | data, MLP, QUBO builder, helpers |
| part_a_ml.py | (a) ML |
| part_b_qubo.py | (b) QUBO + brute-force check |
| part_c_grover.py | (c) Grover search |
| part_d_parallel.py | (d) parallel timing |
| results/ | console outputs, pump_predictions.csv, grover.png, parallel.png |

## (a) Machine learning
600 synthetic rows (vibration, run hours, temperature, age → fails within a month), 70/30 split (420/180), `StandardScaler + MLPClassifier(hidden=(8,))`.

Test **accuracy 0.811, precision 0.772, recall 0.792.**

| Pump | P(fail) | Prediction | Hours |
|---|---|---|---|
| 1 | 0.053 | ok | 1 |
| 2 | 0.230 | ok | 2 |
| 3 | 0.452 | ok | 1 |
| 4 | 0.651 | FAILS | 3 |
| 5 | 0.935 | FAILS | 1 |
| 6 | 0.739 | FAILS | 1 |
| 7 | 0.918 | FAILS | 2 |
| 8 | 0.983 | FAILS | 2 |

## (b) QUBO
Variables: x1…x8 (inspect pump i) + 3 slack bits s (weights 1, 2, 4) = **11 bits** (2048 states). The slack bits turn "hours ≤ 6" into an equality.

E(x) = − Σ pᵢxᵢ + P·(Σ cᵢxᵢ + Σ wⱼsⱼ − 6)²,  P = 2

Expanded for binary variables (xᵢ² = xᵢ), with aᵢ = hours (pumps) or slack weight:
Qᵢᵢ = P(aᵢ² − 12aᵢ) − pᵢ (pᵢ only for pumps), Qᵢⱼ = 2P·aᵢaⱼ (i<j), constant offset = 36P = 72. The full 11×11 matrix is printed in `results/part_b_output.txt`.

**Brute-force check:** lowest QUBO energy over all 2048 strings = **−3.5746** at `00001111|000` = pumps 5, 6, 7, 8 (6 h, benefit 3.5746). A separate enumeration of all feasible subsets (cost ≤ 6, maximise Σp) gives the same answer, so the QUBO is correct.

## (c) Grover search (state-vector simulator, NumPy)
Dürr–Høyer / Grover adaptive search: the oracle marks all states with E < threshold; the threshold drops whenever a better state is measured (99 rounds, 1213 Grover iterations). Final amplification with r = ⌊π/4·√2048⌋ = 35 iterations:

- Best answer: **pumps 5, 6, 7, 8**, bit-string `00001111|000`, **energy −3.5746**
- Appeared **4096 / 4096 shots** (probability ≈ 1.00 vs 0.0005 for a random guess)
- Brute force: identical state and energy → **match**
- Final stage used 35 oracle calls vs 2048 classical evaluations (√N speed-up ≈ 45×; the full adaptive search needed ≈ 1250 because the minimum is not known in advance).

## (d) Parallel computing (ProcessPoolExecutor)
All 2¹¹ = 2048 energies are evaluated in equal chunks with `concurrent.futures.ProcessPoolExecutor` (the sweep is repeated 200× so the runtime is measurable). The parallel results match the NumPy energies exactly. Speed-up S = T₁/Tₚ, efficiency = S/p. Machine: 4 CPU cores.

| Workers | Time (s) | Speed-up | Efficiency |
|---|---|---|---|
| 1 | 1.283 | 1.00 | 100 % |
| 2 | 0.755 | 1.70 | 85 % |
| 4 | 0.936 | 1.37 | 34 % |

**Observation:** 2 workers give a good speed-up (1.70×, 85 % efficiency). With 4 workers the speed-up is only 1.37× (34 % efficiency): the workload is small (2048 states), so process start-up, data transfer and scheduling overhead take a large share, and on a laptop the 4 logical cores are not 4 fully independent cores (hyper-threading, other running programs). Efficiency falls as workers increase (Amdahl's law / overhead). Timings vary slightly from run to run.

## Conclusion 
1. Final choice: inspect pumps **5, 6, 7, 8** (all predicted to fail, 6 of 6 hours used, total risk covered 3.57), verified by brute force and Grover.
2. ML gives the risk scores, the QUBO turns "budget ≤ 6" into a penalty with slack bits, and Grover finds the minimum-energy string in ~√N oracle calls.
3. Parallel processing gave 1.70× with 2 workers but only 1.37× with 4, because overhead dominates for such a small problem.
