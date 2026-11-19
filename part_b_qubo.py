"""Q4(b) QUBO formulation + brute-force verification."""
import numpy as np
from itertools import product
from common import *

p, c = pump_risk()
Q, off = build_qubo(p, c)
E = all_energies(Q, off)

print("QUBO (minimise):  E(x) = - sum_i p_i x_i  +  P * ( sum_i c_i x_i + sum_j w_j s_j - B )^2")
print("  p_i = MLP failure probability, c_i = inspection hours, B = %d, P = %.1f" % (BUDGET, PENALTY))
print("  slack bits s_j with weights w = %s turn 'cost <= B' into an equality (8 pumps + 3 slack = 11 bits)" % SLACK_W.tolist())
print("  expanded: Q_ii = P(a_i^2 - 2Ba_i) [- p_i for pumps],  Q_ij = 2P a_i a_j (i<j),  offset = P B^2 = %.1f\n" % off)
np.set_printoptions(linewidth=200, precision=2, suppress=True)
print("p =", np.round(p, 4), "\nc =", c, "\n\nQ matrix (11x11, upper triangular):\n", Q)

best_i = int(np.argmin(E))
x = [(best_i >> k) & 1 for k in range(N_BITS)]
print("\nQUBO lowest energy over all 2048 strings: E = %.4f at %s" % (E[best_i], bitstring(best_i)))

# independent classical check: enumerate 2^8 pump subsets with cost <= B, maximise sum p
best, best_s = -1, None
for s in product([0, 1], repeat=N_PUMPS):
    s = np.array(s)
    if s @ c <= BUDGET and s @ p > best:
        best, best_s = s @ p, s
print("Direct constrained brute force (cost<=6, max sum p): pumps %s, benefit %.4f, hours used %d"
      % ((np.where(best_s)[0] + 1).tolist(), best, best_s @ c))
ok = np.array_equal(best_s, x[:N_PUMPS]) and abs(E[best_i] + best) < 1e-9
print("QUBO minimum == constrained optimum ?  ", "YES - verified" if ok else "NO - increase penalty")
print("Slack consistent (hours + slack = 6):", int(np.dot(c, x[:N_PUMPS]) + np.dot(SLACK_W, x[N_PUMPS:])) == BUDGET)
print("\nSelected pumps to inspect: %s  (hours used %d of %d)" % ((np.where(best_s)[0] + 1).tolist(), best_s @ c, BUDGET))
