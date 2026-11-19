"""Q4(c) Grover search (amplitude amplification) on a numpy state-vector simulator.
Minimum finding = Durr-Hoyer / Grover Adaptive Search: oracle marks every state with E < threshold,
threshold is lowered each time a better state is measured."""
import math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

N = 2 ** N_BITS
p, c = pump_risk()
Q, off = build_qubo(p, c)
E = all_energies(Q, off)
rng = np.random.default_rng(SEED)


def grover_probs(mask, r):
    psi = np.full(N, 1 / math.sqrt(N))              # H^11 |0>  : uniform superposition
    for _ in range(r):
        psi[mask] *= -1                             # oracle: phase flip marked states
        psi = 2 * psi.mean() - psi                  # diffusion: inversion about the mean
    return psi ** 2


# ---- Phase 1: Grover Adaptive Search (does NOT know the answer) ----
cur = int(rng.integers(N)); thr = E[cur]
m, lam, used, rounds = 1.0, 6 / 5, 0, 0
limit = 22.5 * math.sqrt(N) + 1.4 * math.log2(N) ** 2
while used < limit:
    r = int(rng.integers(0, int(m) + 1))
    pr = grover_probs(E < thr, r); used += r; rounds += 1
    y = int(rng.choice(N, p=pr / pr.sum()))        # measurement
    if E[y] < thr:
        cur, thr, m = y, E[y], 1.0
    else:
        m = min(lam * m, math.sqrt(N))
print("Phase 1 - Grover Adaptive Search: %d rounds, %d Grover iterations (oracle calls)" % (rounds, used))
print("  best state found: %s  E = %.4f" % (bitstring(cur), E[cur]))

# ---- Phase 2: final amplification with the optimal number of iterations ----
mask = E <= E[cur] + 1e-9
k = int(mask.sum())
r_opt = int(math.floor(math.pi / 4 * math.sqrt(N / k)))
pr = grover_probs(mask, r_opt)
shots = 4096
counts = np.bincount(rng.choice(N, size=shots, p=pr / pr.sum()), minlength=N)
top = int(np.argmax(counts))
print("\nPhase 2 - amplify E <= %.4f : %d marked state(s), optimal iterations r = floor(pi/4*sqrt(N/k)) = %d" % (E[cur], k, r_opt))
print("  Probability of the best answer after Grover: %.4f (uniform guess would be %.5f)" % (pr[top], 1 / N))
print("\nGROVER RESULT (4096 shots)")
print("  best bit-string : %s" % bitstring(top))
print("  pumps to inspect: %s" % (np.where([(top >> k_) & 1 for k_ in range(N_PUMPS)])[0] + 1).tolist())
print("  energy          : %.4f" % E[top])
print("  appeared        : %d / %d shots (%.1f%%)" % (counts[top], shots, 100 * counts[top] / shots))
print("  next most frequent outcomes:")
for i in np.argsort(counts)[::-1][1:4]:
    print("     %s  E=%.4f  count=%d" % (bitstring(int(i)), E[i], counts[i]))

bf = int(np.argmin(E))
print("\nBRUTE FORCE (all 2048 states): %s  E = %.4f" % (bitstring(bf), E[bf]))
print("Grover == brute force ?", "YES" if top == bf else "NO")
print("Cost: brute force = 2048 energy evaluations; Grover (final stage) = %d oracle calls, "
      "search total = %d. Quadratic speed-up in theory ~ sqrt(N)=%d." % (r_opt, used + r_opt, int(math.sqrt(N))))
print("(Simulator note: on a real device the oracle is a reversible circuit computing E<threshold; "
      "here it is simulated by a classical energy table.)")

# figure
rs = list(range(0, 31)); pk = [grover_probs(mask, r)[top] for r in rs]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(rs, pk, "o-"); ax[0].axvline(r_opt, color="r", ls="--", label="r = %d" % r_opt)
ax[0].set_xlabel("Grover iterations"); ax[0].set_ylabel("P(best state)"); ax[0].legend(); ax[0].set_title("Amplitude amplification")
top10 = np.argsort(counts)[::-1][:6]
ax[1].bar(range(6), counts[top10]); ax[1].set_xticks(range(6)); ax[1].set_xticklabels([bitstring(int(i)).split(" |")[0] for i in top10], rotation=45, fontsize=7)
ax[1].set_title("Measurement counts (4096 shots)"); plt.tight_layout(); plt.savefig("results/grover.png", dpi=130)
