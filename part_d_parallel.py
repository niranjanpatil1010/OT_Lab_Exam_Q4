"""Q4(d) Parallel energy evaluation of all 2^11 bit-strings with ProcessPoolExecutor."""
import time, numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

REPEATS = 200   # the sweep is repeated (the 'repeated runs' in the question) so there is enough work to time


def eval_chunk(args):
    Q, off, lo, hi, reps = args
    n = Q.shape[0]; Ql = Q.tolist(); out = []
    for _ in range(reps):
        out = []
        for s in range(lo, hi):
            x = [(s >> k) & 1 for k in range(n)]
            e = off
            for i in range(n):
                if x[i]:
                    row = Ql[i]
                    for j in range(i, n):
                        if x[j]:
                            e += row[j]
            out.append(e)
    return out


def run(workers, Q, off, total):
    size = total // workers
    jobs = [(Q, off, w * size, (w + 1) * size, REPEATS) for w in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(eval_chunk, [(Q, off, 0, 2, 1)] * workers))     # warm-up: start the processes
        t = time.perf_counter()
        res = list(ex.map(eval_chunk, jobs))
        dt = time.perf_counter() - t
    return dt, np.concatenate(res)


if __name__ == "__main__":
    import os
    p, c = pump_risk(); Q, off = build_qubo(p, c); total = 2 ** N_BITS
    ref = all_energies(Q, off)
    print("CPU cores available: %d | states: %d | sweep repeated %d times\n" % (os.cpu_count(), total, REPEATS))
    times = {}
    for w in (1, 2, 4):
        best = min(run(w, Q, off, total)[0] for _ in range(3))        # best of 3
        dt, e = run(w, Q, off, total)
        assert np.allclose(e, ref), "parallel result differs!"
        times[w] = best
    print("%-8s %-10s %-10s %-10s" % ("workers", "time (s)", "speed-up", "efficiency"))
    for w in (1, 2, 4):
        s = times[1] / times[w]
        print("%-8d %-10.3f %-10.2f %-10.1f%%" % (w, times[w], s, 100 * s / w))
    print("\nAll parallel energies match the vectorised NumPy result: True")
    print("speed-up S = T1/Tp ; efficiency E = S/p")
    ws = [1, 2, 4]; sp = [times[1] / times[w] for w in ws]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ws, sp, "o-", label="measured"); ax[0].plot(ws, ws, "k--", label="ideal"); ax[0].set_xlabel("workers"); ax[0].set_ylabel("speed-up"); ax[0].legend()
    ax[1].bar([str(w) for w in ws], [100 * s / w for s, w in zip(sp, ws)]); ax[1].set_ylabel("efficiency %"); ax[1].set_xlabel("workers")
    plt.tight_layout(); plt.savefig("results/parallel.png", dpi=130)
