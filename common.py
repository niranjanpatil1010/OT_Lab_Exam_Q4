"""Shared helpers for Q.4 (ML + Quantum + Parallel). Seed = 055 everywhere."""
import numpy as np
from functools import lru_cache
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

SEED = 55
FEATURES = ["vibration_mm_s", "run_hours", "temperature_C", "age_years"]
N_PUMPS, BUDGET = 8, 6
SLACK_W = np.array([1, 2, 4])          # 3 slack bits -> can encode 0..7 unused hours
N_BITS = N_PUMPS + len(SLACK_W)        # 8 + 3 = 11 yes/no choices
PENALTY = 2.0                          # > max benefit of one pump (<1), so cheating never pays


def _features(rng, n):
    return np.column_stack([
        np.clip(rng.normal(4.5, 1.5, n), 0.5, 10),   # vibration
        rng.uniform(100, 8000, n),                    # run hours
        rng.normal(70, 10, n),                        # temperature
        rng.uniform(1, 20, n)])                       # age


def make_dataset(n=600):
    rng = np.random.default_rng(SEED)
    X = _features(rng, n)
    z = (0.9 * (X[:, 0] - 4.5) / 1.5 + 0.8 * (X[:, 1] - 4000) / 2300
         + 0.7 * (X[:, 2] - 70) / 10 + 0.6 * (X[:, 3] - 10) / 5.8)
    p = 1 / (1 + np.exp(-(1.3 * z - 0.8)))
    y = (rng.random(n) < p).astype(int)               # 1 = fails within a month
    return X, y


@lru_cache(maxsize=1)
def trained():
    X, y = make_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    model = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=SEED))
    model.fit(Xtr, ytr)
    return model, Xtr, Xte, ytr, yte


def pumps():
    """8 pumps: features + inspection hours (1-3 h each)."""
    rng = np.random.default_rng(SEED + 1)
    Xp = _features(rng, N_PUMPS)
    hours = rng.integers(1, 4, N_PUMPS)
    return Xp, hours


def pump_risk():
    model = trained()[0]
    Xp, hours = pumps()
    return model.predict_proba(Xp)[:, 1], hours


def build_qubo(p, c, P=PENALTY):
    """E(x) = -sum p_i x_i + P*(sum c_i x_i + sum w_j s_j - B)^2  as upper-triangular Q + offset."""
    a = np.concatenate([c, SLACK_W]).astype(float)
    Q = np.zeros((N_BITS, N_BITS))
    for i in range(N_BITS):
        Q[i, i] = P * (a[i] ** 2 - 2 * BUDGET * a[i])      # x^2 = x for binary
        for j in range(i + 1, N_BITS):
            Q[i, j] = 2 * P * a[i] * a[j]
    Q[np.arange(N_PUMPS), np.arange(N_PUMPS)] -= p          # reward for inspecting risky pumps
    return Q, P * BUDGET ** 2


def all_bits():
    idx = np.arange(2 ** N_BITS)
    return ((idx[:, None] >> np.arange(N_BITS)) & 1).astype(float)   # bit k of index = x_k


def all_energies(Q, offset):
    X = all_bits()
    return np.einsum("ni,ij,nj->n", X, Q, X) + offset


def bitstring(i):
    s = "".join(str((i >> k) & 1) for k in range(N_BITS))
    return s[:N_PUMPS] + " | " + s[N_PUMPS:]               # pumps | slack
