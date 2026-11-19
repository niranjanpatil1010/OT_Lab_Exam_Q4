"""Q4(a) Machine learning: synthetic data, 70/30 split, small MLP, metrics, 8-pump predictions."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from common import *

model, Xtr, Xte, ytr, yte = trained()
yp = model.predict(Xte)
print("Dataset: 600 rows, seed 055, 70/30 stratified split ->", len(ytr), "train /", len(yte), "test")
print("Failure rate in data: %.1f%%" % (100 * (np.r_[ytr, yte]).mean()))
print("Model: StandardScaler + MLPClassifier(hidden=(8,), max_iter=2000)\n")
print("Test accuracy : %.4f" % accuracy_score(yte, yp))
print("Test precision: %.4f" % precision_score(yte, yp))
print("Test recall   : %.4f" % recall_score(yte, yp))

p, hours = pump_risk()
Xp, _ = pumps()
df = pd.DataFrame(Xp, columns=FEATURES).round(2)
df.insert(0, "pump", range(1, N_PUMPS + 1))
df["P(fail<=1 month)"] = p.round(4)
df["prediction"] = np.where(p >= 0.5, "FAILS", "ok")
df["inspect_hours"] = hours
print("\nPredictions for the 8 pumps:\n" + df.to_string(index=False))
df.to_csv("results/pump_predictions.csv", index=False)
