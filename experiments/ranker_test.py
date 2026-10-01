"""Can the coarse-scan summary features predict which refined sites have a lake / a strong score?

Joins prod/*/recs (coarse records) with prod/*/results.jsonl (refinement outcomes) and cross-validates
a small gradient-boosted classifier. Needs scikit-learn.
"""
import glob

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from tdsearch.review import load_results

rows = load_results()
recs = np.concatenate([np.load(f)["recs"] for f in glob.glob("prod/*/recs/chunk_*.npz")])
index = {(int(r["seed"]), int(r["ci"]), int(r["cj"])): k for k, r in enumerate(recs)}
names = ["e", "p5", "m1", "m2", "m3", "r1", "nw3", "nw5", "nw9"]
X, has_lake, good = [], [], []
for r in rows:
    k = index.get((r["seed"], r["ci"], r["cj"]))
    if k is None:
        continue
    X.append([float(recs[n][k]) for n in names])
    has_lake.append(bool(r["lakes"]))
    good.append(r["best"] >= 3000)
X, has_lake, good = np.array(X), np.array(has_lake), np.array(good)
print(f"{len(X)} refined sites joined; {has_lake.sum()} with a lake; {good.sum()} scoring >= 3000")
for label, y in (("has lake", has_lake), ("score >= 3000", good)):
    model = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.08)
    p = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    order = np.argsort(-p)
    print(f"{label}: CV AUC {roc_auc_score(y, p):.3f}; refining only the top 25% keeps "
          f"{y[order[:len(y) // 4]].sum() / y.sum():.0%} of positives, top 40% keeps "
          f"{y[order[:int(len(y) * 0.4)]].sum() / y.sum():.0%}")
