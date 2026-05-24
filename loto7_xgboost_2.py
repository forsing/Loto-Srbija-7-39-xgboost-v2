from __future__ import annotations

"""
XGBRegressor i XGBRFRegressor u jednom prolazu
ENSEMBLE prosek oba modela
multi-label 39 izlaza → top-7
vremenski split, poslednjih 100 za back-test
features: lag + rolling frekvencije + gap + statistike
predikcija iz poslednjeg reda CSV-a
validacija 7 jedinstvenih, 1–39, sortirano
snimanje u loto7_xgboost_2_predikcija.txt
ukupno vreme
"""


import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONHASHSEED"] = "39"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

print()
print("XGBoost version:")
print(xgb.__version__)
print()
"""
XGBoost version:
3.0.5
"""


####   regressor.fit(X, y.ravel())


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)

# 1. Učitaj loto podatke
CSV_PATH = "/data/loto7_4620_k41.csv"
OUT_TXT = Path("/loto7_xgboost_2_predikcija.txt")
N_MIN, N_MAX = 1, 39
K = 7
LAG = 5
WINDOWS = (20, 50, 100)
BACKTEST_N = 100

T0 = time.time()
print("START", datetime.today())

df = pd.read_csv(CSV_PATH, header=None).iloc[:, :K].astype(int)


###################################


print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

      0   1   2   3   4   5   6
4615  4   7  11  20  33  34  39
4616  7  14  17  22  24  26  33
4617  4   9  14  16  19  24  32
4618  3   6  12  14  15  22  27
4619  1   3   7  17  24  25  32
"""


draws = np.sort(df.values, axis=1)
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1-39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")


def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def build_features(draws_arr: np.ndarray, y_multi: np.ndarray) -> np.ndarray:
    n, _ = draws_arr.shape
    lag_blocks = []
    for lag in range(1, LAG + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[lag:] = draws_arr[:-lag]
        lag_blocks.append(shifted)
    lag_block = np.concatenate(lag_blocks, axis=1).astype(float)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for window in WINDOWS:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:window + 1] = cum[:window]
        rolled[window + 1:] = cum[window:-1] - cum[:-window - 1]
        rolling_blocks.append(rolled / float(window))
    rolling_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stats = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, rolling_block, gap, stats], axis=1)


def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d: np.ndarray, y_true: np.ndarray) -> float:
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
X_full = build_features(draws, Y_full)
START = max(LAG, max(WINDOWS))

X_all = X_full[START:].astype(float)
Y_all = Y_full[START:].astype(float)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_back, Y_back = X_all[n_train:], Y_all[n_train:]
X_next_raw = X_full[-1:].astype(float)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_back_s = scaler.transform(X_back)
X_next_s = scaler.transform(X_next_raw)


def make_models() -> dict[str, MultiOutputRegressor]:
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=0,
        random_state=39,
        n_jobs=1,
        tree_method="hist",
    )
    # xgb_model = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 1000)
    xgbrf_model = xgb.XGBRFRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.8,
        subsample=0.8,
        learning_rate=0.1,
        max_depth=5,
        reg_alpha=10,
        n_estimators=1000,
        verbosity=0,
        random_state=39,
        n_jobs=1,
        tree_method="hist",
    )
    return {
        "XGBRegressor": MultiOutputRegressor(xgb_model),
        "XGBRFRegressor": MultiOutputRegressor(xgbrf_model),
    }


print()
print(f"Features: X_train={X_train_s.shape}, X_back={X_back_s.shape}, X_next={X_next_s.shape}")
print()

models = make_models()
scores_back = {}
scores_next = {}
preds = {}

for name, model in models.items():
    print(f"Treniram {name} ...")
    model.fit(X_train_s, Y_train)
    scores_back[name] = model.predict(X_back_s)
    scores_next[name] = model.predict(X_next_s)[0]
    preds[name] = topk_from_scores(scores_next[name])
    print(f"Gotovo: {name}")
    print()

scores_back["ENSEMBLE"] = np.mean(np.stack(list(scores_back.values()), axis=0), axis=0)
scores_next["ENSEMBLE"] = np.mean(np.stack(list(scores_next.values()), axis=0), axis=0)
preds["ENSEMBLE"] = topk_from_scores(scores_next["ENSEMBLE"])

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<16} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
for name, scores in scores_back.items():
    h = avg_hits(scores, Y_back)
    a = safe_auc(Y_back, scores)
    l = safe_lrap(Y_back, scores)
    print(f"{name:<16} {h:>8.3f} {100*h/K:>6.1f}% {a:>7.3f} {l:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()

print("Predicted Next Lottery Numbers:")
for name, pick in preds.items():
    assert len(set(pick.tolist())) == K
    assert pick.min() >= N_MIN and pick.max() <= N_MAX
    assert list(pick) == sorted(pick.tolist())
    print(f"{name:<16} -> {pick.tolist()}  ({describe(pick)})")
print()

with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={df.shape[0]}) ---\n")
    for name, pick in preds.items():
        f.write(f"{name:<16} -> {pick.tolist()}  ({describe(pick)})\n")
print(f"Snimljeno u: {OUT_TXT}")



# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4620, Broj pozicija: 7
"""


elapsed = time.time() - T0
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()




"""

XGBoost version:
3.1.3

START 2026-05-24 15:56:59.399684

Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37


Zadnjih 5 ucitanih kombinacija iz CSV fajla:

      0   1   2   3   4   5   6
4615  4   7  11  20  33  34  39
4616  7  14  17  22  24  26  33
4617  4   9  14  16  19  24  32
4618  3   6  12  14  15  22  27
4619  1   3   7  17  24  25  32


Features: X_train=(4420, 195), X_back=(100, 195), X_next=(1, 195)

Treniram XGBRegressor ...
Gotovo: XGBRegressor

Treniram XGBRFRegressor ...
Gotovo: XGBRFRegressor

Back-test (poslednjih 100 izvlačenja):
model              hits/7    hit%     AUC    LRAP
XGBRegressor        1.170   16.7%   0.529   0.244
XGBRFRegressor      1.210   17.3%   0.490   0.242
ENSEMBLE            1.190   17.0%   0.529   0.242
(slučajan baseline ≈ 1.256 hits/7)

Predicted Next Lottery Numbers:
XGBRegressor     -> [2, x, 21, y, 25, z, 30]  (suma=142, neparnih=4/7, niskih(<=19)=2/7, raspon=28)
XGBRFRegressor   -> [8, x, 23, y, 32, z, 37]  (suma=170, neparnih=2/7, niskih(<=19)=2/7, raspon=29)
ENSEMBLE         -> [2, x, 21, y, 25, z, 30]  (suma=137, neparnih=3/7, niskih(<=19)=2/7, raspon=28)

Snimljeno u: /loto7_xgboost_2_predikcija.txt

Učitano kombinacija: 4620, Broj pozicija: 7


STOP 2026-05-24 15:59:15.137354
Ukupno vreme: 0:02:15  (135.7 s)

"""
