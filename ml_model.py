"""
Part 4: Machine Learning Integration — Linear Regression
Course: Big Data Analytics (CC-342)
Assignment: 02

Dataset: Sales Analytics Dataset
Target (Y): Sales
Features (X): Price, Quantity, Rating
"""

import json
import math
import csv
import os


# ─────────────────────────────────────────────
# Utility: simple stats helpers (no numpy)
# ─────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def mat_mul(A, B):
    """Matrix multiply A (m×n) × B (n×p) → m×p"""
    result = []
    Bt = transpose(B)
    for row_a in A:
        result.append([dot(row_a, col_b) for col_b in Bt])
    return result


def mat_inv_2x2(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return [[M[1][1] / det, -M[0][1] / det],
            [-M[1][0] / det, M[0][0] / det]]


def solve_normal_equation(X, y):
    """
    Solve β = (X^T X)^-1 X^T y using Gaussian elimination.
    Supports any number of features.
    """
    n  = len(X[0])     # features (incl. bias)
    m  = len(X)        # samples

    # Build X^T X  (n×n)
    Xt  = transpose(X)
    XtX = mat_mul(Xt, [row for row in X])

    # Build X^T y  (n×1)
    Xty = [dot(row, y) for row in Xt]

    # Gaussian elimination with partial pivoting
    # Augmented matrix [XtX | Xty]
    aug = [XtX[i][:] + [Xty[i]] for i in range(n)]

    for col in range(n):
        # Pivot
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix — features might be collinear")
        for r in range(n):
            if r != col:
                factor = aug[r][col] / pivot
                aug[r] = [aug[r][j] - factor * aug[col][j] for j in range(n + 1)]

        aug[col] = [v / pivot for v in aug[col]]

    return [aug[i][n] for i in range(n)]


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def load_data():
    rows = []
    with open("dataset.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "Price"    : float(row["Price"]),
                    "Quantity" : float(row["Quantity"]),
                    "Rating"   : float(row["Rating"]),
                    "Sales"    : float(row["Sales"]),
                    "Category" : row["Category"].strip(),
                    "Region"   : row["Region"].strip()
                })
            except (ValueError, KeyError):
                pass
    return rows


def normalize(values):
    """Min-max normalization → [0, 1]"""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    return [(v - mn) / rng for v in values], mn, mx


def denormalize(norm_val, mn, mx):
    return norm_val * (mx - mn) + mn


def train_test_split(data, test_ratio=0.2, seed=42):
    # Deterministic shuffle using seed
    import random
    rng = random.Random(seed)
    data = data[:]
    rng.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def mse(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    y_mean = mean(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────

def run_ml_model(mapreduce_results=None):
    print("\n" + "=" * 60)
    print("  PART 4: Machine Learning — Linear Regression")
    print("=" * 60)

    # ── Load & split ────────────────────────────────────────────
    data = load_data()
    print(f"\n  Dataset size : {len(data)} records")
    print(f"  Features     : Price, Quantity, Rating")
    print(f"  Target       : Sales")

    train_data, test_data = train_test_split(data, test_ratio=0.2)
    print(f"  Train/Test   : {len(train_data)} / {len(test_data)}")

    # ── Extract raw values ──────────────────────────────────────
    def extract(dataset):
        prices    = [r["Price"]    for r in dataset]
        quantities= [r["Quantity"] for r in dataset]
        ratings   = [r["Rating"]   for r in dataset]
        sales     = [r["Sales"]    for r in dataset]
        return prices, quantities, ratings, sales

    tr_price, tr_qty, tr_rat, tr_sales = extract(train_data)
    te_price, te_qty, te_rat, te_sales = extract(test_data)

    # ── Normalize features ──────────────────────────────────────
    norm_price,  mn_p,  mx_p  = normalize(tr_price  + te_price)
    norm_qty,    mn_q,  mx_q  = normalize(tr_qty     + te_qty)
    norm_rat,    mn_r,  mx_r  = normalize(tr_rat     + te_rat)
    norm_sales,  mn_s,  mx_s  = normalize(tr_sales   + te_sales)

    n_tr = len(train_data)
    tr_np = norm_price[:n_tr];  te_np = norm_price[n_tr:]
    tr_nq = norm_qty[:n_tr];    te_nq = norm_qty[n_tr:]
    tr_nr = norm_rat[:n_tr];    te_nr = norm_rat[n_tr:]
    tr_ns = norm_sales[:n_tr];  te_ns = norm_sales[n_tr:]

    # Build design matrix X with bias column
    def build_X(np_, nq_, nr_):
        return [[1.0, np_[i], nq_[i], nr_[i]] for i in range(len(np_))]

    X_train = build_X(tr_np, tr_nq, tr_nr)
    X_test  = build_X(te_np, te_nq, te_nr)

    # ── Fit model ───────────────────────────────────────────────
    print("\n  Training Linear Regression (Normal Equation method)...")
    beta = solve_normal_equation(X_train, tr_ns)
    print(f"\n  Model Coefficients (normalized space):")
    labels = ["Bias (β0)", "Price (β1)", "Quantity (β2)", "Rating (β3)"]
    for lbl, b in zip(labels, beta):
        print(f"    {lbl:<20} : {b:+.6f}")

    # ── Predict ─────────────────────────────────────────────────
    def predict(X, beta):
        return [dot(x, beta) for x in X]

    tr_pred_n = predict(X_train, beta)
    te_pred_n = predict(X_test,  beta)

    # Denormalize predictions and actuals back to original scale
    def denorm_list(lst):
        return [denormalize(v, mn_s, mx_s) for v in lst]

    tr_pred = denorm_list(tr_pred_n)
    te_pred = denorm_list(te_pred_n)
    tr_actuals = tr_sales
    te_actuals = te_sales

    # ── Evaluation ──────────────────────────────────────────────
    tr_r2   = r2_score(tr_actuals, tr_pred)
    te_r2   = r2_score(te_actuals, te_pred)
    te_mse  = mse(te_actuals,  te_pred)
    te_mae  = mae(te_actuals,  te_pred)
    te_rmse = rmse(te_actuals, te_pred)

    print("\n" + "─" * 50)
    print("  Model Evaluation Metrics")
    print("─" * 50)
    print(f"  Training R²  : {tr_r2:.4f}")
    print(f"  Testing  R²  : {te_r2:.4f}")
    print(f"  MSE          : {te_mse:>12,.2f}")
    print(f"  RMSE         : {te_rmse:>12,.2f}")
    print(f"  MAE          : {te_mae:>12,.2f}")
    print("─" * 50)

    # ── Sample Predictions ──────────────────────────────────────
    print("\n  Sample Predictions vs Actual (Test Set)")
    print(f"  {'#':<5} {'Actual Sales':>14} {'Predicted Sales':>16} {'Error':>12}")
    print("  " + "─" * 50)
    for i in range(min(10, len(te_actuals))):
        err = te_pred[i] - te_actuals[i]
        print(f"  {i+1:<5} {te_actuals[i]:>14,.2f} {te_pred[i]:>16,.2f} {err:>+12,.2f}")

    # ── Equation ────────────────────────────────────────────────
    print("\n  Linear Regression Equation (normalized):")
    print(f"  Sales = {beta[0]:+.4f} + {beta[1]:+.4f}·Price_norm "
          f"+ {beta[2]:+.4f}·Qty_norm + {beta[3]:+.4f}·Rating_norm")

    print("\n✔  ML Model training and evaluation complete")

    return {
        "coefficients" : dict(zip(labels, beta)),
        "metrics": {
            "train_r2": tr_r2, "test_r2": te_r2,
            "mse": te_mse, "rmse": te_rmse, "mae": te_mae
        },
        "sample_preds": list(zip(te_actuals[:10], te_pred[:10]))
    }


if __name__ == "__main__":
    run_ml_model()
