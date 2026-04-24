
import math
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
import joblib

x = sp.symbols("x")

FUNC_SPECS = {
    "exp(x)": {"expr": sp.exp(x), "x0_range": (-1.5, 1.5), "delta_range": (-1.4, 1.4)},
    "sin(x)": {"expr": sp.sin(x), "x0_range": (-math.pi, math.pi), "delta_range": (-1.8, 1.8)},
    "cos(x)": {"expr": sp.cos(x), "x0_range": (-math.pi, math.pi), "delta_range": (-1.8, 1.8)},
    "log(1+x)": {"expr": sp.log(1 + x), "x0_range": (-0.65, 1.5), "delta_range": (-0.65, 0.95)},
}

for name, spec in FUNC_SPECS.items():
    expr = spec["expr"]
    spec["func"] = sp.lambdify(x, expr, "numpy")
    spec["derivs"] = [sp.lambdify(x, sp.diff(expr, x, n), "numpy") for n in range(0, 8)]


def taylor_approx(spec, x0, xv, degree):
    delta = xv - x0
    total = 0.0
    for n in range(degree + 1):
        total += spec["derivs"][n](x0) * (delta ** n) / math.factorial(n)
    return total


def build_dataset(n_per_func=3000, threshold=0.01, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for fname, spec in FUNC_SPECS.items():
        x0_lo, x0_hi = spec["x0_range"]
        d_lo, d_hi = spec["delta_range"]
        count = 0

        while count < n_per_func:
            x0 = rng.uniform(x0_lo, x0_hi)
            delta = rng.uniform(d_lo, d_hi)
            xv = x0 + delta

            if fname == "log(1+x)" and (x0 <= -0.95 or xv <= -0.95):
                continue

            true_y = float(spec["func"](xv))
            degree_label = 8

            for degree in range(1, 8):
                pred_y = float(taylor_approx(spec, x0, xv, degree))
                if degree_label == 8 and abs(pred_y - true_y) <= threshold:
                    degree_label = degree

            if degree_label == 8 and rng.random() > 0.45:
                continue

            rows.append({
                "function": fname,
                "x0": x0,
                "x": xv,
                "delta": delta,
                "abs_delta": abs(delta),
                "best_degree": degree_label
            })
            count += 1

    return pd.DataFrame(rows)


def train_models():
    df = build_dataset()
    features = ["function", "x0", "x", "delta", "abs_delta"]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["best_degree"]
    )

    # only 8% labeled
    rng = np.random.default_rng(5)
    labeled_idx = rng.choice(train_df.index, size=int(len(train_df) * 0.08), replace=False)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["function"]),
        ("num", StandardScaler(), ["x0", "x", "delta", "abs_delta"]),
    ])

    supervised = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced")),
    ])
    supervised.fit(train_df.loc[labeled_idx, features], train_df.loc[labeled_idx, "best_degree"])
    sup_acc = accuracy_score(test_df["best_degree"], supervised.predict(test_df[features]))

    ssl_train = train_df.copy()
    ssl_train["label_ssl"] = -1
    ssl_train.loc[labeled_idx, "label_ssl"] = ssl_train.loc[labeled_idx, "best_degree"]

    X_train = preprocessor.fit_transform(ssl_train[features])
    X_test = preprocessor.transform(test_df[features])

    ssl_model = LabelSpreading(kernel="knn", n_neighbors=15, alpha=0.2, max_iter=50)
    ssl_model.fit(X_train, ssl_train["label_ssl"])
    ssl_acc = accuracy_score(test_df["best_degree"], ssl_model.predict(X_test))

    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(ssl_model, "ssl_model.joblib")
    df.to_csv("taylor_ssl_dataset.csv", index=False)

    print(f"Supervised accuracy: {sup_acc:.4f}")
    print(f"Semi-supervised accuracy: {ssl_acc:.4f}")
    print("Saved: preprocessor.joblib, ssl_model.joblib, taylor_ssl_dataset.csv")


if __name__ == "__main__":
    train_models()