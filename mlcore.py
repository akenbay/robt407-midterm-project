"""mlcore: shared NumPy toolkit for the ROBT 407 project.

Conventions
-----------
* ``X`` never includes the bias column. Call ``add_bias(X)`` to prepend a
  column of ones. Functions that take a weight vector ``w`` together with
  ``X`` (the ``*_Ein`` / ``*_grad`` functions) expect ``X`` to already have
  the bias column, so ``w`` has shape ``(d+1,)``.
* Classification labels are ``+1`` / ``-1``. The positive class is ``+1``.
* Any function that uses randomness takes an ``rng`` argument
  (``np.random.Generator``) and never seeds itself. Create one with
  ``rng = np.random.default_rng(seed)`` in the calling code.
* Code is vectorized: there are no Python loops over data points. Loops over
  features, folds or polynomial degrees are allowed.

Shapes use ``N`` for the number of points and ``d`` for the number of
features (without bias).

Only NumPy (and ``scipy.special``) are used here. No scikit-learn.
"""

from itertools import combinations_with_replacement

import numpy as np
from scipy.special import expit

__all__ = [
    "add_bias",
    "legendre_features",
    "poly_exponents",
    "poly_features",
    "standardize_fit",
    "standardize_apply",
    "kfold_indices",
    "confusion_matrix",
    "error",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_curve",
    "auc",
    "gradient_check",
    "sigmoid",
    "linreg_Ein",
    "linreg_grad",
    "logistic_Ein",
    "logistic_grad",
]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _as_2d(X):
    """Return X as a float 2D array. A 1D input of shape (N,) becomes (N, 1)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"expected 1D or 2D input, got shape {X.shape}")
    return X


def add_bias(X):
    """Prepend a column of ones.

    X: (N,) or (N, d). A 1D input is treated as N points with one feature.
    Returns: (N, d+1), first column all ones.
    """
    X = _as_2d(X)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def legendre_features(x, Q):
    """Legendre polynomial features [L0(x), ..., LQ(x)].

    Uses the recursion L0 = 1, L1 = x,
    Lk = ((2k-1)/k) * x * L_{k-1} - ((k-1)/k) * L_{k-2}.

    x: (N,) or (N, 1), values usually in [-1, 1].
    Q: int >= 0, highest degree.
    Returns: (N, Q+1). Column k is Lk(x); column 0 is all ones.
    """
    if Q < 0:
        raise ValueError("Q must be >= 0")
    x = np.asarray(x, dtype=float).ravel()
    L = np.empty((x.shape[0], Q + 1))
    L[:, 0] = 1.0
    if Q >= 1:
        L[:, 1] = x
    for k in range(2, Q + 1):
        L[:, k] = ((2 * k - 1) / k) * x * L[:, k - 1] - ((k - 1) / k) * L[:, k - 2]
    return L


def poly_exponents(d, Q):
    """Exponent tuples for the columns of poly_features(X, Q) with d inputs.

    Order: by total degree 1, 2, ..., Q; within a degree, in the order of
    itertools.combinations_with_replacement(range(d), degree). Each tuple has
    length d and gives the power of each input feature.
    Example (d=2, Q=2): [(1,0), (0,1), (2,0), (1,1), (0,2)]
    i.e. x1, x2, x1^2, x1*x2, x2^2.

    Returns: list of length M = C(d+Q, Q) - 1 of tuples of ints.
    """
    exps = []
    for q in range(1, Q + 1):
        for combo in combinations_with_replacement(range(d), q):
            e = [0] * d
            for j in combo:
                e[j] += 1
            exps.append(tuple(e))
    return exps


def poly_features(X, Q, return_exponents=False):
    """All monomials of the inputs with total degree 1..Q (no constant column).

    Column order is the one documented in poly_exponents(d, Q).

    X: (N,) or (N, d).
    Q: int >= 1, highest total degree.
    return_exponents: if True, also return the list of exponent tuples.
    Returns: Phi of shape (N, M) with M = C(d+Q, Q) - 1,
             or (Phi, exponents) if return_exponents is True.
    """
    X = _as_2d(X)
    d = X.shape[1]
    cols = []
    for q in range(1, Q + 1):
        for combo in combinations_with_replacement(range(d), q):
            cols.append(np.prod(X[:, list(combo)], axis=1))
    Phi = np.column_stack(cols) if cols else np.empty((X.shape[0], 0))
    if return_exponents:
        return Phi, poly_exponents(d, Q)
    return Phi


def standardize_fit(X):
    """Per-feature mean and standard deviation, computed on training data.

    X: (N,) or (N, d).
    Returns: (mu, sd), each of shape (d,). sd is set to 1 where it is 0, so
    constant features map to 0 instead of nan.
    """
    X = _as_2d(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def standardize_apply(X, mu, sd):
    """Apply (X - mu) / sd using statistics from standardize_fit.

    X: (N,) or (N, d); mu, sd: (d,).
    Returns: (N, d).
    """
    return (_as_2d(X) - mu) / sd


def kfold_indices(N, K, rng):
    """Random K-fold split of indices 0..N-1.

    The indices are shuffled with rng.permutation and split into K folds with
    np.array_split, so fold sizes differ by at most 1.

    N: number of points. K: number of folds (2 <= K <= N).
    rng: np.random.Generator.
    Returns: list of K tuples (train_idx, val_idx) of int arrays. Each index
    is in exactly one val_idx; train_idx is all the other indices.
    """
    if not 2 <= K <= N:
        raise ValueError("need 2 <= K <= N")
    folds = np.array_split(rng.permutation(N), K)
    return [
        (np.concatenate([folds[j] for j in range(K) if j != k]), folds[k])
        for k in range(K)
    ]


# ---------------------------------------------------------------------------
# Classification metrics (labels in {+1, -1}, positive class +1)
# ---------------------------------------------------------------------------

def _labels(y, yhat):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same length")
    return y, yhat


def _safe_div(a, b):
    return float(a) / b if b != 0 else 0.0


def confusion_matrix(y, yhat):
    """Confusion counts for labels in {+1, -1}.

    y, yhat: (N,) true and predicted labels.
    Returns: dict with int keys "TP", "FP", "FN", "TN".
      TP: y=+1, yhat=+1    FP: y=-1, yhat=+1
      FN: y=+1, yhat=-1    TN: y=-1, yhat=-1
    As a 2x2 table (rows = predicted +1/-1, cols = true +1/-1):
      [[TP, FP],
       [FN, TN]]
    """
    y, yhat = _labels(y, yhat)
    pos, ppos = y == 1, yhat == 1
    return {
        "TP": int(np.sum(pos & ppos)),
        "FP": int(np.sum(~pos & ppos)),
        "FN": int(np.sum(pos & ~ppos)),
        "TN": int(np.sum(~pos & ~ppos)),
    }


def error(y, yhat):
    """Fraction misclassified. y, yhat: (N,). Returns float."""
    y, yhat = _labels(y, yhat)
    return _safe_div(np.sum(y != yhat), y.size)


def accuracy(y, yhat):
    """Fraction correctly classified. y, yhat: (N,). Returns float."""
    y, yhat = _labels(y, yhat)
    return _safe_div(np.sum(y == yhat), y.size)


def precision(y, yhat):
    """TP / (TP + FP), 0.0 if no positive predictions. y, yhat: (N,)."""
    c = confusion_matrix(y, yhat)
    return _safe_div(c["TP"], c["TP"] + c["FP"])


def recall(y, yhat):
    """TP / (TP + FN), 0.0 if no positive labels. y, yhat: (N,)."""
    c = confusion_matrix(y, yhat)
    return _safe_div(c["TP"], c["TP"] + c["FN"])


def f1(y, yhat):
    """Harmonic mean of precision and recall, 0.0 if both are 0. y, yhat: (N,)."""
    p, r = precision(y, yhat), recall(y, yhat)
    return _safe_div(2 * p * r, p + r)


def roc_curve(y, scores):
    """ROC curve from real-valued scores (higher = more likely +1).

    A point is predicted +1 when score >= threshold. Thresholds are the
    distinct scores in descending order, preceded by +inf. Tied scores are
    treated as one group, so a tie moves the curve diagonally instead of
    depending on the order of the tied points.

    y: (N,) labels in {+1, -1}. scores: (N,).
    Returns: fpr, tpr, thresholds, each of shape (T+1,) where T is the number
    of distinct scores. The curve starts at (0, 0) (threshold +inf) and ends
    at (1, 1) (threshold = min score). If a class is missing, its rate is 0.
    """
    y = np.asarray(y).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s, kind="mergesort")
    s, pos = s[order], (y[order] == 1)

    # last index of each group of tied scores
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tp = np.cumsum(pos)[last]
    fp = np.cumsum(~pos)[last]

    P, Nn = pos.sum(), (~pos).sum()
    tpr = np.r_[0.0, tp / P if P > 0 else np.zeros_like(tp, dtype=float)]
    fpr = np.r_[0.0, fp / Nn if Nn > 0 else np.zeros_like(fp, dtype=float)]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under a curve by the trapezoidal rule.

    fpr, tpr: (T,) points of the curve, fpr non-decreasing.
    Returns: float.
    """
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def gradient_check(f, grad, w, h=1e-6):
    """Compare grad(w) to a central-difference estimate of the gradient of f.

    f: function w -> float. grad: function w -> (D,).
    w: (D,) point to check. h: step size.
    Returns: relative error ||g - g_num|| / max(||g|| + ||g_num||, 1e-12).
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(grad(w), dtype=float).ravel()
    g_num = np.empty(w.size)
    for i in range(w.size):  # loop over parameters, not data points
        e = np.zeros_like(w)
        e.flat[i] = h
        g_num[i] = (f(w + e) - f(w - e)) / (2 * h)
    num = np.linalg.norm(g - g_num)
    den = max(np.linalg.norm(g) + np.linalg.norm(g_num), 1e-12)
    return float(num / den)


def sigmoid(s):
    """Logistic function 1 / (1 + exp(-s)), stable for large |s|.

    s: scalar or array of any shape. Returns: same shape, values in [0, 1].
    """
    return expit(s)


def linreg_Ein(w, X, y):
    """Squared error Ein = (1/N) ||Xw - y||^2.

    w: (d+1,). X: (N, d+1) with bias. y: (N,). Returns: float.
    """
    r = X @ w - y
    return float(r @ r / X.shape[0])


def linreg_grad(w, X, y):
    """Gradient of linreg_Ein: (2/N) X^T (Xw - y).

    w: (d+1,). X: (N, d+1) with bias. y: (N,). Returns: (d+1,).
    """
    return (2.0 / X.shape[0]) * (X.T @ (X @ w - y))


def logistic_Ein(w, X, y):
    """Cross-entropy Ein = (1/N) sum log(1 + exp(-y_n w^T x_n)).

    Computed with np.logaddexp(0, -y*s), so it stays finite for huge w.
    w: (d+1,). X: (N, d+1) with bias. y: (N,) in {+1, -1}. Returns: float.
    """
    s = X @ w
    return float(np.mean(np.logaddexp(0.0, -y * s)))


def logistic_grad(w, X, y):
    """Gradient of logistic_Ein: -(1/N) sum y_n x_n sigmoid(-y_n w^T x_n).

    w: (d+1,). X: (N, d+1) with bias. y: (N,) in {+1, -1}. Returns: (d+1,).
    """
    s = X @ w
    return -(X.T @ (y * sigmoid(-y * s))) / X.shape[0]
