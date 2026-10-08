"""U-SafeAgent core: uncertainty-aware safe-set teacher selection (review §6).

Three layers:
  1. eligibility  -- train-only sanity checks (NaN / missing probe / degenerate feature)
  2. estimation   -- one SHARED weighted ridge for the mean utility ``mu`` (pp) plus a
                     dataset-cluster "group jackknife" spread as a model-uncertainty
                     proxy ``u_model`` (3 fits only -> sensitivity proxy, NOT a
                     conformal or 95% guarantee)
  3. decision     -- safe set ``S(D) = {T : mu - kappa*u > delta}`` and
                     ``a(D) = argmax_{T in S} (mu - kappa*u)`` else ``None``

All quantities are in **pp** (percentage points). Test data never enters this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_DELTA_PP = 0.20   # project pre-registered "Strong Positive" bar
DEFAULT_KAPPA = 1.0
DEFAULT_EPS_PP = 0.0      # IoU-consistency gate slack


def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Solve (X^T W X + lam I) beta = X^T W y with an intercept column."""
    Xa = np.concatenate([X, np.ones((len(X), 1), dtype=np.float64)], axis=1)
    W = w.reshape(-1, 1)
    XtW = (Xa * W).T
    A = XtW @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float64)
    A[-1, -1] -= lam  # do not regularize the intercept
    b = XtW @ y
    return np.linalg.solve(A, b)


@dataclass
class Decision:
    teacher: str | None
    margin_pp: float
    mu_pp: float
    u_pp: float
    reason: str
    per_teacher: dict = field(default_factory=dict)


class SafeSelector:
    def __init__(self, lam: float = 1.0, delta_pp: float = DEFAULT_DELTA_PP,
                 kappa: float = DEFAULT_KAPPA):
        self.lam = float(lam)
        self.delta_pp = float(delta_pp)
        self.kappa = float(kappa)
        self.feature_names: list[str] = []
        self._x_mu = None
        self._x_sd = None
        self._y_mu = 0.0
        self._y_sd = 1.0
        self._beta = None
        self._jackknife_betas: list[np.ndarray] = []
        self._train_groups: list[str] = []

    # ---------------------------------------------------------------- fitting
    def fit(self, X: np.ndarray, y: np.ndarray, groups, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        groups = list(groups)
        # P2-A (review 2026-10-08): a chained comparison is NOT an equality test --
        # ``a != b != c`` means ``a != b and b != c``, so ``len(X)=1, len(y)=2,
        # len(groups)=1`` would have passed silently. Compare all three explicitly.
        if not (len(X) == len(y) == len(groups)):
            raise ValueError(
                f"X / y / groups length mismatch: {len(X)} / {len(y)} / {len(groups)}")
        if len(set(groups)) < 2:
            raise ValueError("need >=2 training datasets for a group jackknife")
        self.feature_names = list(feature_names or [f"f{i}" for i in range(X.shape[1])])

        self._x_mu, self._x_sd = _standardize_fit(X)
        self._y_mu = float(y.mean())
        self._y_sd = float(y.std()) or 1.0
        Xs = (X - self._x_mu) / self._x_sd
        ys = (y - self._y_mu) / self._y_sd

        # dataset-cluster balancing: each dataset contributes equal total weight
        counts = {g: groups.count(g) for g in set(groups)}
        w = np.array([1.0 / counts[g] for g in groups], dtype=np.float64)
        w = w / w.mean()

        self._beta = _weighted_ridge(Xs, ys, w, self.lam)

        # group jackknife: refit leaving one TRAINING dataset out
        self._train_groups = sorted(set(groups))
        self._jackknife_betas = []
        for g in self._train_groups:
            keep = np.array([gg != g for gg in groups])
            if keep.sum() < 2:
                continue
            Xk, yk = Xs[keep], ys[keep]
            wk = w[keep]
            wk = wk / wk.mean()
            self._jackknife_betas.append(_weighted_ridge(Xk, yk, wk, self.lam))
        # P2-A: without at least one leave-one-domain refit there is NO uncertainty
        # information at all. Returning a zero spread here would manufacture false
        # certainty, so refuse to fit instead.
        if not self._jackknife_betas:
            raise ValueError(
                "no leave-one-domain refit could be formed (need >=2 domains with "
                ">=3 members) -- refusing to report a zero (fake-certain) spread")
        return self

    # ------------------------------------------------------------- prediction
    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self._x_mu) / self._x_sd
        return np.concatenate([Xs, np.ones((len(Xs), 1), dtype=np.float64)], axis=1)

    def predict_mu_pp(self, X: np.ndarray) -> np.ndarray:
        return (self._design(X) @ self._beta) * self._y_sd + self._y_mu

    def model_spread_pp(self, X: np.ndarray) -> np.ndarray:
        """SD of the leave-one-training-dataset-out predictions.

        **This is a sensitivity proxy, NOT a 95% CI and NOT a conformal bound.**
        With only three training domains there are three refits, so this number
        carries no coverage guarantee and must never be described as one.
        """
        if not self._jackknife_betas:
            raise RuntimeError(
                "model_spread_pp called without a fitted group jackknife; a zero "
                "spread would be false certainty (P2-A)")
        Xa = self._design(X)
        preds = np.stack([(Xa @ b) * self._y_sd + self._y_mu for b in self._jackknife_betas], axis=0)
        return preds.std(axis=0)

    # --------------------------------------------------------------- decision
    def decide(self, X_teachers: np.ndarray, teacher_ids, u_noise_pp: float = 0.0,
               u_shift_pp: float = 0.0, eligibility=None) -> Decision:
        X = np.asarray(X_teachers, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("non-finite teacher features (missing probe must be explicit)")

        mu = self.predict_mu_pp(X)
        u_model = self.model_spread_pp(X)
        u = np.sqrt(u_model ** 2 + float(u_noise_pp) ** 2 + float(u_shift_pp) ** 2)
        margin = mu - self.kappa * u

        per_teacher = {}
        candidates = []
        for i, tid in enumerate(teacher_ids):
            ok = True if eligibility is None else bool(eligibility(tid, X[i]))
            per_teacher[tid] = {
                "mu_pp": float(mu[i]),
                "u_model_pp": float(u_model[i]),
                "u_noise_pp": float(u_noise_pp),
                "u_pp": float(u[i]),
                "margin_pp": float(margin[i]),
                "eligible": ok,
            }
            if ok and margin[i] > self.delta_pp:
                candidates.append((float(margin[i]), tid))

        if not candidates:
            return Decision(None, 0.0, 0.0, 0.0, "no_supported_positive_utility", per_teacher)
        best_margin, best_tid = max(candidates)
        i = list(teacher_ids).index(best_tid)
        return Decision(best_tid, best_margin, float(mu[i]), float(u[i]),
                        "accepted_by_predeclared_margin", per_teacher)


class RidgeSelector:
    """AG-01 baseline: identical features, ridge mean only, plain ``mu > 0`` gate."""

    def __init__(self, lam: float = 1.0):
        self.inner = SafeSelector(lam=lam, delta_pp=0.0, kappa=0.0)

    def fit(self, X, y, groups, feature_names=None):
        self.inner.fit(X, y, groups, feature_names)
        return self

    def decide(self, X_teachers, teacher_ids, u_noise_pp: float = 0.0, u_shift_pp: float = 0.0,
               eligibility=None) -> Decision:
        return self.inner.decide(X_teachers, teacher_ids, u_noise_pp=0.0, u_shift_pp=0.0,
                                 eligibility=eligibility)


__all__ = ["SafeSelector", "RidgeSelector", "Decision", "DEFAULT_DELTA_PP", "DEFAULT_KAPPA"]
