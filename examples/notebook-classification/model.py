"""
Breast cancer classification pipeline. Agent edits this file.
Baseline: SelectKBest feature selection + LogisticRegression.
"""

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

K_BEST_FEATURES = 15
C_REGULARIZATION = 1.0
MAX_ITER = 1000
SOLVER = "lbfgs"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def make_pipeline():
    """Return an sklearn Pipeline for breast cancer classification."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)),
        ("classifier", LogisticRegression(
            C=C_REGULARIZATION,
            max_iter=MAX_ITER,
            solver=SOLVER,
            random_state=RANDOM_STATE,
        )),
    ])
