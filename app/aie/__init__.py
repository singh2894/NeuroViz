"""
Automated Insight Engine — ML pipeline vendored from
github.com/singh2894/Automated-Insight-Engine.

Pure pandas + scikit-learn stages: understanding, diagnostics, cleaning,
features, selection, models, evaluation, explain, fairness, runner.
NeuroViz merge patches: relative imports in runner, optional
xgboost/lightgbm/shap, Plotly figures removed (UI renders Altair).
"""

__all__ = [
    "cleaning",
    "diagnostics",
    "evaluation",
    "fairness",
    "models",
    "runner",
    "selection",
    "understanding",
]
