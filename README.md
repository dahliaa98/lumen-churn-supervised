# Lumen - Churn Prediction & Watch-Hour Regression

Two supervised ML projects on Lumen subscription data:
1. Classification: predict 30-day churn for paid users
2. Regression: predict next-month watch hours

## Stack
Python 3.13 | scikit-learn | pandas | NumPy | matplotlib

## Key results
- Churn classification: RandomForest, PR-AUC 0.254 +/- 0.010 (vs. 0.286 +/- 0.012 for LR baseline).
  At top-400 (7%) threshold, precision = 34%, recall = 20%.
- Watch-hour regression: Ridge regression on log-target, MAE = 3.39 hours, R2 = 0.78. Last month watch hours is the strongest single predictor (corr = 0.88).

## Design tradeoffs
- Chose Ridge over LR to handle collinear lag features
- Attempted 'log1p' transform on regression target
- Used top-K threshold rather than 0.5 to align model output with retention capacity

## Next steps
- Test a gradient-boosting model
