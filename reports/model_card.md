# Churn Prediction Model Card

## Model Purpose

This model predicts customer churn risk to support early, human-led customer
support and financial assistance. It is a decision-support system, not an
automated eligibility, lending, credit, or loan-denial system.

## Performance

- Overall Recall: 0.5%
- Maximum regional recall disparity: 3.4%
- Fairness threshold: 15.0%
- Bias audit status: PASS

### Regional Recall

- Cape Town: 0.0%
- Durban: 0.3%
- Gqeberha: 3.4%
- Johannesburg: 0.2%
- Port Elizabeth: 0.0%
- Pretoria: 0.0%

## Calibration

The classifier uses sigmoid calibration (Platt scaling) through
`CalibratedClassifierCV` so that predicted probabilities can be interpreted
as calibrated risk estimates rather than raw model scores.

## Interpretability: Top 3 Features

- SHAP values were not generated because the optional `shap` package was unavailable or the fitted model could not be explained safely.

## Critical Limitations

- The model can produce false positives and false negatives.
- Customers with fewer than three months of history may have insufficient
  information for reliable predictions.
- Historical data can contain structural inequalities and measurement bias.
- Regional recall parity does not prove that the model is completely fair.
- Correlation between a feature and churn risk does not establish causation.
- SHAP explanations describe model behaviour and do not establish that a
  feature causes churn.
- Predictions must not replace human judgement or customer engagement.

## Failure Modes

- Missing, stale, incorrectly recorded, or out-of-distribution customer data
  can reduce prediction quality.
- A new region or category can differ from the training distribution.
- Changes in economic conditions can make historical relationships less
  reliable.
- Small subgroup sample sizes can make recall estimates unstable.
- Probability calibration can degrade when the production population changes.
- A subgroup can meet the 15% recall-disparity threshold while still
  experiencing unacceptable absolute performance.

## Ethical Constraints

⚠️ NEVER use for automatic loan denial

**⚠️ ETHICAL ALERT: This model identifies financial strain. NEVER use to deny services without human review and alternative support options.**

High-risk predictions require human review and appropriate alternative support
options. Model outputs must not be used to automatically deny loans, services,
benefits, or financial assistance.

The model must not be deployed where its predictions could perpetuate
exclusion or systematically disadvantage historically underserved customers.

## Human Review Safeguard

Human review is mandatory for high-risk predictions, including customers
showing indicators of financial strain such as debt-to-income ratios above
0.6. Reviewers must consider the customer's circumstances and available
support options rather than treating the prediction as a final decision.

## Monitoring Plan

- Conduct monthly regional fairness audits.
- Alert when township recall drops more than 10% from the established baseline.
- Investigate any regional recall disparity above 15%.
- Monitor calibration quality and probability reliability.
- Track false-positive and false-negative rates by subgroup.
- Monitor missingness, feature drift, and changes in the customer population.
- Retrain quarterly or sooner when material economic or population changes
  occur.
- Record human-review outcomes and investigate repeated model failures.

## Deployment Decision

This model is suitable only for supervised, human-reviewed churn-risk
support. It is not approved for automatic loan denial or automatic denial of
services.

## Accountability

The model owner is responsible for maintaining fairness audits, reviewing
failure modes, documenting material changes, and suspending use when ethical
or performance safeguards are not met.
