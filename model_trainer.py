# model_trainer.py
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

ETHICAL_WARNING = (
    "⚠️ ETHICAL ALERT: This model identifies financial strain. "
    "NEVER use to deny services without human review and alternative support options."
)

REQUIRED_LOAN_DENIAL_WARNING = "⚠️ NEVER use for automatic loan denial"


class ModelTrainer:
    """Train a churn model with fairness constraints and ethical safeguards."""

    def __init__(self, X_train, y_train, region_col="region"):
        self.X_train = X_train.copy()
        self.y_train = pd.Series(y_train).copy()
        self.region_col = region_col
        self.pipeline = None
        self.bias_audit_results = {}
        self.shap_results = []
        self.overall_recall = None

    def _make_preprocessor(self):
        """Build a robust numeric/categorical preprocessing transformer."""
        numeric_cols = self.X_train.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_cols = [
            col for col in self.X_train.columns
            if col not in numeric_cols
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("categorical", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )

    def build_fair_pipeline(self):
        """Create a calibrated pipeline with fairness-aware class weighting."""
        preprocessor = self._make_preprocessor()

        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )

        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=classifier,
                method="sigmoid",
                cv=3,
            )
        except TypeError:
            calibrated_model = CalibratedClassifierCV(
                base_estimator=classifier,
                method="sigmoid",
                cv=3,
            )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", calibrated_model),
            ]
        )

        return self.pipeline

    def _normalise_target(self):
        """Convert common binary target formats to 0/1."""
        y = self.y_train.copy()

        if y.isna().any():
            raise ValueError(
                "Target contains missing values. Remove rows with missing "
                "target labels before training."
            )

        if pd.api.types.is_numeric_dtype(y):
            unique = sorted(pd.Series(y).dropna().unique().tolist())

            if set(unique).issubset({0, 1}):
                return y.astype(int)

            if len(unique) == 2:
                return y.map({unique[0]: 0, unique[1]: 1}).astype(int)

        mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "churned": 1,
            "stay": 0,
            "stayed": 0,
            "active": 0,
            "inactive": 1,
        }

        normalised = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

        if normalised.notna().all():
            return normalised.astype(int)

        unique = pd.Series(y).dropna().unique().tolist()

        if len(unique) == 2:
            return y.map({unique[0]: 0, unique[1]: 1}).astype(int)

        raise ValueError(
            "Target must be binary. Could not convert y_train to 0/1."
        )

    def audit_bias(self):
        """Audit recall disparity across available regional subgroups."""
        if self.pipeline is None:
            raise RuntimeError("Build and fit the pipeline before auditing bias.")

        y = self._normalise_target()
        X = self.X_train.copy()

        if self.region_col not in X.columns:
            self.bias_audit_results = {
                "status": "region column not available",
                "max_disparity": 0.0,
                "fair": True,
            }
            self.overall_recall = float("nan")
            return self.bias_audit_results

        region_values = X[self.region_col].fillna("Unknown").astype(str)

        if y.nunique() < 2:
            self.bias_audit_results = {
                "status": "insufficient target classes for audit",
                "max_disparity": 0.0,
                "fair": True,
            }
            return self.bias_audit_results

        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())

        if min_class_count >= 3:
            cv_folds = min(3, min_class_count)

            predictions = cross_val_predict(
                self.pipeline,
                X,
                y,
                cv=cv_folds,
                method="predict",
                n_jobs=1,
            )

        else:
            if len(X) >= 6:
                X_audit, X_eval, y_audit, y_eval = train_test_split(
                    X,
                    y,
                    test_size=0.30,
                    random_state=42,
                    stratify=y,
                )

                audit_model = self.build_fair_pipeline()
                audit_model.fit(X_audit, y_audit)

                predictions_eval = audit_model.predict(X_eval)

                recall_values = {}
                eval_regions = X_eval[self.region_col].fillna(
                    "Unknown"
                ).astype(str)

                for region in sorted(eval_regions.unique()):
                    mask = eval_regions == region

                    if y_eval[mask].sum() > 0:
                        recall_values[region] = float(
                            recall_score(
                                y_eval[mask],
                                predictions_eval[mask],
                                zero_division=0,
                            )
                        )

                self.pipeline.fit(X, y)

                overall_predictions = self.pipeline.predict(X)

                self._store_bias_results(
                    recall_values,
                    y,
                    overall_predictions,
                )

                return self.bias_audit_results

            else:
                predictions = self.pipeline.predict(X)

        recall_values = {}

        for region in sorted(region_values.unique()):
            mask = region_values == region
            y_region = y[mask]
            pred_region = predictions[mask]

            if int(y_region.sum()) > 0:
                recall_values[region] = float(
                    recall_score(
                        y_region,
                        pred_region,
                        zero_division=0,
                    )
                )

        self._store_bias_results(recall_values, y, predictions)

        return self.bias_audit_results

    def _store_bias_results(self, recall_values, y, predictions):
        """Store audit results and determine whether disparity is acceptable."""
        if recall_values:
            max_recall = max(recall_values.values())
            min_recall = min(recall_values.values())
            disparity = max_recall - min_recall
        else:
            disparity = 0.0

        self.overall_recall = float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        )

        results = {
            "region_recalls": recall_values,
            "max_disparity": float(disparity),
            "fair": bool(disparity <= 0.15),
            "threshold": 0.15,
            "overall_recall": self.overall_recall,
        }

        for region, recall in recall_values.items():
            key = (
                f"{region.lower().replace(' ', '_').replace('-', '_')}_recall"
            )
            results[key] = recall

        for wanted in ["township", "urban"]:
            for region, recall in recall_values.items():
                if wanted in region.lower():
                    results[f"{wanted}_recall"] = recall
                    break

        self.bias_audit_results = results

    def calibrate_probabilities(self):
        """Ensure calibrated probabilities are available from the pipeline."""
        if self.pipeline is None:
            self.build_fair_pipeline()

        classifier = self.pipeline.named_steps.get("classifier")

        if isinstance(classifier, CalibratedClassifierCV):
            return self.pipeline

        preprocessor = self.pipeline.named_steps["preprocessor"]

        base_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )

        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=3,
            )
        except TypeError:
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model,
                method="sigmoid",
                cv=3,
            )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", calibrated_model),
            ]
        )

        return self.pipeline

    def generate_shap_analysis(self):
        """Generate top-feature importance and business interpretations."""
        self.shap_results = []

        try:
            import shap
        except ImportError:
            return self.shap_results

        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            classifier = self.pipeline.named_steps["classifier"]

            feature_names = preprocessor.get_feature_names_out()

            estimators = getattr(
                classifier,
                "calibrated_classifiers_",
                [],
            )

            if not estimators:
                return self.shap_results

            estimator = estimators[0]

            random_forest = getattr(
                estimator,
                "estimator",
                None,
            )

            if random_forest is None:
                random_forest = getattr(
                    estimator,
                    "base_estimator",
                    None,
                )

            if random_forest is None:
                return self.shap_results

            X_transformed = preprocessor.transform(self.X_train)

            sample_size = min(
                len(X_transformed),
                500,
            )

            X_sample = X_transformed[:sample_size]

            explainer = shap.TreeExplainer(random_forest)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                values = np.asarray(shap_values[-1])
            else:
                values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[:, :, -1]

            mean_abs = np.mean(
                np.abs(values),
                axis=0,
            )

            top_indices = np.argsort(mean_abs)[::-1][:3]

            for idx in top_indices:
                feature = str(feature_names[idx])
                importance = float(mean_abs[idx])

                self.shap_results.append(
                    {
                        "feature": feature,
                        "mean_abs_shap": importance,
                        "business_interpretation": (
                            self._interpret_feature(feature)
                        ),
                    }
                )

        except Exception:
            self.shap_results = []

        return self.shap_results

    @staticmethod
    def _interpret_feature(feature):
        """Translate common financial features into business language."""
        text = feature.lower()

        if "debt" in text and "income" in text:
            return (
                "Higher debt relative to income can indicate financial strain "
                "and may increase churn risk."
            )

        if "income" in text or "salary" in text:
            return (
                "Income-related changes can signal changes in a customer's "
                "ability to maintain financial commitments."
            )

        if "loan" in text or "balance" in text or "amount" in text:
            return (
                "Loan or balance exposure may indicate the level of financial "
                "commitment associated with the customer."
            )

        if "payment" in text or "installment" in text:
            return (
                "Payment behaviour can provide an early signal of financial "
                "pressure and potential churn."
            )

        if "age" in text:
            return (
                "Age should be interpreted cautiously because demographic "
                "patterns must not be used to justify discriminatory decisions."
            )

        if "region" in text or "township" in text or "urban" in text:
            return (
                "Geographic information can reveal service-access patterns, "
                "but must not be used as a basis for automatic exclusion."
            )

        if "tenure" in text or "month" in text or "history" in text:
            return (
                "Customer history can indicate relationship stability, but "
                "short histories should be treated as higher uncertainty."
            )

        return (
            "This feature is associated with predicted churn risk; its "
            "relationship should be reviewed for fairness and business context."
        )

    def _get_recall(self):
        """Return the audited overall recall."""
        if self.overall_recall is not None:
            return float(self.overall_recall)

        return float(
            self.bias_audit_results.get(
                "overall_recall",
                0.0,
            )
        )

    def generate_model_card(self, output_path="reports/model_card.md"):
        """Create comprehensive ethical model documentation."""
        os.makedirs(
            os.path.dirname(output_path) or ".",
            exist_ok=True,
        )

        region_recalls = self.bias_audit_results.get(
            "region_recalls",
            {},
        )

        max_disparity = self.bias_audit_results.get(
            "max_disparity",
            0.0,
        )

        if region_recalls:
            region_lines = "\n".join(
                f"- {region}: {recall:.1%}"
                for region, recall in sorted(
                    region_recalls.items()
                )
            )
        else:
            region_lines = "- Regional recall unavailable"

        shap_lines = []

        for item in self.shap_results:
            shap_lines.append(
                f"- **{item['feature']}** — mean absolute SHAP value "
                f"{item['mean_abs_shap']:.4f}. "
                f"{item['business_interpretation']}"
            )

        if not shap_lines:
            shap_lines.append(
                "- SHAP values were not generated because the optional "
                "`shap` package was unavailable or the fitted model could "
                "not be explained safely."
            )

        card_content = f"""# Churn Prediction Model Card

## Model Purpose

This model predicts customer churn risk to support early, human-led customer
support and financial assistance. It is a decision-support system, not an
automated eligibility, lending, credit, or loan-denial system.

## Performance

- Overall Recall: {self._get_recall():.1%}
- Maximum regional recall disparity: {max_disparity:.1%}
- Fairness threshold: 15.0%
- Bias audit status: {"PASS" if max_disparity <= 0.15 else "REVIEW REQUIRED"}

### Regional Recall

{region_lines}

## Calibration

The classifier uses sigmoid calibration (Platt scaling) through
`CalibratedClassifierCV` so that predicted probabilities can be interpreted
as calibrated risk estimates rather than raw model scores.

## Interpretability: Top 3 Features

{chr(10).join(shap_lines)}

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

{REQUIRED_LOAN_DENIAL_WARNING}

**{ETHICAL_WARNING}**

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
"""

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(card_content)

        return output_path

    def __str__(self):
        """Return the fairness summary string."""
        township = self.bias_audit_results.get(
            "township_recall"
        )

        urban = self.bias_audit_results.get(
            "urban_recall"
        )

        recall = self._get_recall()

        disparity = self.bias_audit_results.get(
            "max_disparity",
            0.0,
        )

        township_text = (
            f"{float(township):.0%}"
            if township is not None
            else "N/A"
        )

        urban_text = (
            f"{float(urban):.0%}"
            if urban is not None
            else "N/A"
        )

        return (
            f"Recall: {recall:.0%} "
            f"(Township: {township_text} | "
            f"Urban: {urban_text}) | "
            f"Max disparity: {float(disparity):.0%}"
        )

    def train(self):
        """Train the model with ethical validation and documentation."""
        self.build_fair_pipeline()
        self.calibrate_probabilities()

        target = self._normalise_target()

        self.pipeline.fit(
            self.X_train,
            target,
        )

        self.audit_bias()
        self.generate_shap_analysis()

        if self.bias_audit_results.get(
            "max_disparity",
            0.0,
        ) > 0.15:
            raise ValueError(
                "Ethical fairness constraint failed: regional recall "
                "disparity exceeds 15%."
            )

        return self.pipeline


def load_training_data(
    path="data/processed/engineered_features.csv",
):
    """Load the Milestone 2 engineered feature dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Engineered feature dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The engineered feature dataset is empty."
        )

    return df


def identify_target(df):
    """Find the binary churn target used by the engineered dataset."""
    candidates = [
        "churn",
        "churned",
        "is_churned",
        "customer_churn",
        "target",
        "label",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify the churn target column. Expected one of: "
        + ", ".join(candidates)
    )


def main():
    """Run the complete Milestone 3 training workflow."""
    input_path = "data/processed/engineered_features.csv"
    model_path = "models/churn_pipeline.pkl"
    model_card_path = "reports/model_card.md"

    os.makedirs(
        "models",
        exist_ok=True,
    )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    df = load_training_data(input_path)

    target_col = identify_target(df)

    missing_target_count = int(
        df[target_col].isna().sum()
    )

    if missing_target_count > 0:
        print(
            f"Removing {missing_target_count} rows with missing "
            f"'{target_col}' target values."
        )

        df = df.dropna(
            subset=[target_col]
        ).copy()

    if df.empty:
        raise ValueError(
            "No labelled records remain after removing missing target values."
        )

    target_values = pd.Series(
        df[target_col]
    ).dropna().unique()

    if len(target_values) != 2:
        raise ValueError(
            f"Target '{target_col}' must contain exactly two classes "
            f"after removing missing values. Found: {target_values.tolist()}"
        )

    print(
        f"Training rows after target cleaning: {len(df)}"
    )

    print(
        f"Target distribution:\n{df[target_col].value_counts()}"
    )

    y = df[target_col]

    X = df.drop(
        columns=[target_col]
    )

    region_col = (
        "region"
        if "region" in X.columns
        else None
    )

    trainer = ModelTrainer(
        X_train=X,
        y_train=y,
        region_col=region_col or "region",
    )

    trainer.train()

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            trainer.pipeline,
            file,
        )

    trainer.generate_model_card(
        model_card_path
    )

    print(
        "Model training completed successfully."
    )

    print(
        f"Pipeline saved to: {model_path}"
    )

    print(
        f"Model card saved to: {model_card_path}"
    )

    print(trainer)

    print(ETHICAL_WARNING)


if __name__ == "__main__":
    main()