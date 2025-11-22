"""
Day 28: Assignment - Ethics, Bias, and Explainability
Analyze bias in a model and create explainability report using SHAP and LIME
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")


# ==================== RESPONSIBLE AI PRINCIPLES ====================

class ResponsibleAIPrinciples:
    """
    Overview of Responsible AI principles
    """

    @staticmethod
    def display():
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESPONSIBLE AI PRINCIPLES                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. FAIRNESS                                                          ║
║     - Equal treatment across demographic groups                       ║
║     - Avoid discrimination based on protected attributes              ║
║     - Monitor and mitigate disparate impact                           ║
║                                                                       ║
║  2. TRANSPARENCY                                                      ║
║     - Explainable model decisions                                     ║
║     - Clear documentation of model limitations                        ║
║     - Open about data sources and potential biases                    ║
║                                                                       ║
║  3. ACCOUNTABILITY                                                    ║
║     - Clear ownership of AI systems                                   ║
║     - Regular auditing and monitoring                                 ║
║     - Mechanisms for addressing errors                                ║
║                                                                       ║
║  4. PRIVACY                                                           ║
║     - Protect sensitive information                                   ║
║     - Data minimization principles                                    ║
║     - Consent and user control                                        ║
║                                                                       ║
║  5. SAFETY & SECURITY                                                 ║
║     - Robust against adversarial attacks                              ║
║     - Fail-safe mechanisms                                            ║
║     - Regular security assessments                                    ║
║                                                                       ║
║  6. HUMAN OVERSIGHT                                                   ║
║     - Human-in-the-loop for critical decisions                        ║
║     - Clear escalation paths                                          ║
║     - Ability to override AI decisions                                ║
╚══════════════════════════════════════════════════════════════════════╝
        """)


# ==================== BIAS DETECTION ====================

class BiasDetector:
    """
    Detect and measure bias in ML models
    """

    def __init__(self, model, X, y, sensitive_features, feature_names=None):
        """
        Initialize bias detector

        Args:
            model: Trained model
            X: Feature matrix
            y: Labels
            sensitive_features: Dict mapping feature index to name
            feature_names: List of feature names
        """
        self.model = model
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.predictions = model.predict(X)

    def calculate_group_metrics(self, feature_idx, feature_name):
        """
        Calculate metrics for each group of a sensitive feature
        """
        unique_values = np.unique(self.X[:, feature_idx])

        results = []
        for value in unique_values:
            mask = self.X[:, feature_idx] == value
            group_size = np.sum(mask)

            if group_size == 0:
                continue

            y_true = self.y[mask]
            y_pred = self.predictions[mask]

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Positive prediction rate
            positive_rate = np.mean(y_pred)

            results.append({
                'Feature': feature_name,
                'Group': value,
                'Size': group_size,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Positive Rate': positive_rate
            })

        return pd.DataFrame(results)

    def detect_disparate_impact(self, feature_idx, threshold=0.8):
        """
        Calculate disparate impact ratio

        Disparate Impact = (Positive Rate for unprivileged group) /
                          (Positive Rate for privileged group)

        Value < threshold (typically 0.8) indicates potential discrimination
        """
        unique_values = np.unique(self.X[:, feature_idx])

        if len(unique_values) != 2:
            print(f"Warning: Disparate impact typically calculated for binary features")

        positive_rates = []
        for value in unique_values:
            mask = self.X[:, feature_idx] == value
            rate = np.mean(self.predictions[mask])
            positive_rates.append((value, rate))

        # Sort by positive rate
        positive_rates.sort(key=lambda x: x[1])

        unprivileged_rate = positive_rates[0][1]
        privileged_rate = positive_rates[-1][1]

        if privileged_rate == 0:
            disparate_impact = float('inf')
        else:
            disparate_impact = unprivileged_rate / privileged_rate

        return {
            'unprivileged_group': positive_rates[0][0],
            'unprivileged_rate': unprivileged_rate,
            'privileged_group': positive_rates[-1][0],
            'privileged_rate': privileged_rate,
            'disparate_impact': disparate_impact,
            'passes_threshold': disparate_impact >= threshold
        }

    def calculate_equalized_odds_difference(self, feature_idx):
        """
        Calculate equalized odds difference

        Measures difference in true positive rates and false positive rates
        across groups
        """
        unique_values = np.unique(self.X[:, feature_idx])

        tpr_list = []
        fpr_list = []

        for value in unique_values:
            mask = self.X[:, feature_idx] == value
            y_true = self.y[mask]
            y_pred = self.predictions[mask]

            # True positive rate
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

            # False positive rate
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_diff = max(tpr_list) - min(tpr_list)
        fpr_diff = max(fpr_list) - min(fpr_list)

        return {
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'equalized_odds_diff': max(tpr_diff, fpr_diff)
        }

    def generate_bias_report(self):
        """
        Generate comprehensive bias report
        """
        print("\n" + "=" * 60)
        print("BIAS ANALYSIS REPORT")
        print("=" * 60)

        all_metrics = []

        for feature_idx, feature_name in self.sensitive_features.items():
            print(f"\n--- {feature_name} (Feature {feature_idx}) ---")

            # Group metrics
            group_metrics = self.calculate_group_metrics(feature_idx, feature_name)
            print("\nGroup Performance Metrics:")
            print(group_metrics.to_string(index=False))
            all_metrics.append(group_metrics)

            # Disparate impact
            di = self.detect_disparate_impact(feature_idx)
            print(f"\nDisparate Impact Analysis:")
            print(f"  Privileged group: {di['privileged_group']} (rate: {di['privileged_rate']:.4f})")
            print(f"  Unprivileged group: {di['unprivileged_group']} (rate: {di['unprivileged_rate']:.4f})")
            print(f"  Disparate Impact Ratio: {di['disparate_impact']:.4f}")
            print(f"  Passes 80% Rule: {'✓ Yes' if di['passes_threshold'] else '✗ No (potential bias)'}")

            # Equalized odds
            eo = self.calculate_equalized_odds_difference(feature_idx)
            print(f"\nEqualized Odds:")
            print(f"  TPR Difference: {eo['tpr_difference']:.4f}")
            print(f"  FPR Difference: {eo['fpr_difference']:.4f}")

        return pd.concat(all_metrics, ignore_index=True) if all_metrics else None


# ==================== SHAP EXPLAINER ====================

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model explainability
    """

    def __init__(self, model, X_train, feature_names=None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required")

        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]

        # Create explainer
        if hasattr(model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_train[:100])

    def explain_instance(self, instance, plot=True):
        """
        Explain a single prediction
        """
        instance = np.array(instance).reshape(1, -1)
        shap_values = self.explainer.shap_values(instance)

        prediction = self.model.predict(instance)[0]

        print(f"\nPrediction: {prediction}")
        print("\nFeature contributions:")

        # Handle binary classification
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Class 1 SHAP values
        else:
            sv = shap_values[0]

        contributions = list(zip(self.feature_names, sv))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, value in contributions[:10]:
            direction = "↑" if value > 0 else "↓"
            print(f"  {name}: {value:+.4f} {direction}")

        if plot:
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.force_plot(
                    self.explainer.expected_value[1],
                    shap_values[1][0],
                    instance[0],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
            plt.tight_layout()
            plt.savefig('shap_instance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: shap_instance.png")

        return shap_values

    def explain_global(self, X_sample=None, max_display=20):
        """
        Generate global feature importance explanations
        """
        if X_sample is None:
            X_sample = self.X_train[:500]  # Sample for efficiency

        shap_values = self.explainer.shap_values(X_sample)

        # Summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values[1],
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: shap_summary.png")

        # Bar plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values[1],
                X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: shap_importance.png")

        return shap_values


# ==================== LIME EXPLAINER ====================

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations)
    """

    def __init__(self, model, X_train, feature_names=None, class_names=None):
        if not LIME_AVAILABLE:
            raise ImportError("LIME required")

        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['Class 0', 'Class 1']

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

    def explain_instance(self, instance, num_features=10):
        """
        Explain a single prediction using LIME
        """
        instance = np.array(instance).reshape(-1)

        # Get LIME explanation
        if hasattr(self.model, 'predict_proba'):
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
        else:
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )

        prediction = self.model.predict(instance.reshape(1, -1))[0]

        print(f"\nLIME Explanation for prediction: {prediction}")
        print("\nFeature contributions:")

        for feature, weight in exp.as_list():
            direction = "↑" if weight > 0 else "↓"
            print(f"  {feature}: {weight:+.4f} {direction}")

        # Save plot
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig('lime_explanation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: lime_explanation.png")

        return exp


# ==================== EXPLAINABILITY REPORT GENERATOR ====================

class ExplainabilityReportGenerator:
    """
    Generate comprehensive explainability report
    """

    def __init__(self, model, X_train, X_test, y_test, feature_names=None,
                 sensitive_features=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.sensitive_features = sensitive_features or {}

    def generate_report(self):
        """
        Generate full explainability report
        """
        print("=" * 60)
        print("MODEL EXPLAINABILITY REPORT")
        print("=" * 60)

        # 1. Model Performance
        print("\n1. MODEL PERFORMANCE")
        print("-" * 40)
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

        # 2. Feature Importance (if available)
        print("\n2. FEATURE IMPORTANCE")
        print("-" * 40)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print("\nTop 10 Features:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"  {i + 1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

            # Plot
            plt.figure(figsize=(10, 6))
            top_n = min(15, len(indices))
            plt.barh(range(top_n),
                     importances[indices[:top_n]][::-1],
                     color=plt.cm.viridis(np.linspace(0, 0.8, top_n)))
            plt.yticks(range(top_n),
                       [self.feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Feature Importance')
            plt.title('Model Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150)
            plt.close()
            print("\nSaved: feature_importance.png")

        # 3. SHAP Analysis
        if SHAP_AVAILABLE:
            print("\n3. SHAP ANALYSIS")
            print("-" * 40)
            shap_explainer = SHAPExplainer(self.model, self.X_train, self.feature_names)
            shap_explainer.explain_global(self.X_test[:200])

            # Explain a sample instance
            print("\nSample Instance Explanation:")
            shap_explainer.explain_instance(self.X_test[0])

        # 4. LIME Analysis
        if LIME_AVAILABLE:
            print("\n4. LIME ANALYSIS")
            print("-" * 40)
            lime_explainer = LIMEExplainer(self.model, self.X_train, self.feature_names)
            lime_explainer.explain_instance(self.X_test[0])

        # 5. Bias Analysis
        if self.sensitive_features:
            print("\n5. BIAS ANALYSIS")
            print("-" * 40)
            bias_detector = BiasDetector(
                self.model,
                self.X_test,
                self.y_test,
                self.sensitive_features,
                self.feature_names
            )
            bias_detector.generate_bias_report()

        print("\n" + "=" * 60)
        print("REPORT COMPLETE")
        print("=" * 60)


# ==================== SYNTHETIC DATASET WITH BIAS ====================

def create_biased_dataset(n_samples=1000, bias_strength=0.3):
    """
    Create a synthetic dataset with intentional bias for demonstration
    """
    np.random.seed(42)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    # Add demographic features
    # Gender (0 = male, 1 = female)
    gender = np.random.binomial(1, 0.5, n_samples)

    # Age group (0 = young, 1 = old)
    age_group = np.random.binomial(1, 0.4, n_samples)

    # Inject bias: females get positive outcomes less often
    bias_mask = (gender == 1) & (np.random.random(n_samples) < bias_strength)
    y[bias_mask] = 0

    # Add demographic features to X
    X = np.column_stack([X, gender, age_group])

    feature_names = [f'feature_{i}' for i in range(10)] + ['gender', 'age_group']
    sensitive_features = {10: 'gender', 11: 'age_group'}

    return X, y, feature_names, sensitive_features


# ==================== MAIN ====================

def run_explainability_analysis():
    """
    Run complete explainability analysis
    """

    # Display principles
    ResponsibleAIPrinciples.display()

    print("=" * 60)
    print("BIAS AND EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Create biased dataset
    print("\n1. Creating biased dataset for demonstration...")
    X, y, feature_names, sensitive_features = create_biased_dataset(n_samples=2000)

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Sensitive features: {sensitive_features}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (except demographic)
    scaler = StandardScaler()
    X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
    X_test[:, :10] = scaler.transform(X_test[:, :10])

    # Train model
    print("\n2. Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.4f}")

    # Generate explainability report
    print("\n3. Generating explainability report...")
    report_generator = ExplainabilityReportGenerator(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        sensitive_features=sensitive_features
    )
    report_generator.generate_report()

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR BIAS MITIGATION")
    print("=" * 60)
    print("""
    1. PRE-PROCESSING TECHNIQUES:
       - Re-sampling to balance groups
       - Re-weighting samples
       - Removing sensitive features

    2. IN-PROCESSING TECHNIQUES:
       - Adversarial debiasing
       - Fairness constraints during training
       - Regularization for fairness

    3. POST-PROCESSING TECHNIQUES:
       - Threshold adjustment per group
       - Calibration across groups
       - Reject option classification

    4. MONITORING & AUDITING:
       - Regular bias audits
       - Continuous monitoring in production
       - Feedback loops for corrections
    """)

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = run_explainability_analysis()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    This module demonstrated:
    1. Responsible AI principles
    2. Bias detection and fairness metrics
       - Disparate impact
       - Equalized odds
    3. Model explainability with SHAP
       - Global feature importance
       - Local explanations
    4. Model explainability with LIME
       - Instance-level explanations
    5. Comprehensive explainability report

    Output files:
    - feature_importance.png
    - shap_summary.png
    - shap_importance.png
    - shap_instance.png
    - lime_explanation.png
    """)
