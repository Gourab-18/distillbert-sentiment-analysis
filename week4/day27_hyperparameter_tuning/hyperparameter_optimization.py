"""
Day 27: Assignment - Hyperparameter Tuning & Optimization
Grid Search, Random Search, Bayesian Optimization with Optuna
Goal: Improve model performance by 5%+
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, learning_curve, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
import time
import warnings
warnings.filterwarnings('ignore')

# Optuna
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")


# ==================== SEARCH METHODS COMPARISON ====================

class HyperparameterSearchMethods:
    """
    Compare different hyperparameter search methods
    """

    @staticmethod
    def explain_methods():
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║              HYPERPARAMETER OPTIMIZATION METHODS                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. GRID SEARCH                                                       ║
║     - Exhaustively searches all parameter combinations                ║
║     - Pros: Guaranteed to find best in search space                   ║
║     - Cons: Exponentially slow with more parameters                   ║
║     - Time: O(n^k) where n=values, k=parameters                       ║
║                                                                       ║
║  2. RANDOM SEARCH                                                     ║
║     - Randomly samples from parameter distributions                   ║
║     - Pros: More efficient, can find good solutions fast              ║
║     - Cons: May miss optimal, no guided exploration                   ║
║     - Time: O(n) where n=number of samples                            ║
║                                                                       ║
║  3. BAYESIAN OPTIMIZATION (Optuna)                                    ║
║     - Uses probabilistic model to guide search                        ║
║     - Pros: Efficient, learns from previous evaluations               ║
║     - Cons: Overhead for small search spaces                          ║
║     - Uses: TPE (Tree-structured Parzen Estimator)                    ║
║                                                                       ║
║  4. GENETIC ALGORITHMS                                                ║
║     - Evolution-inspired optimization                                 ║
║     - Good for complex, non-convex search spaces                      ║
║                                                                       ║
║  5. HYPERBAND                                                         ║
║     - Early stopping of bad configurations                            ║
║     - Efficient for expensive evaluations                             ║
╚══════════════════════════════════════════════════════════════════════╝
        """)


class GridSearchOptimizer:
    """
    Grid Search implementation
    """

    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.grid_search = None

    def fit(self, X, y):
        """Perform grid search"""
        start_time = time.time()

        self.grid_search = GridSearchCV(
            self.estimator,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        self.grid_search.fit(X, y)

        self.search_time = time.time() - start_time

        print(f"\nGrid Search completed in {self.search_time:.2f}s")
        print(f"Best parameters: {self.grid_search.best_params_}")
        print(f"Best score: {self.grid_search.best_score_:.4f}")

        return self.grid_search

    def get_results_df(self):
        """Get results as DataFrame"""
        if self.grid_search is None:
            return None

        results = pd.DataFrame(self.grid_search.cv_results_)
        return results.sort_values('rank_test_score')


class RandomSearchOptimizer:
    """
    Random Search implementation
    """

    def __init__(self, estimator, param_distributions, n_iter=100,
                 cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_search = None

    def fit(self, X, y):
        """Perform random search"""
        start_time = time.time()

        self.random_search = RandomizedSearchCV(
            self.estimator,
            self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )

        self.random_search.fit(X, y)

        self.search_time = time.time() - start_time

        print(f"\nRandom Search completed in {self.search_time:.2f}s")
        print(f"Best parameters: {self.random_search.best_params_}")
        print(f"Best score: {self.random_search.best_score_:.4f}")

        return self.random_search


class OptunaOptimizer:
    """
    Bayesian Optimization with Optuna
    """

    def __init__(self, model_type='random_forest', n_trials=100):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required")

        self.model_type = model_type
        self.n_trials = n_trials
        self.study = None
        self.best_model = None

    def objective_random_forest(self, trial, X, y, cv):
        """Objective function for Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    def objective_gradient_boosting(self, trial, X, y, cv):
        """Objective function for Gradient Boosting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }

        model = GradientBoostingClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    def objective_svm(self, trial, X, y, cv):
        """Objective function for SVM"""
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        }

        model = SVC(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    def objective_mlp(self, trial, X, y, cv):
        """Objective function for MLP"""
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256))

        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        }

        model = MLPClassifier(**params, max_iter=500, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    def optimize(self, X, y, cv=5):
        """Run optimization"""
        start_time = time.time()

        # Select objective function
        objectives = {
            'random_forest': self.objective_random_forest,
            'gradient_boosting': self.objective_gradient_boosting,
            'svm': self.objective_svm,
            'mlp': self.objective_mlp
        }

        objective = objectives.get(self.model_type)
        if objective is None:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        self.study.optimize(
            lambda trial: objective(trial, X, y, cv),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.search_time = time.time() - start_time

        print(f"\nOptuna optimization completed in {self.search_time:.2f}s")
        print(f"Best parameters: {self.study.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")

        return self.study

    def get_best_model(self, X, y):
        """Get the best model trained on full data"""
        if self.study is None:
            raise ValueError("Run optimize() first")

        best_params = self.study.best_params

        # Create model based on type
        if self.model_type == 'random_forest':
            self.best_model = RandomForestClassifier(**best_params, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.best_model = GradientBoostingClassifier(**best_params, random_state=42)
        elif self.model_type == 'svm':
            self.best_model = SVC(**best_params, random_state=42)
        elif self.model_type == 'mlp':
            n_layers = best_params.pop('n_layers')
            layers = tuple(best_params.pop(f'n_units_l{i}') for i in range(n_layers))
            self.best_model = MLPClassifier(hidden_layer_sizes=layers, **best_params,
                                            max_iter=500, random_state=42)

        self.best_model.fit(X, y)
        return self.best_model

    def plot_results(self, save_prefix='optuna'):
        """Plot optimization results"""
        if self.study is None:
            return

        # Optimization history
        fig = plot_optimization_history(self.study)
        fig.write_image(f'{save_prefix}_history.png')
        print(f"Saved: {save_prefix}_history.png")

        # Parameter importance
        try:
            fig = plot_param_importances(self.study)
            fig.write_image(f'{save_prefix}_importance.png')
            print(f"Saved: {save_prefix}_importance.png")
        except Exception as e:
            print(f"Could not plot parameter importance: {e}")

        # Parallel coordinate
        try:
            fig = plot_parallel_coordinate(self.study)
            fig.write_image(f'{save_prefix}_parallel.png')
            print(f"Saved: {save_prefix}_parallel.png")
        except Exception as e:
            print(f"Could not plot parallel coordinates: {e}")


# ==================== LEARNING CURVES ====================

def plot_learning_curves(estimator, X, y, title="Learning Curves", cv=5):
    """
    Plot learning curves to analyze model performance vs training size
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='orange')

    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')

    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.close()
    print("Saved: learning_curves.png")

    # Diagnosis
    final_train = train_mean[-1]
    final_test = test_mean[-1]
    gap = final_train - final_test

    print(f"\nLearning Curve Analysis:")
    print(f"  Final training score: {final_train:.4f}")
    print(f"  Final CV score: {final_test:.4f}")
    print(f"  Gap: {gap:.4f}")

    if gap > 0.1:
        print("  Diagnosis: HIGH VARIANCE (overfitting)")
        print("  Solutions: More data, regularization, simpler model")
    elif final_test < 0.7:
        print("  Diagnosis: HIGH BIAS (underfitting)")
        print("  Solutions: More features, complex model, less regularization")
    else:
        print("  Diagnosis: Good fit")

    return train_sizes, train_mean, test_mean


# ==================== MAIN OPTIMIZATION PIPELINE ====================

def run_optimization_comparison():
    """
    Run and compare all optimization methods
    """
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPARISON")
    print("=" * 60)

    # Explain methods
    HyperparameterSearchMethods.explain_methods()

    # Load data
    print("\n1. Loading Dataset")
    print("-" * 40)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")

    # Baseline model
    print("\n2. Baseline Model (Default Parameters)")
    print("-" * 40)
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
    baseline_cv = cross_val_score(baseline_model, X_train, y_train, cv=5).mean()

    print(f"Baseline Test Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline CV Accuracy: {baseline_cv:.4f}")

    # Results storage
    results = {
        'Method': ['Baseline'],
        'Best CV Score': [baseline_cv],
        'Test Accuracy': [baseline_accuracy],
        'Time (s)': [0],
        'Improvement': [0]
    }

    # === Grid Search ===
    print("\n3. Grid Search")
    print("-" * 40)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    grid_optimizer = GridSearchOptimizer(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5
    )
    grid_search = grid_optimizer.fit(X_train, y_train)

    grid_test_acc = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
    results['Method'].append('Grid Search')
    results['Best CV Score'].append(grid_search.best_score_)
    results['Test Accuracy'].append(grid_test_acc)
    results['Time (s)'].append(grid_optimizer.search_time)
    results['Improvement'].append((grid_test_acc - baseline_accuracy) * 100)

    # === Random Search ===
    print("\n4. Random Search")
    print("-" * 40)
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    random_optimizer = RandomSearchOptimizer(
        RandomForestClassifier(random_state=42),
        param_distributions,
        n_iter=50,
        cv=5
    )
    random_search = random_optimizer.fit(X_train, y_train)

    random_test_acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
    results['Method'].append('Random Search')
    results['Best CV Score'].append(random_search.best_score_)
    results['Test Accuracy'].append(random_test_acc)
    results['Time (s)'].append(random_optimizer.search_time)
    results['Improvement'].append((random_test_acc - baseline_accuracy) * 100)

    # === Optuna ===
    if OPTUNA_AVAILABLE:
        print("\n5. Bayesian Optimization (Optuna)")
        print("-" * 40)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optuna_optimizer = OptunaOptimizer(
            model_type='random_forest',
            n_trials=50
        )
        study = optuna_optimizer.optimize(X_train, y_train, cv=5)

        best_model = optuna_optimizer.get_best_model(X_train, y_train)
        optuna_test_acc = accuracy_score(y_test, best_model.predict(X_test))

        results['Method'].append('Optuna (Bayesian)')
        results['Best CV Score'].append(study.best_value)
        results['Test Accuracy'].append(optuna_test_acc)
        results['Time (s)'].append(optuna_optimizer.search_time)
        results['Improvement'].append((optuna_test_acc - baseline_accuracy) * 100)

        # Try different models with Optuna
        print("\n6. Optuna with Different Models")
        print("-" * 40)

        for model_type in ['gradient_boosting', 'svm']:
            print(f"\n  Optimizing {model_type}...")
            optimizer = OptunaOptimizer(model_type=model_type, n_trials=30)
            study = optimizer.optimize(X_train, y_train, cv=5)
            best_model = optimizer.get_best_model(X_train, y_train)
            test_acc = accuracy_score(y_test, best_model.predict(X_test))

            results['Method'].append(f'Optuna ({model_type})')
            results['Best CV Score'].append(study.best_value)
            results['Test Accuracy'].append(test_acc)
            results['Time (s)'].append(optimizer.search_time)
            results['Improvement'].append((test_acc - baseline_accuracy) * 100)

    # === Results Summary ===
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False)
    print(results_df.to_string(index=False))

    # Best result
    best_idx = results_df['Test Accuracy'].idxmax()
    best_method = results_df.loc[best_idx, 'Method']
    best_accuracy = results_df.loc[best_idx, 'Test Accuracy']
    improvement = (best_accuracy - baseline_accuracy) * 100

    print(f"\nBest Method: {best_method}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}%")

    if improvement >= 5:
        print("\n✓ SUCCESS: Achieved 5%+ improvement!")
    else:
        print(f"\n○ Improvement is {improvement:.2f}% (target: 5%+)")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))
    bars = plt.barh(results_df['Method'], results_df['Test Accuracy'], color=colors)
    plt.xlabel('Test Accuracy')
    plt.title('Model Comparison')
    plt.axvline(x=baseline_accuracy, color='red', linestyle='--', label='Baseline')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.barh(results_df['Method'], results_df['Improvement'], color=colors)
    plt.xlabel('Improvement (%)')
    plt.title('Improvement over Baseline')
    plt.axvline(x=5, color='green', linestyle='--', label='5% Target')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150)
    plt.close()
    print("\nSaved: optimization_comparison.png")

    # Learning curves for best model
    print("\n7. Learning Curves Analysis")
    print("-" * 40)
    best_params = grid_search.best_params_  # Use grid search best for plotting
    plot_learning_curves(
        RandomForestClassifier(**best_params, random_state=42),
        X_train, y_train,
        title="Learning Curves (Optimized Random Forest)"
    )

    return results_df


if __name__ == "__main__":
    results = run_optimization_comparison()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    This module demonstrated:
    1. Grid Search - exhaustive parameter search
    2. Random Search - random sampling from distributions
    3. Bayesian Optimization (Optuna) - intelligent guided search
    4. Learning curves analysis

    Key findings:
    - Bayesian optimization often finds better solutions faster
    - Random search is surprisingly effective for many problems
    - Learning curves help diagnose overfitting/underfitting

    Output files:
    - optimization_comparison.png: Method comparison chart
    - learning_curves.png: Training vs validation analysis
    - optuna_*.png: Optuna visualization plots (if Optuna available)
    """)
