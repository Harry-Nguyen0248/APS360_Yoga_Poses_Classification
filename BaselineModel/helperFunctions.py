import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import accuracy_score

def plot_hyperparameter_tuning_results(grid_search):
    # Convert GridSearchCV results to a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)

    # Plot the validation scores for different hyperparameters
    plt.figure(figsize=(12, 6))
    for param in grid_search.param_grid['n_estimators']:
        subset = results[results['param_n_estimators'] == param]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'], label=f'n_estimators={param}')
    
    plt.xlabel('max_depth')
    plt.ylabel('Mean Validation Accuracy')
    plt.title('Hyperparameter Tuning Results')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()




def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Define a function to calculate error
def calculate_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


