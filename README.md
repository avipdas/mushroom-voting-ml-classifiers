# Machine Learning Classifier Implementation: Naïve Bayes and Decision Trees

## Project Overview
This repository contains a custom implementation of Naïve Bayes and Decision Tree classifiers for categorical data, applied to two classic datasets from the UCI Machine Learning Repository: the Mushroom dataset and the Congressional Voting Records dataset. The implementations demonstrate fundamental machine learning concepts without relying on pre-built models from libraries like scikit-learn.

## Technical Skills Demonstrated
- **Algorithm Implementation**: Hand-crafted machine learning algorithms from theoretical foundations
- **Probabilistic Modeling**: Correct implementation of conditional probability calculations with smoothing
- **Information Theory**: Entropy-based calculations for decision tree splitting
- **Data Processing**: Handling of categorical features and missing values
- **Model Evaluation**: Implementation of performance metrics and confusion matrix reporting


## Implementation Details

### Naïve Bayes Classifier
The implementation features:
- Laplace (additive) smoothing with configurable parameter
- Log-probability calculations to prevent numerical underflow
- Proper handling of missing values as their own category
- Efficient calculation of conditional probabilities using pandas

### Decision Tree Classifier
The implementation includes:
- Information gain criterion for optimal feature selection
- Configurable hyperparameters (maximum depth, minimum samples per leaf)
- Recursively built tree structure with proper handling of categorical features
- Special handling for missing values

### Data Utilities
- Train/dev/test splitting with configurable proportions
- Performance metrics calculation
- Confusion matrix generation
- Data loading and preprocessing

## Results Summary
Both classifiers achieve strong performance on the test datasets. The implementation demonstrates the following key findings:
- Naïve Bayes performs particularly well on high-dimensional categorical data
- Decision trees provide interpretable models with competitive accuracy
- Feature selection using information gain effectively identifies discriminative attributes
- Proper handling of missing values significantly impacts performance

## How to Run

### Prerequisites
- Python 3.7+
- NumPy
- Pandas

### Basic Usage
```python
# Import the classifiers
from classifiers import NaiveBayes, DecisionTree, split_data, accuracy, confusion, load_data

# Load data
X, y = load_data('path/to/dataset.csv')

# Split data
X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

# Train and evaluate Naive Bayes
nb = NaiveBayes(smoothing=1.0).fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"Naive Bayes Test Accuracy: {accuracy(y_test, y_pred_nb):.4f}")
print(confusion(y_test, y_pred_nb))

# Train and evaluate Decision Tree
dt = DecisionTree(max_depth=5, min_samples=2).fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f"Decision Tree Test Accuracy: {accuracy(y_test, y_pred_dt):.4f}")
print(confusion(y_test, y_pred_dt))
```

### Hyperparameter Tuning
The classifiers support various hyperparameters that can be tuned:

**Naïve Bayes**:
- `smoothing`: Controls the amount of Laplace smoothing (default: 1.0)

**Decision Tree**:
- `max_depth`: Maximum depth of the tree (default: None, unlimited)
- `min_samples`: Minimum number of samples required to split a node (default: 2)

## Experiment with Different Settings
To experiment with different model configurations:

```python
# Try different smoothing values for Naive Bayes
for smoothing in [0.1, 0.5, 1.0, 2.0, 5.0]:
    nb = NaiveBayes(smoothing=smoothing).fit(X_train, y_train)
    print(f"Smoothing={smoothing}, Dev Accuracy: {accuracy(y_dev, nb.predict(X_dev)):.4f}")

# Try different depths for Decision Tree
for depth in [3, 5, 7, 10, None]:
    dt = DecisionTree(max_depth=depth).fit(X_train, y_train)
    print(f"Max Depth={depth}, Dev Accuracy: {accuracy(y_dev, dt.predict(X_dev)):.4f}")
```

## Core Implementation Features

### Naïve Bayes
- **Class Prior Calculation**: Accurately computes class probabilities from training data
- **Feature Probability Estimation**: Calculates conditional probabilities with additive smoothing
- **Log-Likelihood Calculation**: Uses logarithms to prevent numerical underflow in probability products
- **Missing Value Handling**: Treats nulls as separate category with proper probability assignment

### Decision Tree
- **Recursive Tree Building**: Constructs a tree that recursively partitions data
- **Information Gain Calculation**: Selects features that maximize information gain at each split
- **Entropy Computation**: Measures uncertainty in class distribution before and after splits
- **Early Stopping Criteria**: Implements multiple stopping conditions to prevent overfitting
- **Prediction via Tree Traversal**: Efficiently traverses the tree structure for new predictions

