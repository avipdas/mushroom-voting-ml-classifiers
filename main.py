import numpy as np
import pandas as pd
import random

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(set(y))
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / len(X)
            self.feature_probs[c] = {
                col: X_c[col].value_counts(dropna=False).add(self.smoothing).div(len(X_c) + self.smoothing * X[col].nunique())
                for col in X.columns
            }
        return self

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            scores = {}
            for c in self.classes:
                score = np.log(self.class_priors[c])
                for col in X.columns:
                    val = row[col]
                    probs = self.feature_probs[c][col]
                    prob = probs.get(val, self.smoothing / (sum(probs) + self.smoothing * len(probs)))
                    score += np.log(prob)
                scores[c] = score
            preds.append(max(scores, key=scores.get))
        return preds

class DecisionTree:
    class Node:
        def __init__(self):
            self.feature = None
            self.children = {}
            self.prediction = None

    def __init__(self, max_depth=None, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = self.Node()
        node.prediction = y.mode()[0]
        if len(set(y)) == 1 or depth == self.max_depth or len(y) < self.min_samples:
            return node
        gains = {col: self._gain(X[col], y) for col in X.columns}
        best = max(gains, key=gains.get)
        if gains[best] == 0:
            return node
        node.feature = best
        for val in X[best].dropna().unique():
            mask = X[best] == val
            node.children[val] = self._build(X[mask], y[mask], depth+1)
        if X[best].isna().any():
            mask = X[best].isna()
            node.children[np.nan] = self._build(X[mask], y[mask], depth+1)
        return node

    def _gain(self, col, y):
        def entropy(s):
            counts = s.value_counts()
            probs = counts / len(s)
            return -(probs * np.log2(probs)).sum()
        total = entropy(y)
        weighted = sum(len(y[col == v]) / len(y) * entropy(y[col == v]) for v in col.dropna().unique())
        if col.isna().any():
            mask = col.isna()
            weighted += len(y[mask]) / len(y) * entropy(y[mask])
        return total - weighted

    def predict(self, X):
        return [self._predict_row(row, self.root) for _, row in X.iterrows()]

    def _predict_row(self, row, node):
        while node.feature is not None and node.children:
            val = row.get(node.feature, np.nan)
            next_node = node.children.get(val, node.children.get(np.nan))
            if next_node is None or next_node == node:
                break
            node = next_node
        return node.prediction



# Utilities

def split_data(X, y, test_size=0.1, dev_size=0.1):
    n = len(X)
    idx = np.random.permutation(n)
    test = int(n * test_size)
    dev = int(n * dev_size)
    return X.iloc[idx[dev+test:]], X.iloc[idx[test:dev+test]], X.iloc[idx[:test]], \
           y.iloc[idx[dev+test:]], y.iloc[idx[test:dev+test]], y.iloc[idx[:test]]

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(y_true, y_pred):
        mat.loc[t, p] += 1
    return mat

# Data loading

def load_data(path):
    df = pd.read_csv(path, header=None)
    return df.drop(columns=0), df[0]

# Main experiment

def run():
    for name, path in [('Mushroom', 'mushroom.data'), ('Voting', 'house-votes-84.data')]:
        print(f"\n=== {name} Dataset ===")
        X, y = load_data(path)
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

        print("\nNaive Bayes")
        nb = NaiveBayes().fit(X_train, y_train)
        print(f"Dev acc: {accuracy(y_dev, nb.predict(X_dev)):.4f}")
        y_pred = nb.predict(X_test)
        print(f"Test acc: {accuracy(y_test, y_pred):.4f}")
        print(confusion(y_test, y_pred))

        print("\nDecision Tree")
        dt = DecisionTree(max_depth=5).fit(X_train, y_train)
        print(f"Dev acc: {accuracy(y_dev, dt.predict(X_dev)):.4f}")
        y_pred = dt.predict(X_test)
        print(f"Test acc: {accuracy(y_test, y_pred):.4f}")
        print(confusion(y_test, y_pred))

if __name__ == "__main__":
    run()