from collections import Counter

import numpy as np
import pandas as pd
import ipdb
import time
import matplotlib.pyplot as plt

# set random seed
np.random.seed(0)

configs = [
    {"iterations": 1000, "learning_rate": 0.01},
    {"iterations": 2000, "learning_rate": 0.1},
]

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Use min-max scaling to do normalization, transform features to [0, 1]
    """
    # TODO: 1%
    # compute the min and max values for each feature
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # find dimensions where x_min == x_max
    zero_variance_dims = np.where(X_max == X_min)[0]

    X_scaled = X.copy()
    # set value to 0 for dimensions where min == max
    for dim in zero_variance_dims:
        X_scaled[:, dim] = 0

    # min-max scaling for dimensions where min != max
    non_zero_variance_dims = np.where(X_max != X_min)[0]
    for dim in non_zero_variance_dims:
        X_scaled[:, dim] = (X[:, dim] - X_min[dim]) / (X_max[dim] - X_min[dim])

    return (X - X_min) / (X_max - X_min)


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Use z-score scaling to standardize the features.
    """
    # compute the mean for each feature
    means = np.mean(X, axis=0)

    # compute the standard deviation for each feature
    stds = np.std(X, axis=0)

    return (X - means) / stds


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_int[label] for label in y])

    return encoded_labels


def cross_validate(model_type, X, y, param_candidates, k=10):
    """
    Do cross validation for random forests.
    """
    fold_size = X.shape[0] // k  # do k-fold cross validation
    best_params = None
    best_score = float("-inf")

    for n_estimators, max_depth in param_candidates:
        scores = []
        for i in range(k):
            X_val = X[i * fold_size : (i + 1) * fold_size]
            y_val = y[i * fold_size : (i + 1) * fold_size]
            X_train = np.concatenate(
                [X[: i * fold_size], X[(i + 1) * fold_size :]], axis=0
            )
            y_train = np.concatenate(
                [y[: i * fold_size], y[(i + 1) * fold_size :]], axis=0
            )

            model = RandomForest(model_type, n_estimators, max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            if model_type == "classifier":
                score = accuracy(y_val, y_pred)
            elif model_type == "regressor":
                score = -mean_squared_error(y_val, y_pred)

            scores.append(score)

        avg_score = sum(scores) / len(scores)
        print("params", n_estimators, max_depth, avg_score)
        if (model.model_type == "classifier" and avg_score > best_score) or (
            model.model_type == "regressor" and avg_score < best_score
        ):
            best_score = avg_score
            best_params = [n_estimators, max_depth]

    return best_params


# 3. Models
class LinearModel:
    """
    I referred to several sources to complete my code.
    - softmax regression: http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/#:~:text=Softmax%20regression%20(or%20multinomial%20logistic,kinds%20of%20hand%2Dwritten%20digits.
    - logistic regression github repo: https://github.com/bamtak/machine-learning-implemetation-python/blob/master/Multi%20Class%20Logistic%20Regression.ipynb
    - ChatGPT
    - Professor Vivian's class presentation
    """

    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type
        self.weights = None
        self.n_classes = 0
        self.n_features = 0
        self.accuracys = []

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(
            X, 0, 1, axis=1
        )  # add one column with value 1 (for the interception b) in every row

        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        # TODO: 2%
        if self.model_type == "logistic":
            # do one-hot encoding for y
            self.y_one_hot = np.zeros((y.shape[0], self.n_classes))
            self.y_one_hot[np.arange(y.shape[0]), y] = 1

            # initialize weights for gradient descent
            # since for each row, we need the model to tell us the prob. of it be labeled as a certain class
            # we need the weight shape to be: class count * feature count
            self.weights = np.zeros((self.n_classes, self.n_features))

            # gradient descent
            for _ in range(self.iterations):
                gradients = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * gradients
            """
            # code for plotting model convergence
            for config in configs:
                self.accuracys = []
                self.weights = np.zeros((self.n_classes, self.n_features))
                for _ in range(config["iterations"]):
                    gradients = self._compute_gradients(X, y)
                    self.weights -= config["learning_rate"] * gradients
                    pred = self.predict(X)
                    self.accuracys.append(accuracy(y, pred))
                print("acc", self.accuracys[-1])
                plt.plot(
                    range(1, config["iterations"] + 1),
                    self.accuracys,
                    label=f"LR={config['learning_rate']}, Iter={config['iterations']}",
                )

            plt.title("Accuracy vs Iterations for Different Configurations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.show()
            """
        else:
            # implemented gradient descent
            # self.weights = np.zeros((1, self.n_features))
            # for _ in range(self.iterations):
            #     gradients = self._compute_gradients(X, y)
            #     self.weights -= self.learning_rate * gradients
            # close-form solution
            pseudo_inverse_X = np.linalg.pinv(X)
            weights = np.dot(pseudo_inverse_X, y)
            self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)  # TODO: add back
        if self.model_type == "linear":
            # TODO: 2%
            pred_vals = np.dot(X, self.weights.T)
            return pred_vals
        elif self.model_type == "logistic":
            # TODO: 2%
            pred_vals = np.dot(X, self.weights.T).reshape(-1, self.n_classes)
            class_with_prob = self._softmax(
                pred_vals
            )  # use softmax for multi-class, sigmoid for binary class

            return np.argmax(class_with_prob, axis=1)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            h = np.dot(X, self.weights.T).reshape(-1)
            error = h - y
            gradient = np.dot(X.T, error) / len(y)

            return gradient
        elif self.model_type == "logistic":
            # TODO: 3%
            # compute differentiated error as the gradient
            # TODO: I used a different way to compute the gradient
            h = self._softmax(np.dot(X, self.weights.T).reshape(-1, self.n_classes))
            gradient = np.dot((h - self.y_one_hot).T, X) / len(X)

            return gradient

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the probability of the row to be labeled as each class.
        """
        exp = np.exp(z)

        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    """
    I referred to several sources to complete my code.
    - decision tree for multi-classification: https://pandulaofficial.medium.com/implementing-cart-algorithm-from-scratch-in-python-5dd00e9d36e
    - ChatGPT
    - Professor Vivian's class presentation
    """

    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type
        self.tree = None

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        left_X, left_y, right_X, right_y = self._split_data(X, y, feature, threshold)
        left_tree = self._build_tree(left_X, left_y, depth + 1)
        right_tree = self._build_tree(right_X, right_y, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            return np.bincount(y).argmax()  # label as the most frequent class
        else:
            # TODO: 1%
            return np.mean(y)  # label as the mean value

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """
        Try out each feature to find the best split (branching factor).
        We use gini inpurity for classification problems and mse for linear regression
        to measure quality of each feature.
        """
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])  # sort rows based on features
            for i in range(1, len(X)):
                # if the i-th row has a different value from the previous row for the current feature
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2  # set the split point
                    mask = X[:, feature] <= threshold
                    left_y, right_y = (
                        y[mask],
                        y[~mask],
                    )  # divide data based on the threshold

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """
        A method to measure the purity of a group. It ranges from 0 to 0.5, where:
        - 0 means all members are in the same class
        - 0.5 means members are evenly distributed in each class
        """
        # TODO: 4%
        size_left = len(left_y)
        size_right = len(right_y)
        total_size = size_left + size_right

        if total_size == 0:
            return 0

        # compute the probability of being labeled as each class
        p_left = (
            np.bincount(left_y) / size_left
            if size_left > 0
            else np.zeros(len(np.bincount(left_y)))
        )
        p_right = (
            np.bincount(right_y) / size_right
            if size_right > 0
            else np.zeros(len(np.bincount(right_y)))
        )

        # compute gini inpurity (gini index) for left and right trees
        gini_left = 1 - np.sum(p_left**2)
        gini_right = 1 - np.sum(p_right**2)

        # compute weighted average of gini inpurity
        gini = (size_left / total_size) * gini_left + (
            size_right / total_size
        ) * gini_right

        return gini

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # compute mse for left and right subtree
        mse_left = 0
        if len(left_y) > 0:
            mse_left = np.mean((left_y - np.mean(left_y)) ** 2)
        mse_right = 0
        if len(right_y) > 0:
            mse_right = np.mean((right_y - np.mean(right_y)) ** 2)

        # compute weighted mse
        mse = (len(left_y) * mse_left + len(right_y) * mse_right) / (
            len(left_y) + len(right_y)
        )

        return mse

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node

    def _split_data(
        self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float
    ) -> tuple:
        """
        Split the array (do branching) based on the feature and the corresponding threshold.
        """
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_X = X[left_indices]
        left_y = y[left_indices]
        right_X = X[right_indices]
        right_y = y[right_indices]

        return left_X, left_y, right_X, right_y


class RandomForest:
    """
    I referred to chatGPT and Professor Vivian's class presentation to complete my code.
    """

    def __init__(
        self,
        model_type: str = "classifier",
        n_estimators: int = 100,
        max_depth: int = 5,
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = [
            DecisionTree(max_depth=max_depth, model_type=model_type)
            for _ in range(n_estimators)
        ]  # initialize n decision trees first, will fit them later

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Random sample input data in this case, rather than performing
        feature bagging.
        """
        # TODO: 2%
        n_samples = X.shape[0]
        for tree in self.trees:
            # bootstrap sampling
            bootstrap_indices = np.random.choice(
                n_samples, n_samples, replace=True
            )  # replace means after a sample is selected, it'll be placed back into the pool to be sampled again
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For each sample, we derive predictions from different decision trees.
        Finally, uniform their predictions using
        - majority for classification
        - mean value for linear regression
        """
        # TODO: 2%
        # derive predictions, shape is (tree count, sample count)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # uniform predictions
        if self.model_type == "classifier":
            result = []
            for sample_pred in tree_predictions.T:
                unique, counts = np.unique(sample_pred, return_counts=True)
                result.append(unique[np.argmax(counts)])
            return np.array(result)
        else:
            return np.mean(tree_predictions, axis=0)


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    correct_predictions = np.sum(y_pred == y_true)
    accuracy = correct_predictions / y_true.shape[0]

    return accuracy


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    squared_error = (y_true - y_pred) ** 2
    mse = np.mean(squared_error)

    return mse


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    # X_train, X_test = standardize(X_train), standardize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    # param_candidates = [[100, 2], [100, 5], [100, 10], [200, 2], [200, 5], [200, 10]]
    # best_params = cross_validate(
    #     "classifier", X_train, y_train, param_candidates
    # )  # find validation set from training set
    # random_forest_classifier = RandomForest(
    #     model_type="classifier", n_estimators=best_params[0], max_depth=best_params[1]
    # )
    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)
    # X_train, X_test = standardize(X_train), standardize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    # param_candidates = [[100, 2], [100, 5], [100, 10], [200, 2], [200, 5], [200, 10]]
    # best_params = cross_validate(
    #     "regressor", X_train, y_train, param_candidates
    # )  # find validation set from training set
    # random_forest_regressor = RandomForest(
    #     model_type="regressor", n_estimators=best_params[0], max_depth=best_params[1]
    # )
    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
