import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.matmul(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        Xt = np.transpose(X)
        eye = np.eye(X.shape[1])
        eye[0, 0] = 0
        mt = np.matmul(Xt, X) + self.reg_lambda * eye
        w_opt = np.matmul(np.linalg.inv(mt), np.matmul(Xt, y))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        xb = np.empty(shape=(X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        xb[:, 0] = 1
        xb[:, 1:] = X
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======

        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        polynomial = sklearn.preprocessing.PolynomialFeatures(degree=self.degree)
        X_poly = np.empty((X.shape[0], 5), dtype=X.dtype)

        X_poly[:, 0] = X[:, 13]
        X_poly[:, 1] = X[:, 6]
        X_poly[:, 2] = X[:, 11]
        X_poly[:, 3] = X[:, 3]
        X_poly[:, 4] = X[:, 10]

        X_poly = polynomial.fit_transform(X_poly)

        X_transformed = np.empty((X.shape[0], X_poly.shape[1] + 2))
        X_transformed[:, 0:X_poly.shape[1]] = X_poly
        X_transformed[:, X_poly.shape[1]] = np.sqrt(X[:, 8])
        X_transformed[:, X_poly.shape[1] + 1] = np.sqrt(X[:, 7])
        # X_transformed = self.polynomial.fit_transform(X)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    feature_names = list(df.columns)
    feature_corr = []
    for feature in feature_names:
        if feature == target_feature:
            continue
        corr = df[target_feature].corr(df[feature])
        feature_corr.append((corr, feature))
    feature_corr.sort(reverse=True, key=lambda x: abs(x[0]))

    top_n_features = []
    top_n_corr = []

    for corr, feature in feature_corr[:n]:
        top_n_features.append(feature)
        top_n_corr.append(corr)

    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    kf = sklearn.model_selection.KFold(n_splits=k_folds)

    best_params = {}

    best_mse = None
    best_degree = None
    best_lambda = None

    for degree in degree_range:
        for lambda_r in lambda_range:
            mse = 0
            cnt = 0

            model.set_params(bostonfeaturestransformer__degree=degree, linearregressor__reg_lambda=lambda_r)
            # model = sklearn.pipeline.make_pipeline(
            #     BiasTrickTransformer(),
            #     BostonFeaturesTransformer(degree),
            #     LinearRegressor(lambda_r)
            # )

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                mse += np.sum((y_test - y_test_pred) ** 2)
                cnt += y_test.shape[0]
            mse /= cnt
            if best_mse is None or best_mse > mse:
                best_mse = mse
                best_degree = degree
                best_lambda = lambda_r
    best_params['bostonfeaturestransformer__degree'] = best_degree
    best_params['linearregressor__reg_lambda'] = best_lambda
    # ========================

    return best_params
