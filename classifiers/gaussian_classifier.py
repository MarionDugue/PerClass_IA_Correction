from scipy.stats import multivariate_normal as mvn
import numpy as np
import logging

class GaussianClassifier:
    def __init__(self, enable_logging=False):
        self.info = dict()
        self.info['type'] = 'gaussian'
        self.means = []
        self.cov = []

        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.CRITICAL)

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]

        self.means = np.full([self.n_classes, self.n_features], np.nan)
        self.cov = np.full([self.n_classes, self.n_features, self.n_features], np.nan)
        self.mvn = dict()

        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.means[cls, :] = X_cls.mean(axis=0)
            self.cov[cls, :, :] = np.cov(X_cls.T)
            self.mvn[str(cls)] = mvn(self.means[cls], self.cov[cls, :, :])

    def predict(self, X_test):
        N_test, d_test = X_test.shape
        proba = np.full((N_test, self.n_classes), np.nan)

        for cls in self.classes:
            proba[:, cls] = self.mvn[str(cls)].pdf(X_test)
        
        y_pred = np.argmax(proba, axis=1)
        return y_pred, proba
