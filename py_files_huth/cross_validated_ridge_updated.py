# Functions to estimate cost for each lambda, by voxel:
from __future__ import division

from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time
from scipy.stats import zscore


def corr(X, Y):
    return np.mean(zscore(X) * zscore(Y), 0)


def R2(Pred, Real):
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0)
    return np.nan_to_num(1 - SSres / SStot)



def ridge_sk(X, Y, lmbda, solver = 'auto'):
    rd = Ridge(alpha=lmbda)
    rd.fit(X, Y)
    return rd.coef_.T


def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000]), solver = 'auto'):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge_sk(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error




class SimpleCVRidgeDiffLambda:
    def __init__(self, n_splits=10,lambdas=np.array([10 ** i for i in range(-6, 10)]), solver = 'auto'):
        """Cross-validated ridge estimator with different penalty per output.
        Arguments:
                  n_folds: number of CV folds.
                  lambdas: possible values of ridge penalty.
                   method: ridge sub-function to use: 
                                'plain': simple closed form solution.
                                  'svd': using SVD to speed up inverse.
                             'ridge_sk': using sklearn Ridge
        """
        self.n_splits = n_splits
        self.lambdas = lambdas
        self.ridge_CV_function = ridge_by_lambda_sk
        self.ridge_function = ridge_sk
        self.solver = solver
        self.weights = None

    def fit(self, X, Y, verbose=False): #, groups=None
        """Fit ridge model to given data.
        Arguments:
                 X: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
                 Y: 2-dimensional tensor of shape (n, m) where m is the number
                    of targets.
            groups: groups used for cross-validation; passed directly to
                    cv.split.
        A separate model is learned for each target i.e. Y[:, j].
        """

        n_outputs = Y.shape[1]
        nL = self.lambdas.shape[0]
        r_cv = np.zeros((nL, n_outputs))

        kf = KFold(n_splits=self.n_splits)
        start_t = time.time()
        for icv, (trn, val) in enumerate(kf.split(Y)):
            # Run training function for fold and score the result with R2
            cost = self.ridge_CV_function( X[trn], Y[trn], X[val], Y[val],lambdas=self.lambdas, solver = self.solver)
            r_cv += cost
            if verbose:
                print("split {} of {}, average split length {:.2f}".format(icv+1,self.n_splits,
                    (time.time() - start_t) / (icv + 1)))

        # Store the CV scores
        self.cv_scores = r_cv / self.n_splits

        # choose best lambda at each output
        argmin_lambda = np.argmin(r_cv, axis=0)
        self.chosen_lambda = np.array([self.lambdas[i] for i in argmin_lambda])
        self.weights = np.zeros((X.shape[1], Y.shape[1]))
        # iterate over lambdas, this is much faster than iterating over outputs most of the time!
        for idx_lambda in range(self.lambdas.shape[0]):  
            idx_output = argmin_lambda == idx_lambda
            if sum(idx_output)>0:
                self.weights[:, idx_output] = self.ridge_function(
                    X, Y[:, idx_output], self.lambdas[idx_lambda],  solver = self.solver
                )

        return self.weights, self.chosen_lambda, self.cv_scores

    def predict(self,X):
        assert(not self.weights is None)
        return np.dot(X,self.weights)

    def fit_predict(self,X,Y,Xtest=None):
        if Xtest is None:
            Xtest = X
        self.fit(X,Y)
        return self.predict(Xtest)

    def fit_predict_score(self,X,Y,Xtest,Ytest,cost_function=R2):
        pred = self.fit_predict(X,Y,Xtest)
        return cost_function(pred,Ytest)
    
    def predict_score(self,Xtest,Ytest,cost_function=R2):
        assert(not self.weights is None)
        pred = self.predict(Xtest)
        return cost_function(pred,Ytest)





