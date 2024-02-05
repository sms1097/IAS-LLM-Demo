import os
import numpy as np

from scipy.optimize import brentq
from scipy.stats import binom, norm
from stqdm import stqdm


def binomial_iid(N, alpha, muhat):
    def invert_upper_tail(mu):
        return binom.cdf(N * muhat, N, mu) - (alpha / 2)

    def invert_lower_tail(mu):
        return binom.cdf(N * muhat, N, mu) - (1 - alpha / 2)

    u = brentq(invert_upper_tail, 0, 1)
    l = brentq(invert_lower_tail, 0, 1)
    return np.array([l, u])


def pp_mean_iid_asymptotic(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha):
    n = Y_labeled.shape[0]
    N = Yhat_unlabeled.shape[0]
    tildethetaf = Yhat_unlabeled.mean()
    rechat = (Yhat_labeled - Y_labeled).mean()
    thetahatPP = tildethetaf - rechat
    sigmaftilde = np.std(Yhat_unlabeled)
    sigmarec = np.std(Yhat_labeled - Y_labeled)
    hw = norm.ppf(1 - alpha / 2) * np.sqrt((sigmaftilde**2 / N) + (sigmarec**2 / n))
    return [thetahatPP - hw, thetahatPP + hw]


def calculate_ppi(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha, num_trials=100):
    n_max = Y_labeled.shape[0]  # Total number of labeled ballots
    ns = np.linspace(1, n_max, 20).astype(int)

    imputed_estimate = (Yhat_labeled.sum() + Yhat_unlabeled.sum()) / (
        Yhat_labeled.shape[0] + Yhat_unlabeled.shape[0]
    )

    # Run prediction-powered inference and classical inference for many values of n
    ci = np.zeros((num_trials, ns.shape[0], 2))
    ci_classical = np.zeros((num_trials, ns.shape[0], 2))

    for i in stqdm(range(ns.shape[0]), desc="Running Prediciton Powered Inference"):
        for j in range(num_trials):
            # Prediction-Powered Inference
            n = ns[i]
            rand_idx = np.random.permutation(n)
            f = Yhat_labeled.astype(float)[rand_idx[:n]]
            y = Y_labeled.astype(float)[rand_idx[:n]]

            ci[j, i, :] = pp_mean_iid_asymptotic(y, f, Yhat_unlabeled, alpha)

            # Classical interval
            ci_classical[j, i, :] = binomial_iid(n, alpha, y.mean())

    ci_imputed = binomial_iid(Yhat_unlabeled.shape[0], alpha, imputed_estimate)

    return ci, ci_classical, ci_imputed, ns

