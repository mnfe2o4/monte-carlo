import numpy as np
import scipy.stats as stats

# Constants for numerical computations
def TARGET_FUNCTION(x1, x2):
    return np.exp(-x1**2 / 2 - 0.5*(2*x2 + x1**2 - 1)**2)

MEAN = [0, 0.5]
COV = np.array([[1, 0], [0, 0.25]])

def INITIAL_PROPOSAL_PDF(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    orig_shape = x1.shape

    pts = np.column_stack((x1.ravel(), x2.ravel()))
    vals = stats.multivariate_normal.pdf(pts, mean=MEAN, cov=COV)

    return vals.reshape(orig_shape)

def REFINED_PROPOSAL_PDF(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    orig_shape = x1.shape

    MEAN_REF = [0, 0.19]
    COV_REF = np.array([[1, 0], [0, 0.25]]) * (2.28**2)

    pts = np.column_stack((x1.ravel(), x2.ravel()))
    vals = stats.multivariate_normal.pdf(pts, mean=MEAN_REF, cov=COV_REF)

    return vals.reshape(orig_shape)

def INITIAL_WEIGHT_FUNCTION(x1, x2):
    rho = TARGET_FUNCTION(x1, x2)
    proposal_pdf = INITIAL_PROPOSAL_PDF(x1, x2)
    return rho / proposal_pdf

def REFINED_WEIGHT_FUNCTION(x1, x2):
    rho = TARGET_FUNCTION(x1, x2)
    proposal_pdf = REFINED_PROPOSAL_PDF(x1, x2)
    return rho / proposal_pdf
