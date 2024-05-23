import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error, \
    mean_absolute_error, r2_score, explained_variance_score, \
        max_error, mean_squared_log_error, median_absolute_error

def vectorial_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient for vectorial values.

    The Pearson correlation coefficient is calculated using the formula:
    
    .. math::
        r = \\frac{\\sum_{i=1}^N (x_i - \\bar{x}) \\cdot (y_i - \\bar{y})}
                 {\\sqrt{\\sum_{i=1}^N (x_i - \\bar{x})^2 \\cdot \\sum_{i=1}^N (y_i - \\bar{y})^2}}

    where:
    - x_i and y_i are the vector components of the samples.
    - \\bar{x} and \\bar{y} are the means of the vector components of x and y, respectively.
    - N is the number of samples.

    Parameters:
    x (np.ndarray): A 2D numpy array where each row is a vector representing a sample.
    y (np.ndarray): A 2D numpy array where each row is a vector representing a sample.

    Returns:
    float: The Pearson correlation coefficient between the vectorial values of x and y.

    Raises:
    ValueError: If x and y do not have the same shape.
    AssertionError: If the computed Pearson correlation coefficient is not within the valid range [-1, 1].
    """
    # Check if the input arrays have the same shape
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Calculate the mean of each column (vector component) across all samples
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)

    # Calculate the deviations from the mean
    sigma_x = x - x_mean
    sigma_y = y - y_mean

    # Numerator: sum of element-wise products of deviations
    num = np.sum(sigma_x * sigma_y, axis=1)
    num = np.sum(num)

    # Denominator: product of the sum of squared deviations for x and y
    cov_x = np.sum(np.square(sigma_x))
    cov_y = np.sum(np.square(sigma_y))
    den = np.sqrt(cov_x * cov_y)

    # Pearson correlation coefficient
    r = num / den

    # Check that the result is within the valid range [-1, 1]
    assert np.abs(r) <= 1, "Pearson correlation coefficient out of range [-1, 1]"

    return r

#---------------------------------------#
# Dictionary of regression metrics
metrics = {
    "mse"    : mean_squared_error,
    "rmse"   : lambda x,y,**argv : mean_squared_error(x,y,squared=False,**argv),
    "relrmse"   : lambda x,y,**argv : mean_squared_error(x,y,squared=False,**argv)/np.linalg.norm(y,**argv),
    "norm"   : lambda x,y,**argv : np.linalg.norm(x-y,**argv),
    "mae"    : mean_absolute_error,
    "ev"     : explained_variance_score,
    # "me"    : max_error,
    # "msle"  : mean_squared_log_error,
    "medae"  : median_absolute_error,
    "r2"     : r2_score,
    "1-r2"   : lambda x,y,**argv : 1-r2_score(x,y,**argv),
    "r"   : lambda x,y,**argv : np.corrcoef(x,y,**argv)[0, 1],
    "1-r" : lambda x,y,**argv : 1-np.corrcoef(x,y,**argv)[0, 1],
    "vecr" : lambda x,y : vectorial_pearson(x,y)
}
