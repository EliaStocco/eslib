import numpy as np
from sklearn.metrics import mean_squared_error, \
    mean_absolute_error, r2_score, explained_variance_score, \
        max_error, mean_squared_log_error, median_absolute_error

#---------------------------------------#
# Dictionary of regression metrics
metrics = {
    "mse"   : mean_squared_error,
    "rmse"  : lambda x,y,**argv : mean_squared_error(x,y,squared=False,**argv),
    "norm"  : lambda x,y,**argv : np.linalg.norm(x-y,**argv),
    "mae"   : mean_absolute_error,
    "r2"    : r2_score,
    "ev"    : explained_variance_score,
    # "me"    : max_error,
    # "msle"  : mean_squared_log_error,
    "medae" : median_absolute_error,
}
