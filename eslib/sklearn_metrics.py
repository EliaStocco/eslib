import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error, \
    mean_absolute_error, r2_score, explained_variance_score, \
        max_error, mean_squared_log_error, median_absolute_error

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
}
