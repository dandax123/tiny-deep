import numpy as np
def compute_cost(y_pred, y_true, m):
    cost = -1 / m * np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return np.squeeze(np.array(cost))