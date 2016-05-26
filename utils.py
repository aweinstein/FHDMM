import numpy as np

def softmax(Qs, beta):
    """Compute softmax probabilities for all actions.

    Parameters
    ----------

    Qs: array-like
        Action values for all the actions
    beta: float
        Inverse temperature

    Returns
    -------
    array-like
        Probabilities for each action
    """

    num = np.exp(Qs * beta)
    den = np.exp(Qs * beta).sum()
    return num / den
