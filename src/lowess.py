import numpy as np
import math
from bisect import insort_left

def loess_query(x_query, X, y, alpha):
  if not isinstance(x_query, np.ndarray):
    x_query = np.array(x_query)
  elif isinstance(x_query, np.matrix):
    x_query = x_query.A

  if not isinstance(X, np.matrix):
    raise TypeError, 'X must be of type np.matrix'

  if isinstance(y, np.ndarray):
    y = np.mat(y).T

  if alpha <= 0 or alpha > 1:
    raise ValueError, 'ALPHA must be between 0 and 1'

  # inserting constant ones into X and X_QUERY for intercept term
  X = np.insert(X, obj=0, values=1, axis=1)
  x_query = np.insert(x_query, obj=0, values=1)

  # computing weights matrix using a tricube weight function
  W = weights_matrix(x_query, X, alpha)

  # computing theta from closed form solution to locally weighted linreg
  theta = (X.T * W * X).I * X.T * W * y

  # returning prediction
  return np.matrix.dot(theta.A.T, x_query)


def weights_matrix(x_query, X, alpha):
  if isinstance(x_query, np.matrix):
    x_query = x_query.A

  m = len(X)                # number of data points
  r = int(round(alpha * m)) # size of local region
  W = np.identity(m)        # skeleton for weights matrix

  sorted_list = []
  for i,row in enumerate(X):
    delta_norm = vector_norm(row - x_query)
    insort_left(sorted_list, delta_norm)
    W[i][i] = delta_norm

  # computing normalization constant based on alpha
  h_i = 1 / sorted_list[r - 1]

  # normalizing weights matrix
  W = W * h_i

  # applying tricube weight function to weights matrix
  for i in range(0, len(W)):
    W[i][i] = (1 - (W[i][i] ** 3)) ** 3 if W[i][i] < 1 else 0

  return np.mat(W)


def vector_norm(vector):
  len(vector)

  dot_product = np.matrix.dot(vector, vector.T)
  return math.sqrt(dot_product)