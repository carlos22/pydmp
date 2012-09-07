'''
Locally Weighted Regression (Eager)

@author: Karl Glatz

Ported from C++ lib_lwr; added offset calculation
'''
import numpy as np
import math
from numpy import linalg as LA

class LWR(object):
  '''
    Eager Locally Weighted Regression for one dimension.
    Eager means there is no data stored, the local models are fitted once to a fixed set of basis functions.
  '''

  def __init__(self, n_rfs=20, activation=0.1, cutoff=0.001, exponentially_spaced=True, use_offset=False):
    '''
    Constructor
    '''
    # number of basis functions
    self.n_rfs = int(n_rfs)
    
    # widths of the kernels
    self.widths = np.zeros((self.n_rfs, 1))
    
    # centers of the kernels
    self.centers = np.zeros((self.n_rfs, 1))
    
    # slopes of the linear models [a in y = ax + b]
    self.slopes = [None]*self.n_rfs
    
    # offsets b in y = ax + b
    self.offsets = [0.0]*self.n_rfs
    
    self.activation = activation
    
    # default learn method
    if use_offset:
      self.learn = self.learn_with_offset
    else:
      self.learn = self.learn_without_offset
    
    
    # initialize centers and widths (which is done at prediction time in lazy-lwr)    
    if exponentially_spaced:
      # exponentially_spaced kernels (set centers and widths)
      last_input_x = 1.0
      alpha_x = -math.log(cutoff)
      for i in range(self.n_rfs): 
        t = (i + 1) * (1. / (n_rfs - 1)) * 1.0; # 1.0 is the default duration
        input_x = math.exp(-alpha_x * t)
        self.widths[i] = (input_x - last_input_x) ** 2 / -math.log(activation)
        self.centers[i] = last_input_x
        last_input_x = input_x
    else:
      # equally spaced
      diff = 0
      if self.n_rfs == 1:
        self.centers[0] = 0.5;
        diff = 0.5;
      else:
        for i in range(self.n_rfs):
          self.centers[i] = float(i) / float(self.n_rfs - 1)
        diff = float(1.0) / float(self.n_rfs - 1);
      
      width = -pow(diff / 2.0, 2) / math.log(activation)
      for i in range(self.n_rfs):
        self.widths[i] = width;
    
      

  def _generate_basis_function_mat(self, input_vec):
    '''
    Generates a Matrix of Basis functions with num_of_points x basis_functions (rows x cols)
    '''
    # create empty matrix
    basisfunc_mat = np.zeros((len(input_vec), len(self.centers)))
    
    for i in xrange(basisfunc_mat.shape[0]):
      for j in xrange(basisfunc_mat.shape[1]):
        basisfunc_mat[i, j] = self._evaluate_kernel(input_vec[i], j)
        
    return basisfunc_mat
    
    
  def _evaluate_kernel(self, x_input, center_idx):
    '''
      Gaussian Kernel
    '''
    return math.exp(-(1.0 / self.widths[center_idx]) * pow(x_input - self.centers[center_idx], 2))


  def learn_with_offset(self, x_input_vec, y_target_vec):
    assert len(x_input_vec) == len(y_target_vec)
    data_len = len(x_input_vec)
    
    # add constant 1
    x_vec = [np.asarray([1.0,x]).T for x in x_input_vec]

    # matrix with all x_i as cols
    X = np.asarray(x_vec)
    y_target_vec = np.asarray(y_target_vec)

    
    # calculate n_rfs betas
    for idx in range(self.n_rfs):
      
      # create diagonal weight matrix 
      W = np.zeros((data_len,data_len))
      for i, x in enumerate(x_input_vec):
        W[i,i] = self._evaluate_kernel(x, idx)
        
        
      beta = LA.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y_target_vec)
      
      self.offsets[idx] = beta[0]
      self.slopes[idx] = beta[1]

  
  def learn_without_offset(self, x_input_vec, y_target_vec):
    # array or matrix? -> http://www.scipy.org/NumPy_for_Matlab_Users/#head-e9a492daa18afcd86e84e07cd2824a9b1b651935
    
    assert len(x_input_vec) == len(y_target_vec)
    
    # ensure input are numpy arrays (lists are converted without copy overhead)
    x_input_vec = np.asarray(x_input_vec)
    y_target_vec = np.asarray(y_target_vec)
    
    # generate basis function matrix
    basis_function_matrix = self._generate_basis_function_mat(x_input_vec)
    
    # convert input to matrix_like arrays (row vectors)
    x_input_vec = x_input_vec.reshape(len(x_input_vec), 1)
    y_target_vec = y_target_vec.reshape(len(y_target_vec), 1)
    
    tmp_matrix_a = np.dot(np.square(x_input_vec), np.ones((1, self.n_rfs))) 
    
    tmp_matrix_a = (tmp_matrix_a * basis_function_matrix)
    assert tmp_matrix_a.shape == (len(x_input_vec), self.n_rfs)
    
    
    tmp_matrix_sx = tmp_matrix_a.sum(axis=0) # column wise summation
    assert tmp_matrix_sx.shape == (self.n_rfs,) 
    
    tmp_matrix_b = np.dot((x_input_vec * y_target_vec), np.ones(shape=(1, self.n_rfs)))
    tmp_matrix_b = (tmp_matrix_b * basis_function_matrix)
    assert tmp_matrix_b.shape == (len(x_input_vec), self.n_rfs)
    
    tmp_matrix_sxtd = tmp_matrix_b.sum(axis=0)
    assert tmp_matrix_sxtd.shape == (self.n_rfs,)
    
    # TODO: thinkg about this...
    ridge_regression = 0.0000000001
    self.slopes = (tmp_matrix_sxtd / (tmp_matrix_sx + ridge_regression))
    
    
  def predict(self, x_query):
    if self.slopes == None:
      raise Exception("No theta values available! Call learn() first or set thetas using set_thetas()!")
    
    sx = 0.0
    sxtd = 0.0
    for i in range(self.n_rfs):
      psi = self._evaluate_kernel(x_query, i)
      sxtd += psi * (self.slopes[i] * x_query + self.offsets[i])
      sx += psi
      
    # TODO: thinkg about this...
    if sx < 0.000000001 and sx > -0.000000001:
      return 0
    
    return sxtd / sx
     
    
  def get_thetas(self):
    return self.slopes
  
  def set_thetas(self, thetas):
    self.slopes = thetas
    
  def plot_kernels(self, ax):

    kernel_x = np.arange(0.0, 1.0, 0.001)
    
    for i in range(self.n_rfs):
      kernel_y = []
      for j in kernel_x:
        kernel_y.append(self._evaluate_kernel(j, i))
        
      ax.plot(kernel_x, kernel_y)


  def plot_linear_models(self, ax):
    eval_kernel_vec = np.vectorize(self._evaluate_kernel)

    for i, t in enumerate(self.get_thetas()):
      
      xvals = np.arange(start=0.0, stop=1.0, step=0.001)
      values = eval_kernel_vec(xvals, i)
      
      start, end = None, None
      for k, v in enumerate(values):
        if start == None and v >= self.activation:
          start = k
        if start != None and v < self.activation:
          end = k
          break
      
      #print i, "start: ", start, " end: ", end
      xpart = xvals[start:end]
      
      ax.axvline(x=xpart[0], color='lightgrey', linestyle='dashed')
      
      ax.plot(xpart, t* xpart + self.offsets[i])
      
      
      #ax.axvline(x=float(xpart[-1]))

# some basic tests
if __name__ == '__main__':
  
  # import pylab here, so the module does not depend on it!!
  import pylab as plt
  
  num_learn = 100
  num_query = 2000
  
  stop = 1.0
  
  # test simple LWR and plot it
  #testfunc = lambda x:-pow(x - 0.5, 2)
  #testfunc_vec = np.vectorize(testfunc)
  testfunc_vec = lambda x: x*np.sin(x*10) + np.random.randn(len(x))/10
  
  test_x = np.arange(start=0.0, stop=stop, step=stop / num_learn)
  test_y = testfunc_vec(test_x)
  
  # create LWR Model
  lwr = LWR(n_rfs=10, activation=0.7, cutoff=0.001, exponentially_spaced=False)
  
  # learn
  lwr.learn_with_offset(test_x, test_y)
  
  # create query x values
  test_xq = np.arange(start=0.0, stop=stop, step=stop / num_query)
  test_ypredicted = []

  # calc prediction
  for x in test_xq:
    test_ypredicted.append(lwr.predict(x))
  
  
  print "lwr.get_thetas(): ", lwr.get_thetas()
  
  
  
  # ----- PLOT
  fig = plt.figure('lwr', figsize=(12, 10))
  

  ax1 = fig.add_subplot(311)
  
  
  # plot testfunc
  ax1.plot(test_x, test_y, 'o', color='gray', label="training data")
  
  # plot prediction
  ax1.plot(test_xq, test_ypredicted, 'r', linewidth=2, label="prediction")

  plt.setp(ax1.get_xticklabels(), visible=False) 

  ax1.set_ylabel("$f(x)$")
  
  ax1.legend(loc='upper left')

  # linear models
  ax_lm = fig.add_subplot(312, sharex = ax1, sharey = ax1)  
  
  ax_lm.set_ylabel("$f(x)$")
  plt.setp(ax_lm.get_xticklabels(), visible=False) 
  
  lwr.plot_linear_models(ax_lm)
  
  
  # plot kernels
  ax_k = fig.add_subplot(313, sharex = ax1)
  lwr.plot_kernels(ax_k)
  ax_k.set_ylabel("$f(x)$")
  ax_k.set_xlabel("$x$")
  
  
  fig.tight_layout()
  
  # show plot
  plt.show()
  
