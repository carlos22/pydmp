'''
Locally Weighted Regression (Eager)

@author: Karl Glatz

Ported from C++ lib_lwr; could be improved with some some vectorization
'''
import numpy as np
import math

class LWR(object):
  '''
    Eager Locally Weighted Regression
  '''

  def __init__(self, n_rfs=20, activation=0.1, cutoff=0.001, exponentially_spaced=True):
    '''
    Constructor
    '''
    # number of basis functions
    self.n_rfs = int(n_rfs)
    
    # widths of the kernels
    self.widths = np.zeros((self.n_rfs, 1))
    
    # centers of the kernels
    self.centers = np.zeros((self.n_rfs, 1))
    
    # slopes of the linear models
    self.slopes = None
    
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

  
  def learn(self, x_input_vec, y_target_vec):
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
      sxtd += psi * self.slopes[i] * x_query
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


# some basic tests
if __name__ == '__main__':
  
  # import pylab here, so the module does not depend on it!!
  import pylab as plt
  
  num_learn = 1000
  num_query = 2000
  
  # test simple LWR and plot it
  testfunc = lambda x:-pow(x - 0.5, 2)
  testfunc_vec = np.vectorize(testfunc)
  
  test_x = np.arange(start=0.0, stop=1.0, step=1.0 / num_learn)
  test_y = testfunc_vec(test_x)
  
  # create LWR Model
  lwr = LWR(n_rfs=20, activation=0.7, cutoff=0.001, exponentially_spaced=False)
  
  # learn
  lwr.learn(test_x, test_y)
  
  # create query x values
  test_xq = np.arange(start=0.0, stop=1.0, step=1.0 / num_query)
  test_ypredicted = []

  # calc prediction
  for x in test_xq:
    test_ypredicted.append(lwr.predict(x))
  
  
  print "lwr.get_thetas(): ", lwr.get_thetas()
  
  
  
  # ----- PLOT
  fig = plt.figure()
  
  # plot kernels
  ax1 = fig.add_subplot(211)
  lwr.plot_kernels(ax1)
  
  ax2 = fig.add_subplot(212)
  
  # plot testfunc
  ax2.plot(test_x, test_y, 'o')
  
  # plot prediction
  ax2.plot(test_xq, test_ypredicted, 'rx')
  
  # show plot
  plt.show()
  
