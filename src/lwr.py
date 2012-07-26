'''
Created on 26.07.2012

@author: karl
'''
import numpy as np

class LWR(object):
  '''
    Eager Locally Weighted Regression for DMP
  '''

  def __init__(self, n_rfs):
    '''
    Constructor
    '''
    # number of basis functions
    self.n_rfs = n_rfs
    
    # widths of the kernels
    self.k_widths = []
    
    # heights of the kernels
    self.k_heights = []
    
    
    
  def _evaluate_kernel(self, x_input, center_idx):
    # gaussian kernel
    return np.exp(-(1.0 / self.widths[center_idx]) * (x_input - self.centers[center_idx] ** 2))

  def predict(self, x_query):
    pass
  
  def learn(self, input_vec, output_vec):
    pass
    
  def get_thetas(self):
    pass
  
  def set_thetas(self):
    pass
    
    

if __name__ == '__main__':
  
  # test simple LWR
  
  
