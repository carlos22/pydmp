'''

Simple discrete DMP implementation

Based on: 

Created on 25.07.2012

@author: Karl Glatz <glatz@hs-weingarten.de>
'''

import math
import numpy as np

class DiscreteDMP:

  #TYPE = ['original', 'improved']

  def __init__(self): #, method=TYPE[0], alpha=None, cutoff=0.001, k_gain=50.0, delta_t=0.001):
    
    ## Default values for parameters
               
    # s_min 
    cutoff = 0.001
    
    ## time constants choosen for critical damping
    # alpha (canonical system parameter)
    self.alpha_exec = 0.01
    self.alpha = abs(math.log(cutoff))
    
    # spring
    self.k_gain = 50.0
    
    # damping
    self.d_gain = self.k_gain / 4

    # time steps (for the run)
    self.delta_t = 0.001
    
    '''start position aka $x_0$'''
    self.start = 0.0
    
    '''goal position aka $g$'''
    self.goal = 1.0

    ''' temporal scaling factor $\tau$''' 
    self.tau = 2.0

    ## Transformation System
    #self.transformation_system = State()
    # current position, velocity, acceleration
    '''$x$ (position)'''
    self.x = 0.0
    '''$\dot x$ aka v (velocity)''' 
    self.xd = 0.0
    '''$\ddot x$ aka \dot v (acceleration)'''   
    self.xdd = 0.0 

    # internal variables
    ''' xd not scaled by tau'''
    self._raw_xd = 0.0
    
    '''current value of function f (perturbation function)'''
    self.f = 0.0
    
    #'''$\phi$ kernels of the function f'''
    #self.psi = np.exp(-0.5*)
    
    #'''weights of the function f'''
    #self.w = []
    
    #self.c = 
    #self.h =

    self.target_function_input = None
    self.target_function_ouput = None

    from lwpr import LWPR
    self.lwr_model = LWPR(1, 1)
    #self.lwr_model.init_D = 20 * np.eye(1)
    #self.lwr_model.update_D = True
    #self.lwr_model.init_alpha = 40 * np.eye(1)
    #self.lwr_model.meta = False
    
    #self.f_model = None
        

    # Canonical System
    #self.canonical_system = State()
    self.s = 1.0 # canonical system is starting with 1.0
    self.s_time = 0.0
        
    # is the DMP initialized?
    self._initialized = False
    
#  def _transformation_system_original(self, g, x, raw_xd, start, goal, f, s=0):
#    return (self.k_gain * (goal - x) - self.d_gain * raw_xd + (goal - start) * f) / self.tau
#
#  def _target_function_original(self, y, yd, ydd, goal, start, tau, s=0):
#    return ((-1 * self.k_gain * (goal - y) + self.d_gain * yd + tau * ydd) / (goal - start))

  def predict_f(self, x):
    
    # if nothing is learned we assume f=0.0
    if self.target_function_input == None:
      return 0.0
    
    #from lowess import loess_query
    #return loess_query(x, np.mat(self.target_function_input).T, self.target_function_ouput, 0.35)
    return self.lwr_model.predict(np.asarray([x]))[0]

  def setup(self, start, goal):
    assert not self._initialized
    # remember start position
    self.start = start
    # set current x to start (this is only a good idea if we not already started the dmp, which is the case for setup)
    self.x = start

    self.goal = goal
    self._initialized = True

  def _create_training_set(self, trajectory, frequency=1000):
    
    # get goal and start from trajectory
    start = trajectory[0][0]
    goal = trajectory[-1][0]
    tau = 1.0 # scaling factor for learning is always 1
    
    # the target function (transformation system solved for f, and plugged in y for x)
    ft = lambda y, yd, ydd: ((-1 * self.k_gain * (goal - y) + self.d_gain * yd + tau * ydd) / (goal - start))
    #ft = self._target_function_original
    
    # number of traning samples
    n_samples = len(trajectory)
    
    # duration of the movement
    duration = float(n_samples) / float(frequency)
    
    print "learn movement duration: %f" % duration
    
    # time step
    dt = duration / float(n_samples) 
    
    # evalutate function to get the target values for given training data
    target_function_ouput = []
    for d in trajectory:
      target_function_ouput.append(ft(d[0], d[1], d[2]))

    # target function input (canonical system)
    time_steps = np.arange(n_samples) * dt
    target_function_input = np.exp(time_steps * (self.alpha * -1. / tau))
    
    return target_function_input, np.asarray(target_function_ouput)
  
  def learn_batch(self, sample_trajectory, frequency):
    
    assert len(sample_trajectory) > 0
    
    if len(sample_trajectory[0]) != 3:
      # TODO: calculate xd and xdd if sample_trajectory does not contain it
      pass
    
    # get input and output of desired target function
    target_function_input, target_function_ouput = self._create_training_set(sample_trajectory, frequency)
    
    print target_function_input, target_function_ouput

    self.target_function_input = target_function_input
    self.target_function_ouput = target_function_ouput
    
    inM = np.asmatrix(target_function_input).T
    outM = np.asmatrix(target_function_ouput).T
    
    # learn lwpr model
    for i in range(len(target_function_input)):
      #print "value", outM[i]
      self.lwr_model.update(inM[i], outM[i])
    
    
    # TODO: learn weights

  def run_step(self):
    assert self._initialized
    
    # update s (canonical system)
    self.s = np.exp((self.alpha * -1 / self.tau) * self.s_time)
    self.s_time += self.delta_t
    
    # update f(s)
    self.f = self.predict_f(self.s)
    #print self.f
    
    # calculate xdd (vd) according to the transformation system equation 1
    self.xdd = (self.k_gain * (self.goal - self.x) - self.d_gain * self._raw_xd + (self.goal - self.start) * self.f) / self.tau

    # calculate xd using the raw_xd (scale by tau)
    self.xd = (self._raw_xd / self.tau)
    
    # integrate (update xd with xdd)
    self._raw_xd += self.xdd * self.delta_t
    
    # integrate (update x with xd) 
    self.x += self.xd * self.delta_t


if __name__ == '__main__':
  
  import pylab as plt
  from plot_tools import plot_pos_vel_acc_trajectory
  
  # only Transformation system (f=0)
  dmp = DiscreteDMP()
  dmp.setup(1.1, 2.2)
  
  traj = []
  for x in range(1000):
    #if x == 500:
    #  dmp.goal = 4.0
    dmp.run_step()
    traj.append([dmp.x, dmp.xd, dmp.xdd])
  
  fig = plt.figure('f=0 (transformation system only)')
  plot_pos_vel_acc_trajectory(fig, traj, dmp.delta_t)
  plt.show()



