'''
Simple one-dimensional discrete DMP implementation

@author: Karl Glatz <glatz@hs-weingarten.de>
'''
import math
import numpy as np
from lwr import LWR

class DiscreteDMP:

  def __init__(self, improved_version=False):
    
    ## time constants choosen for critical damping
    
    # s_min 
    self.cutoff = 0.001
    
    # alpha (canonical system parameter)
    self.alpha = abs(math.log(self.cutoff))
    
    # spring term
    self.k_gain = 50.0
    
    # damping term
    self.d_gain = self.k_gain / 4

    # time steps (for the run)
    self.delta_t = 0.001
    
    ## state variables 
    
    '''start position aka $x_0$'''
    self.start = 0.0
    
    '''goal position aka $g$'''
    self.goal = 1.0

    '''movement duration: temporal scaling factor $\tau$''' 
    self.tau = 1.0

    ## Transformation System
    # current position, velocity, acceleration
    '''$x$ (position)'''
    self.x = 0.0
    '''$\dot x$ aka v (velocity)''' 
    self.xd = 0.0
    '''$\ddot x$ aka \dot v (acceleration)'''   
    self.xdd = 0.0 

    # internal variables
    ''' xd not scaled by tau aka v'''
    self._raw_xd = 0.0
    
    '''current value of function f (perturbation function)'''
    self.f = 0.0
    
    # target function input (x) and output (y)
    self.target_function_input = None
    self.target_function_ouput = None
    # debugging (y values predicted by fitted lwr model)
    self.target_function_predicted = None

    # do not predict f by approximated function, use values of ft directly - only works if duration is 1.0!!
    self.use_ft = False

    # create LWR model and set parameters
    self.lwr_model = LWR(activation=0.1, exponentially_spaced=True, n_rfs=20)
            
    # Canonical System
    self.s = 1.0 # canonical system is starting with 1.0
    self.s_time = 0.0
        
    # is the DMP initialized?
    self._initialized = False
    
    # set the correct transformation and ftarget functions
    if improved_version:
      self._transformation_func = self._transformation_func_improved
      self._ftarget_func = self._ftarget_func_improved
    else:
      self._transformation_func = self._transformation_func_original
      self._ftarget_func = self._ftarget_func_original
  
  # original formulation
  @staticmethod
  def _transformation_func_original(k_gain, d_gain, x, raw_xd, start, goal, tau, f, s):
    return (k_gain * (goal - x) - d_gain * raw_xd + (goal - start) * f) / tau
  
  @staticmethod
  def _ftarget_func_original(k_gain, d_gain, y, yd, ydd, goal, start, tau, s):
    return ((-1 * k_gain * (goal - y) + d_gain * yd + tau * ydd) / (goal - start))

  # improved version of formulation
  @staticmethod
  def _transformation_func_improved(k_gain, d_gain, x, raw_xd, start, goal, tau, f, s):
    #return (k_gain * (goal - x) - d_gain * raw_xd + k_gain * (goal - start) * s + k_gain * f) / tau
    return (k_gain * (goal - x) - d_gain * raw_xd - k_gain * (goal - start) * s + k_gain * f) / tau
  
  @staticmethod
  def _ftarget_func_improved(k_gain, d_gain, y, yd, ydd, goal, start, tau, s):
    #return ((tau * ydd - d_gain * yd) / k_gain ) + (goal - y) - ((goal - start) * s)
    return ((tau**2 * ydd + d_gain * yd * tau) / k_gain ) - (goal - y) + ((goal - start) * s)


  # predict f
  def predict_f(self, x):
    
    # if nothing is learned we assume f=0.0
    if self.target_function_input == None:
      return 0.0
    
    #return self.lwr_model.predict(np.asarray([x]))[0]
    return self.lwr_model.predict(x)

  def setup(self, start, goal, duration):
    assert not self._initialized
    # set start position
    self.start = start
    
    # set current x to start (this is only a good idea if we not already started the dmp, which is the case for setup)
    self.x = start
    
    # set goal
    self.goal = goal
    
    self.tau = duration
    
    self._initialized = True

  def _create_training_set(self, trajectory, frequency):
    '''
      Prepares the data set for the supervised learning
      @param trajectory: list of 3-Tuples with (pos,vel,acc) 
    '''
    # number of training samples
    n_samples = len(trajectory)
    
    # duration of the movement
    duration = float(n_samples) / float(frequency)
    
    # set tau to duration for learning
    tau = duration
    
    # initial goal and start obtained from trajectory
    start = trajectory[0][0]
    goal = trajectory[-1][0]
    
    print "create training set of movement from trajectory with %i entries (%i hz) with duration: %f, start: %f, goal: %f" % (n_samples, frequency, duration, start, goal)
    
    ##### compute target function input (canonical system) [rollout]
    
    # compute alpha_x such that the canonical system drops
    # below the cutoff when the trajectory has finished
    alpha = -(math.log(self.cutoff))
    
    # time steps
    dt = 1.0 / n_samples # delta_t for learning
    time = 0.0
    target_function_input = np.zeros(n_samples)
    for i in xrange(len(target_function_input)):
      target_function_input[i] = math.exp(-(alpha / tau) * time) 
      time += dt
    
#    import pylab as plt
    
#    plt.figure()
#    plt.plot(target_function_input)
#    plt.show()
    
    # vectorized:
    #time_steps = np.arange(0, 1.0, dt)
    #target_function_input2 = np.exp(-(alpha) * time_steps)
    
    #if (target_function_input == target_function_input2).all():
    #  print "same!"
    
    # print "target_function_input",len(target_function_input)
    
    ##### compute values of target function
    
    # the target function (transformation system solved for f, and plugged in y for x)
    #ft = lambda y, yd, ydd, s: ((-(self.k_gain) * (goal - y) + self.d_gain * yd + tau * ydd) / (goal - start))
    ft = lambda y, yd, ydd, s: self._ftarget_func(self.k_gain, self.d_gain, y, yd, ydd, goal, start, tau, s)
    
    # evaluate function to get the target values for given training data
    target_function_ouput = []
    for i, d in enumerate(trajectory):
      # compute f_target(y, yd, ydd) * s
      #print "s ", target_function_input[i], "y ", d[0], "yd ", d[1], "ydd", d[2], " ft:", ft(d[0], d[1], d[2])
      target_function_ouput.append(ft(d[0], d[1], d[2], target_function_input[i]))
    
    return target_function_input, np.asarray(target_function_ouput)
  
  @staticmethod
  def compute_derivatives(pos_trajectory, frequency):
    # ported from trajectory.cpp 
    # no fucking idea why doing it this way - but hey, the results are better ^^
    
    add_pos_points = 4
    #add points to start
    for _ in range(add_pos_points):
      first_point = pos_trajectory[0]
      pos_trajectory.insert(0, first_point)
      
    # add points to the end
    for _ in range(add_pos_points):
      first_point = pos_trajectory[-1]
      pos_trajectory.append(first_point)
      
    # derive positions
    vel_trajectory = []
    
    for i  in range(len(pos_trajectory) - 4):
      vel = (pos_trajectory[i] - (8.0 * pos_trajectory[i + 1]) + (8.0 * pos_trajectory[i + 3]) - pos_trajectory[i + 4]) / 12.0
      vel *= frequency
      vel_trajectory.append(vel)
    
    
    # derive velocities
    acc_trajectory = []
    for i  in range(len(vel_trajectory) - 4):     
      acc = (vel_trajectory[i] - (8.0 * vel_trajectory[i + 1]) + (8.0 * vel_trajectory[i + 3]) - vel_trajectory[i + 4]) / 12.0
      acc *= frequency
      acc_trajectory.append(acc)
        
    result_traj = zip(pos_trajectory[4:], vel_trajectory[2:], acc_trajectory)
  
    return result_traj
  
  def learn_batch(self, sample_trajectory, frequency):
    '''
     Learns the DMP by a given sample trajectory
     @param sample_trajectory: list of tuples (pos,vel,acc)
    '''
    assert len(sample_trajectory) > 0
    
    if isinstance(sample_trajectory[0], float):
      # calculate yd and ydd if sample_trajectory does not contain it
      print "automatic derivation of yd and ydd"
      sample_trajectory = self.compute_derivatives(sample_trajectory, frequency)
      
    
    if len(sample_trajectory[0]) != 3:
      raise Exception("malformed trajectory, has to be a list with 3-tuples [(1,2,3),(4,5,6)]")
    
    # get input and output of desired target function
    target_function_input, target_function_ouput = self._create_training_set(sample_trajectory, frequency)

    # save input/output of f_target
    self.target_function_input = target_function_input
    self.target_function_ouput = target_function_ouput
    print  "target_function_ouput len: ", len(target_function_ouput)
    
    # learn LWR Model for this transformation system
    self.lwr_model.learn(target_function_input, target_function_ouput)
    
    
#    inM = np.asmatrix(target_function_input).T
#    outM = np.asmatrix(target_function_ouput).T
#    # learn lwpr model
#    for i in range(len(target_function_input)):
#      #print "value", outM[i]
#      self.lwr_model.update(inM[i], outM[i])

    # debugging: compute learned ft(x)
    self.target_function_predicted = []
    for x in target_function_input:
      self.target_function_predicted.append(self.predict_f(x))      
      
      

  def run_step(self):
    '''
      runs a integration step - updates variables self.x, self.xd, self.xdd
    '''
    assert self._initialized
    
    dt = self.delta_t 
    
    ### integrate transformation system   
    
    # update f(s)
    # debugging: use raw function output (time must be 1.0)
    if self.use_ft:
      print "DEBUG: using ft without approximation"
      ftinp = list(self.target_function_input)
      ft = self.target_function_ouput[ftinp.index(self.s)]
      self.f = ft
    else:
      f = self.predict_f(self.s)
      self.f = f
    
    
    # calculate xdd (vd) according to the transformation system equation 1
    #self.xdd = (self.k_gain * (self.goal - self.x) - self.d_gain * self._raw_xd + (self.goal - self.start) * self.f) / self.tau
    self.xdd = self._transformation_func(self.k_gain, self.d_gain, self.x, self._raw_xd, self.start, self.goal, self.tau, self.f, self.s) 

    # calculate xd using the raw_xd (scale by tau)
    self.xd = (self._raw_xd / self.tau)
    
    # integrate (update xd with xdd)
    self._raw_xd += self.xdd * dt
    
    # integrate (update x with xd) 
    self.x += self.xd * dt
    
    
    # integrate canonical system
    self.s = math.exp(-(self.alpha / self.tau) * self.s_time)
    self.s_time += dt
    


if __name__ == '__main__':
  
  import pylab as plt
  from plot_tools import plot_pos_vel_acc_trajectory
  
  # only Transformation system (f=0)
  dmp = DiscreteDMP()
  dmp.setup(1.1, 2.2, 1.0)
  
  traj = []
  for x in range(1000):
    #if x == 500:
    #  dmp.goal = 4.0
    dmp.run_step()
    traj.append([dmp.x, dmp.xd, dmp.xdd])
  
  fig = plt.figure('f=0 (transformation system only)', figsize=(10, 3))
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)
  ax3 = fig.add_subplot(133)
  plot_pos_vel_acc_trajectory((ax1, ax2, ax3), traj, dmp.delta_t, label='DMP $f=0$', linewidth=1)
  
  fig.tight_layout()
  
  plt.show()



