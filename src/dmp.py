'''
Simple one-dimensional discrete DMP implementation

@author: Karl Glatz <glatz@hs-weingarten.de>
'''
import math
import numpy as np
from lwr import LWR

class DiscreteDMP:

  #TYPE = ['original', 'improved']

  def __init__(self): #, method=TYPE[0], alpha=None, cutoff=0.001, k_gain=50.0, delta_t=0.001):
    
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
    ''' xd not scaled by tau'''
    self._raw_xd = 0.0
    
    '''current value of function f (perturbation function)'''
    self.f = 0.0
    
    # target function input (x) and output (y)
    self.target_function_input = None
    self.target_function_ouput = None
    # debugging (y values predicted by fitted lwr model)
    self.target_function_predicted = None

    # create LWR model and set parameters
    self.lwr_model = LWR(activation=0.1, exponentially_spaced=True, n_rfs=20)
    
#    from lwpr import LWPR
#    self.lwr_model = LWPR(1, 1)
#    self.lwr_model.init_D = 20 * np.eye(1)
#    self.lwr_model.update_D = True
#    self.lwr_model.init_alpha = 40 * np.eye(1)
#    self.lwr_model.meta = False
            
    # Canonical System
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
    ft = lambda y, yd, ydd: ((-(self.k_gain) * (goal - y) + self.d_gain * yd + tau * ydd) / (goal - start))
    
    # evaluate function to get the target values for given training data
    target_function_ouput = []
    for i, d in enumerate(trajectory):
      # compute f_target(y, yd, ydd) * s
      #print "s ", target_function_input[i], "y ", d[0], "yd ", d[1], "ydd", d[2], " ft:", ft(d[0], d[1], d[2])
      target_function_ouput.append(ft(d[0], d[1], d[2]))
    
    return target_function_input, np.asarray(target_function_ouput)
  
  def learn_batch(self, sample_trajectory, frequency):
    '''
     Learns the DMP by a given sample trajectory
     @param sample_trajectory: list of tuples (pos,vel,acc)
    '''
    assert len(sample_trajectory) > 0
    
    if len(sample_trajectory[0]) != 3:
      # TODO: calculate yd and ydd if sample_trajectory does not contain it
      raise NotImplementedError("automatic derivation of yd and ydd not yet implemented!")
    
    # get input and output of desired target function
    target_function_input, target_function_ouput = self._create_training_set(sample_trajectory, frequency)

    # save input/output of f_target
    self.target_function_input = target_function_input
    self.target_function_ouput = target_function_ouput
    
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
    self.f = self.predict_f(self.s)
    #print self.f
    
    # calculate xdd (vd) according to the transformation system equation 1
    self.xdd = (self.k_gain * (self.goal - self.x) - self.d_gain * self._raw_xd + (self.goal - self.start) * self.f) / self.tau

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



