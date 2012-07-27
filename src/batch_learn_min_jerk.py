#!/usr/bin/env python
'''
Tests for the Dynamic Movement Primitive
'''
import numpy as np
import pylab as plt

from dmp import DiscreteDMP
from min_jerk import min_jerk_step
from plot_tools import plot_pos_vel_acc_trajectory

def main():

  # duration of min jerk traj
  duration = 1.0
  
  # timestep (min jerk traj)
  delta_t = 0.001
  
  # start and goal of the movement (1-dim)
  start = 0.1
  goal = 0.9


  # generate min jerk traj
  traj = []
  # inital values (could be start)
  t, td, tdd = start, 0, 0
  for i in range(int(2 * duration / delta_t)):
    try:
      t, td, tdd = min_jerk_step(t, td, tdd, goal, duration - i * delta_t, delta_t)
    except:
      break
    traj.append([t, td, tdd])
  traj = np.asarray(traj)
  traj_freq = duration / delta_t

  ####### learn DMP
  dmp = DiscreteDMP()
  dmp.learn_batch(traj, traj_freq)
  
  
  ####### learn DMP
  
  # setup DMP with start and goal (same as for trajectory)
  dmp.setup(start, goal)
  
  
  # trajectory
  reproduced_traj = []
  
  # states of canonical system (plotting)
  s = []
  s_time = []
  
  for x in range(1000):
    # change goal while execution
    #if x == 500:
    #  dmp.goal = 4.0
    
    # run a single integration step
    dmp.run_step()
    
    # remember canonical system values
    s.append(dmp.s)
    s_time.append(dmp.s_time)
    
    # save reproduced trajectory
    reproduced_traj.append([dmp.x, dmp.xd, dmp.xdd])


  ####### PLOTTING
  
  fig = plt.figure('min jerk traj')
  
  plot_pos_vel_acc_trajectory(fig, traj, delta_t)
  plot_pos_vel_acc_trajectory(fig, reproduced_traj, dmp.delta_t)
  
  plt.show()

  # plot learned lwr model (compared to training data) and kernels
  fig = plt.figure("target function and kernels")
  
  ax = fig.add_subplot(231)
  ax.plot(dmp.target_function_input, dmp.target_function_ouput)#, label=r'$f(s)$')
  ax.plot(dmp.target_function_input, dmp.target_function_predicted)#, label=r'$f_predicted(s)$')
  ax.set_xlabel(r'$s$')
  ax.set_ylabel(r'$f(s)$')
  #ax.legend('upper left')
  
  ax2 = fig.add_subplot(232)
  dmp.lwr_model.plot_kernels(ax2)
  ax2.set_xlabel(r'$s$')
  ax2.set_ylabel(r'$\phi(s)$')
  
  ax3 = fig.add_subplot(233)
  ax3.plot(s_time, s)
  ax3.set_xlabel('Time [s]')
  ax3.set_ylabel(r'$s$')
  
  fig.tight_layout()
  
  plt.show()
  
  

if __name__ == '__main__':
  main()

