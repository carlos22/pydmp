#!/usr/bin/env python
'''
Dynamic Movement Primitive (discrete)
based on: 
'''
import numpy as np
import pylab as plt

from dmp import DiscreteDMP
from min_jerk import min_jerk_step
from plot_tools import plot_pos_vel_acc_trajectory

def main():

  ## App specific
  # temporal scaling factor
  tau = 1.0
  # timestep
  delta_t = 0.001
  # start and goal of the movement (1-dim)
  start = 0.3
  goal = 1.5
  # number of kernels (LWR)
  n_rfs = 20


  # generate min jerk traj
  traj = []
  # inital values (could be start)
  t, td, tdd = start, 0, 0
  for i in range(int(2 * tau / delta_t)):
    try:
      t, td, tdd = min_jerk_step(t, td, tdd, goal, tau - i * delta_t, delta_t)
    except:
      break
    traj.append([t, td, tdd])
  traj = np.asarray(traj)


  # learn DMP
  #dmp = DiscreteDMP()
  
  

  # reproduce DMP
  


  # PLOTTING
  

  fig = plt.figure('min jerk traj')
  
  plot_pos_vel_acc_trajectory(fig, traj, delta_t)
  
#  plot_time = np.arange(len(traj[:,0]))*delta_t
#  plt.subplot(311)
#  plt.plot(plot_time, traj[:,0], label=r'$x$')
#  plt.legend(loc='upper left')
#  plt.xlabel(r'$t$')
#  plt.ylabel(r'Position [$m$]')
#
#  plt.subplot(312)
#  plt.plot(plot_time, traj[:,1], label=r'$\dot x$')
#  plt.legend()
#  plt.xlabel(r'$t$')
#  plt.ylabel(r'Velocity [$m/s$]')
#
#  plt.subplot(313)
#  plt.plot(plot_time, traj[:,2], label=r'$\ddot x$')
#  plt.legend()
#  plt.xlabel(r'$t$')
#  plt.ylabel(r'Acceleration [$m/s^2$]')

  plt.show()

if __name__ == '__main__':
  main()

