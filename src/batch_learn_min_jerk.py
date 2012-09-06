#!/usr/bin/env python
'''
Simple DMP example (plotting)
'''
import pylab as plt
import math

from dmp import DiscreteDMP
from min_jerk import min_jerk_traj
from plot_tools import plot_pos_vel_acc_trajectory
from lwr import LWR
import numpy as np

def main():
  
  # start and goal of the movement (1-dim)
  start = 0.5
  goal = 1.0

  ####### generate a trajectory (minimum jerk)
  
  # duration
  duration = 1.0
  
  # time steps
  delta_t = 0.001
  
  # trajectory is a list of 3-tuples with (pos,vel,acc)
  traj = min_jerk_traj(start, goal, 1.0, delta_t)
  traj_freq = int(1.0 / delta_t)

  ####### learn DMP
  
  dmp = DiscreteDMP(False, LWR(activation=0.3, exponentially_spaced=False, n_rfs=8, use_offset=True))
  dmp.learn_batch(traj, traj_freq)
  
  
  ####### learn DMP
  
  # setup DMP with start and goal
  dmp.setup(start+0.4, goal+0.2, duration)
  
  # trajectory
  reproduced_traj = []
  
  # states of canonical system (plotting)
  s = []
  s_time = []
  
  # run steps (for each point of the sample trajectory)
  for _ in xrange(int(dmp.tau / dmp.delta_t)):
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

  fig_pos = plt.figure('dmp batch learn from min jerk', figsize=(7, 5))
  ax_pos2 = fig_pos.add_subplot(111)
  plot_time = np.arange(len(traj)) * delta_t


  ax_pos2.plot(plot_time, np.asarray(traj)[:, 0], label='demonstration')
  ax_pos2.plot(plot_time, np.asarray(reproduced_traj)[:, 0], label='adapted reproduction', linestyle='dashed')
  
  ax_pos2.legend(loc='upper left')

  plt.show()
  
  # create figure
  fig = plt.figure('dmp batch learn from min jerk', figsize=(16, 4.6))
  # create axes
  ax_pos = fig.add_subplot(131)
  ax_vel = fig.add_subplot(132)
  ax_acc = fig.add_subplot(133)
  
    
  # plot on the axes
  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), traj, delta_t, label='demonstration')
  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), reproduced_traj, dmp.delta_t, label='reproduction', linestyle='dashed')
  
  fig.tight_layout()
  
  plt.show()
  
  fig = plt.figure('ftarget', figsize=(16, 4.6))
  
  # plot  ftarget (real and predicted)
  ax_ft = fig.add_subplot(131)
  ax_ft.plot(dmp.target_function_input, dmp.target_function_ouput, linewidth=2, label=r'$f_{target}(s)$')
  ax_ft.plot(dmp.target_function_input, dmp.target_function_predicted, '--', linewidth=2, label=r'$f_{predicted}(s)$') #dashes=(3, 3)
  ax_ft.set_xlabel(r'$s$')
  ax_ft.set_ylabel(r'$f(s)$')
  ax_ft.legend()
  
  # kernels (lwr)
  ax_kernel = fig.add_subplot(132)
  dmp.lwr_model.plot_kernels(ax_kernel)
  ax_kernel.set_xlabel(r'$s$')
  ax_kernel.set_ylabel(r'$\psi(s)$')
  
  # canonical system
#  ax_cs = fig.add_subplot(236)
#  ax_cs.plot(s_time, s)
#  ax_cs.set_xlabel('Time [s]')
#  ax_cs.set_ylabel(r'$s$')


  # weights of kernels (w)
  ax_w = fig.add_subplot(133)
  ax_w.set_xlabel(r'$s$')
  ax_w.set_ylabel(r'$f(s)$')
  dmp.lwr_model.plot_linear_models(ax_w)
#  lwr_wights = dmp.lwr_model.get_thetas()
#  ax_w.bar(range(len(lwr_wights)), lwr_wights, width=0.3)
#  ax_w.set_xlabel('$w_i$')
#  ax_w.set_ylabel('')


  fig.tight_layout()
  
  plt.show()  
  

if __name__ == '__main__':
  main()

