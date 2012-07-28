#!/usr/bin/env python
'''
Simple DMP example
'''
import numpy as np
import pylab as plt

from dmp import DiscreteDMP
from min_jerk import min_jerk_step
from plot_tools import plot_pos_vel_acc_trajectory

def main():
  
  # start and goal of the movement (1-dim)
  start = 0.3
  goal = 1.4

  ####### generate min jerk traj
  
  # duration of min jerk traj
  duration = 1.0
  
  # timestep (min jerk traj)
  delta_t = 0.001
  
  # array of values (pos,vel,acc)
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
  dmp.setup(start+.3, goal-.3)
  
  
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

  
  # create figure
  fig = plt.figure('dmp batch learn from min jerk')
  # create axes
  ax_pos = fig.add_subplot(231)
  ax_vel = fig.add_subplot(232)
  ax_acc = fig.add_subplot(233)
    
  # plot on the axes
  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), traj, delta_t, label='demonstration')
  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), reproduced_traj, dmp.delta_t, label='reproduction', linestyle='dashed')
  
  # subplot for ftarget     
  ax_ft = fig.add_subplot(234)
  ax_ft.plot(dmp.target_function_input, dmp.target_function_ouput, linewidth=2, label=r'$f_{target}(s)$')
  ax_ft.plot(dmp.target_function_input, dmp.target_function_predicted, '--', linewidth=2, label=r'$f_{predicted}(s)$') #dashes=(3, 3)
  ax_ft.set_xlabel(r'$s$')
  ax_ft.set_ylabel(r'$f(s)$')
  ax_ft.legend()
  
  # kernels (lwr)
  ax_kernel = fig.add_subplot(235)
  dmp.lwr_model.plot_kernels(ax_kernel)
  ax_kernel.set_xlabel(r'$s$')
  ax_kernel.set_ylabel(r'$\phi(s)$')
  
  # canonical system
  ax_cs = fig.add_subplot(236)
  ax_cs.plot(s_time, s)
  ax_cs.set_xlabel(r'$Time [s]$')
  ax_cs.set_ylabel(r'$s$')
  
  fig.tight_layout()
  
  plt.show()
  
  

if __name__ == '__main__':
  main()

