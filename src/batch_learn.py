#!/usr/bin/env python
'''
DMP Adaption
'''
import pylab as plt
import numpy as np
import json
from dmp import DiscreteDMP

from lr import LR

def main():
  
  # duration
  duration = 1.0
  
  # adaption offset
  adapt_offset = +0.02
  
  # time steps
  delta_t = 0.001
  
  # load position trajectory
  traj_pos = json.load(open("traj_full.json", 'r'))["x"]#[2000:8000][::6] #[5000:8000][::3]
  
  # rest start and goal position out of trajectory
  start = traj_pos[0]
  goal = traj_pos[-1]
    
  traj_freq = int(1.0 / delta_t)
  
  # calc it here for easier drawing
  traj = DiscreteDMP.compute_derivatives(traj_pos, traj_freq)

  ####### learn DMP
   
  dmp = DiscreteDMP(True, reg_model=LR(activation=0.1, exponentially_spaced=True, n_rfs=20))
  #dmp.use_ft = True
  dmp.learn_batch(traj, traj_freq)
  
  
  dmp_adapt = DiscreteDMP(True, reg_model=dmp.lwr_model) #copy.deepcopy(dmp.lwr_model))
  dmp_adapt._is_learned = True  
  
  ####### learn DMP
  
  # setup DMP with start and goal
  dmp.delta_t = delta_t
  dmp.setup(start, goal, duration)
  
  dmp_adapt.delta_t = delta_t  
  dmp_adapt.setup(start, goal + adapt_offset, duration)
  
  
  # trajectory
  traj_reproduced = []
  traj_adapted = []
  
  # states of canonical system (plotting)
  s = []
  s_time = []
  
  # run steps (for each point of the sample trajectory)
  for x in xrange(int(dmp.tau / dmp.delta_t)):
    # change goal while execution
    #if x == 500:
    #  dmp.goal = 4.0
    
    # run a single integration step
    dmp.run_step()
    dmp_adapt.run_step()
    
    # remember canonical system values
    s.append(dmp.s)
    s_time.append(dmp.s_time)
    
    # save reproduced trajectory
    traj_reproduced.append([dmp.x, dmp.xd, dmp.xdd])
    traj_adapted.append([dmp_adapt.x, dmp_adapt.xd, dmp_adapt.xdd])
  


  ####### PLOTTING

  
  # create figure
  fig = plt.figure('dmp', figsize=(6, 4))
  # create axes
  ax_pos = fig.add_subplot(111)
    
    
  # plot on the axes
  #ax_pos.plot(np.arange(0,1,0.001), traj_pos)
  plot_time = np.arange(len(traj)) * delta_t
  
  ax_pos.plot(plot_time, np.asarray(traj)[:, 0], label='demonstration')
  ax_pos.plot(plot_time, np.asarray(traj_reproduced)[:, 0], label='reproduction', linestyle='dashed')
  ax_pos.plot(plot_time, np.asarray(traj_adapted)[:, 0], label='adapted (%+0.2f)' % adapt_offset)
  
  ax_pos.legend(loc='lower left')
  
  
#  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), traj, dmp.delta_t, label='demonstration')
#  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), traj_reproduced, dmp.delta_t, label='reproduction', linestyle='dashed')
#  plot_pos_vel_acc_trajectory((ax_pos, ax_vel, ax_acc), traj_adapted, dmp.delta_t, label='adapted (+0.02)')
  
  fig.tight_layout()
  
  plt.show()
#  
  fig2 = plt.figure('lwr', figsize=(6, 4))
#  
  ax_ft = fig2.add_subplot(211)
  
  ax_ft.plot(dmp.target_function_input, dmp.target_function_ouput, linewidth=2, label=r'$f_{target}(s)$')
  ax_ft.plot(dmp.target_function_input, dmp.target_function_predicted, '--', linewidth=2, label=r'$f_{predicted}(s)$') #dashes=(3, 3)
 
  plt.show()
  
  
#  ax2 = fig2.add_subplot(212)
#  
#  dmp.lwr_model.plot_kernels(ax1)
#  
#  #ax2.plot(plot_time, np.asarray(traj)[:, 0], label='reproduction')
#  dmp.lwr_model.plot_linear_models(ax2)
#  print dmp.lwr_model.get_thetas()
# 
  
  
  
#  fig = plt.figure('ftarget', figsize=(14, 4.6))
#  
#  # plot  ftarget (real and predicted)
#  ax_ft = fig.add_subplot(131)
#  ax_ft.plot(dmp.target_function_input, dmp.target_function_ouput, linewidth=2, label=r'$f_{target}(s)$')
#  ax_ft.plot(dmp.target_function_input, dmp.target_function_predicted, '--', linewidth=2, label=r'$f_{predicted}(s)$') #dashes=(3, 3)
#  ax_ft.set_xlabel(r'$s$')
#  ax_ft.set_ylabel(r'$f(s)$')
#  ax_ft.legend()
#  
#  # kernels (lwr)
#  ax_kernel = fig.add_subplot(132)
#  dmp.lwr_model.plot_kernels(ax_kernel)
#  ax_kernel.set_xlabel(r'$s$')
#  ax_kernel.set_ylabel(r'$\psi(s)$')
#  
#  # canonical system
##  ax_cs = fig.add_subplot(236)
##  ax_cs.plot(s_time, s)
##  ax_cs.set_xlabel('Time [s]')
##  ax_cs.set_ylabel(r'$s$')
#
#
#  # weights of kernels (w)
#  ax_w = fig.add_subplot(133)
#  lwr_wights = dmp.lwr_model.get_thetas()
#  ax_w.bar(range(len(lwr_wights)), lwr_wights, width=0.3)
#  ax_w.set_xlabel('$w_i$')
#  ax_w.set_ylabel('')
#
#
#  fig.tight_layout()
#  
#  plt.show()  
  

if __name__ == '__main__':
  main()

