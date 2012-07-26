'''
Created on 25.07.2012

@author: karl
'''
import pylab as plt
import numpy as np

ax1 = None
ax2 = None
ax3 = None

def plot_pos_vel_acc_trajectory(fig, traj, delta_t, labels=['$x$', '$\dot x$', '$\ddot x$']):
  global ax1, ax2, ax3
  if not ax1: ax1 = fig.add_subplot(311)
  
  traj = np.asarray(traj)
  
  plot_time = np.arange(len(traj[:, 0])) * delta_t
  
  ax1.plot(plot_time, traj[:, 0], label=labels[0])
  ax1.set_xlabel(r'$t [s]$')
  ax1.set_ylabel(r'Position [$m$]')
  ax1.legend()
  
  if not ax2: ax2 = fig.add_subplot(312)
  ax2.plot(plot_time, traj[:, 1], label=labels[1])
  ax2.set_xlabel(r'$t [s]$')
  ax2.set_ylabel(r'Velocity [$m/s$]')
  ax2.legend()
  
  if not ax3: ax3 = fig.add_subplot(313)
  ax3.plot(plot_time, traj[:, 2], label=labels[2])
  ax3.set_xlabel(r'$t [s]$')
  ax3.set_ylabel(r'Acceleration [$m/s^2$]')
  ax3.legend()
  
  fig.tight_layout()
  
