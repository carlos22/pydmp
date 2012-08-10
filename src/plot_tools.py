'''
Created on 25.07.2012

@author: karl
'''
import numpy as np


def plot_pos_vel_acc_trajectory(axes, traj, delta_t, label='', linestyle='-',linewidth=2, loc='lower right'):
  ax1, ax2, ax3 = axes
  
  #if not ax1: ax1 = fig.add_subplot(311)
  
  traj = np.asarray(traj)
  time_label = r'Time [s]'
  
  plot_time = np.arange(len(traj[:, 0])) * delta_t
  
  ax1.plot(plot_time, traj[:, 0], label=label, linewidth=linewidth, linestyle=linestyle)
  ax1.set_xlabel(time_label)
  ax1.set_ylabel(r'Position [$m$]')
  ax1.legend(loc=loc)
  
  #if not ax2: ax2 = fig.add_subplot(312)
  ax2.plot(plot_time, traj[:, 1], linewidth=linewidth, linestyle=linestyle)
  ax2.set_xlabel(time_label)
  ax2.set_ylabel(r'Velocity [$m/s$]')
  #ax2.legend()
  
  #if not ax3: ax3 = fig.add_subplot(313)
  ax3.plot(plot_time, traj[:, 2], linewidth=linewidth, linestyle=linestyle)
  ax3.set_xlabel(time_label)
  ax3.set_ylabel(r'Acceleration [$m/s^2$]')
  #ax3.legend()
  
  #fig.tight_layout()
  
