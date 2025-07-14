
import numpy as np

def smooth_growth(t):
  if t<.08:
    stellar_mass = 0
  elif t<.1:
    stellar_mass = 1e2 + (9e2/.02)*(t - 0.08)
  elif t<.15:
    stellar_mass = 1e3 + ((1.5e4 - 1e3)/.05)*(t - 0.1)
  elif t<.17:
    stellar_mass = 1.5e4 + ((1e5 - 1.5e4)/.02)*(t - 0.15)
  elif t<.2:
    stellar_mass = 1e5 + (0.5e5/0.03)*(t-0.17)
  elif t<.25:
    stellar_mass = 1.5e5 + (1.5e5)/0.05 *(t-.2)
  elif t<= 0.3:
    stellar_mass = 3e5 + (1e5)/0.05 *(t-.25)
  elif t< 0.4:
    stellar_mass = 4e5 + (6e5)/0.1 *(t-.3)
  else:
    stellar_mass = 1e6 + (2.89e6)/0.6 * (t-.4)
  return stellar_mass

def smooth2(t):
  if t<.08:
    stellar_mass = 0
  elif t<.1:
    stellar_mass = 1e3 + (5e2/.02)*(t - 0.08)
  elif t<.135:
    stellar_mass = 1.5e3 + ((1e4 - 1.5e3)/.035)*(t - 0.1)
  elif t<.15:
    stellar_mass = 1e4 + ((1e5 - 1e4)/.015)*(t - 0.135)
  elif t<.18:
      stellar_mass = 1e5 + ((2e5 - 1e5)/.03)*(t - 0.15)
  elif t<.2:
    stellar_mass = 2e5 + (3e5 - 2e5)/0.02*(t-0.18)
  elif t<.25:
    stellar_mass = 3e5 + (1.5e5)/0.05 *(t-.2)
  elif t<= 0.3:
    stellar_mass = 4.5e5 + (.5e5)/0.05 *(t-.25)
  elif t< 0.4:
    stellar_mass = 5e5 + (5e5)/0.1 *(t-.3)
  elif t<0.9:
    stellar_mass = 1e6 + (1.5e6/.5) * (t-.4)
  else:
    stellar_mass = 2.5e6 + (3.89e6-2.5e6)/0.1 * (t-.9)
  return stellar_mass

def get_points(npts, model):
    t = np.arange(0.054,1,1/npts)
    pts =  list(map(model, t))
    return t, pts




def b5(t):
    burst_times = [.092,.160, .265,.440,.7]
    burst_mass = 2.69e6/5
    m_max = 2.5e7 + burst_mass
    if t < burst_times[0]:
        m = m_max/burst_times[0] * t
    elif t == burst_times[0]:
        m = burst_mass
    elif t < burst_times[1]:
        m = burst_mass + (m_max/(burst_times[1] - burst_times[0]))*(t-burst_times[0])
    elif t < burst_times[2]:
        m = 2*burst_mass + (m_max/(burst_times[2] - burst_times[1]))*(t-burst_times[1])
    elif t < burst_times[3]:
        m = 3*burst_mass + (m_max/(burst_times[3] - burst_times[2]))*(t-burst_times[2])
    elif t < burst_times[4]:
        m = 4*burst_mass + (m_max/(burst_times[4] - burst_times[3]))*(t-burst_times[3])
    else:
        m = 5*burst_mass
    return m/0.6909
