import numpy as np
import matplotlib.pyplot as plt

# Initial Assumptions and their assumed values:
# Time frame of t=0 to max time of 20s
tmin, tmax= 0.0, 1.0
h= 0.005 #step value

# Vales of the rate constants
k1=1.324 #Forward rate
k2=1.627 #Backward rate
r_permeation = 0.01  # Rate at which H2 gas is permeated out of the reactor
n_steps = (tmax-tmin)/h
CO_i , H2O_i , CO2_i , H2_i= 5.0, 5.0 , 0.0 , 0.0 #initial concentrations
r=np.array([CO_i , H2O_i , CO2_i , H2_i], float)
t=np.arange(tmin,tmax,h)

# Solving ODE's simultaneoulsy using the Runge-Kutta method
def system_of_ODEs(t, y):
    # y[0] corresponds to CO, y[1] to H2O, y[2] to CO2, y[3] to H2
    dydt = [f1(t, y), f2(t, y), f3(t, y), f4(t, y)]
    return dydt

# Functions for the rate of change for each component
f1 = lambda t, y: -k1 * y[0] * y[1] + k2 * y[2] * y[3]
f2 = lambda t, y: -k1 * y[0] * y[1] + k2 * y[2] * y[3]
f3 = lambda t, y: k1 * y[0] * y[1] - k2 * y[2] * y[3]
f4 = lambda t, y: k1 * y[0] * y[1] - k2 * y[2] * y[3] - r_permeation


# Runge-Kutta 4 method for solving first order diffrential equations
def rk4(system, y0, t):
    n = len(y0)
    h = t[1] - t[0]
    y = np.zeros((len(t), n))
    y[0] = y0 #initial value

    for i in range(len(t) - 1):
        k1 = h * np.array(system(t[i], y[i]))
        k2 = h * np.array(system(t[i] + h / 2, y[i] + k1 / 2))
        k3 = h * np.array(system(t[i] + h / 2, y[i] + k2 / 2))
        k4 = h * np.array(system(t[i] + h, y[i] + k3))

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

# Storing the values at all amrked points within the time frame
solution = rk4(system_of_ODEs, r, t)

# Separating the concentration profiles
CO_profile = solution[:, 0]
H2O_profile = solution[:, 1]
CO2_profile = solution[:, 2]
H2_profile = solution[:, 3]

# Plotting the concentration profiles
fig, ax1 = plt.subplots(2,2, figsize=(6.8,6.9))
ax1[0, 0].plot(t, CO_profile)
ax1[0, 0].set_title('CO')
ax1[0, 1].plot(t, H2O_profile, 'tab:orange')
ax1[0, 1].set_title('H2O')
ax1[1, 0].plot(t, CO2_profile, 'tab:green')
ax1[1, 0].set_title('CO2')
ax1[1, 1].plot(t, H2_profile, 'tab:red')
ax1[1, 1].set_title('H2')
fig.supxlabel("Time")
fig.supylabel("Concentration")
#plt.legend(['r_permeation=0.01'], loc='upper center')
plt.show()
#print(CO2_profile)