#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def Iterator(energy:float, eps, t):
    z = energy * np.eye(2) - 1e-4j*np.eye(2)   #dimension?
    alpha = t * np.eye(2)
    beta = np.conjugate(t) * np.eye(2)
    Epsilon_surf = eps * np.eye(2) 
    Epsilon_bulk = eps * np.eye(2)
    Epsilon = eps * np.eye(2)
    for i in range(70):
        aux = np.linalg.inv(z - Epsilon)
        alpha = alpha @ aux @ alpha 
        beta = beta @ aux @ beta
        Epsilon_surf = Epsilon_surf + alpha @ aux @ beta 
        Epsilon = Epsilon + alpha @ aux @ beta + beta @ aux @ alpha


    Gs = np.linalg.inv(z - Epsilon_surf) # replace with np.linalg.inv
    Gb = np.linalg.inv(z - Epsilon_bulk)

    return Gs,Gb
#%%
delta = 1
t = 0.9
onsite = 1
eps =np.array([[onsite, delta], [np.conjugate(delta), -onsite]]) 
field_parameter = np.array([[-t, 0], [0, t]])

pos_energy_vals = np.linspace(-np.pi, np.pi, 70)
neg_energy_vals = -1 * pos_energy_vals
energy_array = np.array([pos_energy_vals , neg_energy_vals])
Gs,Gb = Iterator(energy_array, eps, field_parameter)

plt.figure(figsize=(10,9))
plt.subplot(2,1,2)
plt.plot(energy_array,t* Gb.imag ,label ='bulk')
plt.grid()
plt.tight_layout()
plt.legend()


plt.subplot(2,2,2)
plt.plot(energy_array,t*Gs.imag ,label ='surf')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()





# %%
