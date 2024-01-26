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
    for i in range(100):
        aux = np.linalg.inv(z - Epsilon)
        alpha = alpha @ aux @ alpha 
        beta = beta @ aux @ beta
        Epsilon_surf = Epsilon_surf + alpha @ aux @ beta 
        Epsilon = Epsilon + alpha @ aux @ beta + beta @ aux @ alpha


    Gs = np.linalg.inv(z - Epsilon_surf) # replace with np.linalg.inv
    Gb = np.linalg.inv(z - Epsilon_bulk)

    return Gs,Gb

#%%
delta = 0.9
t = 0.5
onsite = 0.2
eps =np.array([[onsite, delta], [np.conjugate(delta), -onsite]]) 
field_parameter = np.array([[-t, 0], [0, t]])
energy_array = np.linspace(-np.pi, np.pi , 200)

Gs_imag_list = []
Gb_imag_list = []

# Schleife über Energiewerte
for energy_value in energy_array:
    Gs, Gb = Iterator(energy_value, eps, field_parameter)
    Gs_imag_list.append(np.imag(Gs))
    Gb_imag_list.append(np.imag(Gb))

plt.figure(figsize=(12,8))
plt.subplot(2,1,2)
for i in range(2):
    plt.plot(energy_array, [t * (Gb_imag[i, i]) for Gb_imag in Gb_imag_list], label=f'Im of Gb[{i+1},{i+1}]')
plt.xlabel('Energy')
plt.ylabel('DOS')
plt.legend()
plt.tight_layout()
plt.grid()
plt.title('Imaginary parts of the diagonal matrix elements of Gb')

plt.subplot(2,2,2)
for i in range(2):
    plt.plot(energy_array, [t * (Gs_imag[i , i]) for Gs_imag in Gs_imag_list], label=f'Im of Gs[{i+1},{i+1}]') 
plt.xlabel('Energy')
plt.ylabel('DOS')
plt.legend()
plt.tight_layout()
plt.grid()
plt.title('Imaginary parts of the diagonal matrix elements of Gs')
plt.show()
#%%
