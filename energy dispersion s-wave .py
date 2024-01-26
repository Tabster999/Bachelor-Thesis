#%% 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import constants 
#%%

def tight_binding_hamiltonian(t:float, k_value:float, chem_potential:float):
    '''
    Define a function to calculate the tight binding hamiltonian for a 1d linear chain
    
    t:  self energy 
    k_value: absolute value of k-vector
    chem_potential: chemical potential

    Returns the Hamiltonian as a 2x2 Matrix 
    '''
    H = np.array([[(k_value**2 - chem_potential), t],
                  [- t, - (k_value**2 - chem_potential)]], dtype=complex)  
    return H 


def calculate_band_structure(t, k_values, chem_potential):
    '''
    Create a function to calculate the eigenvalues(eigenenergies) of the hamiltonian.
    Parameters are the same as in the function for the hamiltonian.  
    k_values: Where to calculate the energies 

    Returns an array with eigenenergies
    '''
    energies = []

    for k_value in k_values:
        H = tight_binding_hamiltonian(t, k_value, chem_potential)
        eigenvalues = np.linalg.eigvalsh(H)
        energies.append(eigenvalues)   
    return np.array(energies)


'''arrange parameters'''
delta = 1
mu = 2
k_values = np.linspace(-np.pi, np.pi, 100)
energies = calculate_band_structure(delta, k_values, mu)


def positive_values_of_array(arr_input): 
    '''Define a functon that returns an array with only the positive entries of arr_input'''
    energies_e = arr_input[arr_input >= 0]
    return energies_e


def negative_values_of_array(arr_input):
    '''Define a functon that returns an array with only the negative entries of arr_input'''
    energies_h = arr_input[arr_input <= 0]
    return energies_h


'''set up the plot'''
plt.figure(figsize=(12, 9))
plt.grid()
plt.xlim(-np.pi, np.pi)
plt.ylim(-10, 10)
plt.xlabel('k')
plt.ylabel('Energy')
plt.title('Energy dispersion of s-wave superconductor')
plt.plot(k_values, positive_values_of_array(energies), label='Electorns', c='Blue')
plt.plot(k_values, negative_values_of_array(energies), label='Holes', c='Red')
plt.legend()
plt.show()


H_k0 = np.array([[mu**2, delta], [delta,- mu**2]], dtype=complex)
eigenvalues_k0 = np.linalg.eigvals(H_k0)
print(eigenvalues_k0)
#%%

