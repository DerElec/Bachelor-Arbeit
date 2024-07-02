import numpy as np

def generate_random_density_matrix(n):
    # create random complex matrix
    random_matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    
    # make the matrix Hermitian
    hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
    
    # compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
    
    # ensure positive eigenvalues
    positive_eigenvalues = np.abs(eigenvalues)
    
    # create new density matrix with positive eigenvalues
    density_matrix = eigenvectors @ np.diag(positive_eigenvalues) @ eigenvectors.conj().T
    
    # normalize trace to one 
    density_matrix /= np.trace(density_matrix)
    
    return density_matrix

# example 3x3 
density_matrix = generate_random_density_matrix(3)

# print the density matrix
print("Zufällige Dichtematrix:")
print(density_matrix)

# verification of density matrix properties
print("\nHermitesch:", np.allclose(density_matrix, density_matrix.conj().T))
print("Positiv semidefinit:", np.all(np.linalg.eigvals(density_matrix) >= 0))
print("Spur gleich 1:", np.isclose(np.trace(density_matrix), 1))

# formatted output
for i in range(density_matrix.shape[0]):
    for j in range(density_matrix.shape[1]):
        value = density_matrix[i, j]
        formatted_value = f"{value.real:.17f}+{value.imag:.17f}j" if value.imag >= 0 else f"{value.real:.17f}{value.imag:.17f}j"
        print(f"psi{i}{j} = {formatted_value}")

