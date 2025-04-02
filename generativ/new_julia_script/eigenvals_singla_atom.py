import numpy as np

# English comments: This function returns the 3x3 matrix |i><j| 
# for a three-level system with states 0,1,2.
def m_ij(i, j):
    mat = np.zeros((3, 3), dtype=complex)
    mat[i, j] = 1.0
    return mat

# Create a dictionary of all m_{ij}
m_dict = {}
for i in range(3):
    for j in range(3):
        m_dict[(i, j)] = m_ij(i, j)

# Print them
for (i, j), mat in m_dict.items():
    print(f"m_({i}{j}) =\n{mat}\n")
