import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import linalg


def random_ising_model(num_rows, num_cols):
    # transverse field terms
    h = ((np.random.uniform(0,1, size = (num_rows, num_cols)) * 2) - 1.0) * 1.0
    # links within a row
    jr = ((np.random.uniform(0,1, size = (num_rows, num_cols - 1)) * 2) - 1.0) * 1.0
    # links within a column
    jc = ((np.random.uniform(0,1, size = (num_rows - 1, num_cols)) * 2) - 1.0) * 1.0
    return (h, jr, jc)


def tens_prod(m1, m2):
    if m1.size == 0:
        return m2
    elif m2.size == 0:
        return m1
    else:
        return np.kron(m1, m2)


def eigensystem(matrix, num_eigens):

    rng = np.random.RandomState(seed=11)  # Ensures deterministic output.
    initial_vec = rng.rand(matrix.shape[0])
    eigenvalues, vectors = linalg.eigsh(
        matrix, k=num_eigens, which='SA', v0=initial_vec)
    vectors = np.asarray(vectors)

    return eigenvalues, vectors


def get_ham_matrix(h, jr, jc, t):
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    num_rows = list(h.shape)[0]
    num_cols = list(h.shape)[1]
    ham_mat = np.zeros((2**(num_rows*num_cols),2**(num_rows*num_cols)))

    for i in range(num_rows):
        for j in range(num_cols):
            index = j + i*num_cols
            local_mat = (1.0-t)*sigma_x + t*h[i,j]*sigma_z
            mat_1 = np.eye(2 ** index)
            mat_2 = np.eye(2 ** (num_rows * num_cols - index - 1))
            ham_mat += tens_prod(tens_prod(mat_1, local_mat), mat_2)

    for (i, j), jr_ij in np.ndenumerate(jr):
        index_1 = j + i*num_cols
        index_2 = j + i*num_cols + 1
        mat_1 = np.eye(2 ** index_1)
        mat_2 = np.eye(2 ** (index_2 - index_1 - 1))
        mat_3 = np.eye(2 ** (num_rows * num_cols - index_2 - 1))
        ham_mat += t*jr_ij*tens_prod(tens_prod(tens_prod(tens_prod(mat_1, sigma_z), mat_2),sigma_z),mat_3)

    for (i, j), jc_ij in np.ndenumerate(jc):
        index_1 = j + i*num_cols
        index_2 = j + (i+1)*num_cols
        mat_1 = np.eye(2 ** index_1)
        mat_2 = np.eye(2 ** (index_2 - index_1 - 1))
        mat_3 = np.eye(2 ** (num_rows * num_cols - index_2 - 1))
        ham_mat += t*jc_ij*tens_prod(tens_prod(tens_prod(tens_prod(mat_1, sigma_z), mat_2),sigma_z),mat_3)
    return sparse.csr_matrix(ham_mat)

def get_energies(h, jr, jc, t, n):
    ham_mat = get_ham_matrix(h, jr, jc, t)
    energies, _ = eigensystem(ham_mat, n)
    return energies

def plot_annealing_trajectory(h, jr, jc, n, num_points = 10):
    times = np.linspace(0, 1, num_points)
    energies = np.zeros((n, num_points))
    for idx, time in np.ndenumerate(times):
        eigen_energies = get_energies(h, jr, jc, time, n)
        for i in range(n):
            energies[i, idx] = eigen_energies[i]
    plt.figure()
    for energy in energies:
        plt.plot(times, energy)
    plt.show()
