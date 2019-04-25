import numpy as np
import cirq
from matplotlib import pyplot as plt

from swap_network_qaoa import QAOACircuit, measure_bits, calc_energy

simulator = cirq.Simulator()
qubits = [cirq.GridQubit(0, i) for i in range(9)]

# Specify some random local_field terms in the Hamiltonian
local_fields = {1: 1.2, 2: -0.4, 4: -1.1, 5: 0.3, 7: 1.5, 8: -0.9}

# Specify some random interaction terms in the Hamiltonian
int_terms = {(0, 4): 1.8,
             (1, 5): -0.7,
             (2, 4): 1.1,
             (2, 6): -1.5,
             (1, 6): -1.1,
             (3, 1): 1.0,
             (2, 8): -0.1,
             (5, 8): -0.3,
             (7, 4): 1.2}

qaoa_circ = QAOACircuit(qubits, local_fields, int_terms)

# Pick some random variational angles.
betas = [0.2, 0.3, 0.7]
gammas = [0.5, 0.6, 0.4]

# Print out the QAOA circuit with and without the SWAP network.
circuit_0, _ = qaoa_circ.build(betas, gammas, use_swap=True)
circuit_1, _ = qaoa_circ.build(betas, gammas, use_swap=False)
print(circuit_0)
print(circuit_1)

# Now sweep a variational parameter \gamma_2 and see if the cost functions (
# energy expectation values) of the circuits without SWAP network agree with
# those with SWAP network.
gamma_sweep = np.linspace(0, 1.0, 20)
energy_0 = []
energy_1 = []

for gamma in gamma_sweep:
    gammas = [0.5, 0.6, gamma]
    circuit_0, q_order_0 = qaoa_circ.build(betas, gammas, use_swap=True)
    circuit_1, q_order_1 = qaoa_circ.build(betas, gammas, use_swap=False)
    mat_0 = measure_bits(simulator, circuit_0, q_order_0, 50000)
    mat_1 = measure_bits(simulator, circuit_1, q_order_1, 50000)
    energy_0.append(calc_energy(local_fields, int_terms, mat_0))
    energy_1.append(calc_energy(local_fields, int_terms, mat_1))

fig = plt.figure()
plt.plot(gamma_sweep, energy_0, figure=fig, label='Swap Network')
plt.plot(gamma_sweep, energy_1, figure=fig, label='Direct')
plt.xlabel(r'$\gamma_3$')
plt.ylabel(r'$<E>$')
plt.legend()
plt.show()


