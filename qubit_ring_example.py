"""Example for simulating the optimal approximation ratio of a 1D Ising model
with and without control errors."""

from typing import Tuple, List, Sequence

import numpy
import pybobyqa
from matplotlib import pyplot as plt

from qubit_ring import QubitRing


def optimal_angle_solver(qubit_ring: QubitRing, p_val: int, num_inits: int = 20
                         ) -> Tuple[float, List[Tuple[float, float]]]:
    """Solves for the optimal approximation ratio at a given circuit depth (p).

    Args:
        qubit_ring: Stores information of the Ising Hamiltonian. See
            implementation of the QubitRing class.
        p_val: Circuit depth (number of (\gamma, \beta) pairs).
        num_inits: How many restarts with random initial guesses.

    Returns:
        best_ratio: The best approximation ratio at circuit depth p.
        best_angles: The optimal angles that give the best approximation ratio.
    """
    energy_extrema, indices, e_list = qubit_ring.get_all_energies()

    def cost_function(angles_list: Sequence[float]) -> float:
        angles_list = [(angles_list[k], angles_list[k + 1]) for k in
                       numpy.array(range(p_val)) * 2]
        _, state_probs = qubit_ring.compute_wavefunction(angles_list)
        return -float(numpy.sum(e_list * state_probs))

    lower_bounds = numpy.array([0.0] * (2 * p_val))
    upper_bounds = numpy.array([1.0, 2.0] * p_val) * 16

    best_cost = None
    best_angles = None

    for i in range(num_inits):
        guess = numpy.random.uniform(0, 4, p_val * 2)
        guess[0::2] = guess[0::2] / 2.0
        res = pybobyqa.solve(cost_function, guess,
                             bounds=(lower_bounds, upper_bounds),
                             maxfun=1000)
        cost = res.f
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_angles = res.x

    best_angles = [(best_angles[i], best_angles[i + 1]) for i in
                   numpy.array(range(p_val)) * 2]

    e_max, e_min = energy_extrema['E_max'], energy_extrema['E_min']
    best_ratio = (-best_cost - e_min) / (e_max - e_min)

    return best_ratio, best_angles

# Specify two qubit rings with and without control noise.
quiet_ring = QubitRing(8)
noisy_ring = QubitRing(8, noise_j=0.3)

approx_ratios = []
approx_ratios_with_noise = []

# Find and plot the maximum approximation ratios vs p (up to p = 3) for both
# cases.
p_vals = range(1, 4)
for p_val in p_vals:
    ratio, _ = optimal_angle_solver(quiet_ring, p_val)
    ratio_with_noise, _ = optimal_angle_solver(noisy_ring, p_val)
    approx_ratios.append(ratio)
    approx_ratios_with_noise.append(ratio_with_noise)
    print('Best approximation ratio at p = {} is {}, without control '
          'noise'.format(p_val, ratio))
    print('Best approximation ratio at p = {} is {}, with control '
          'noise \n'.format(p_val, ratio_with_noise))

fig = plt.figure()
plt.plot(p_vals, approx_ratios, 'ro-', figure=fig,
         label='Without control noise')
plt.plot(p_vals, approx_ratios_with_noise, 'bo-',
         figure=fig, label='With control noise')
plt.xlabel('Circuit depth, p')
plt.ylabel('Optimal approximation ratio')
plt.legend()
plt.show()
