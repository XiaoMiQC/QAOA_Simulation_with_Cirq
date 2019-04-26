import itertools
from typing import List, Iterator, Sequence, Tuple, Dict

import cirq
import numpy as np


class QubitRing:
    """QAOA Simulation for a 1D qubit ring with control and readout errors."""
    def __init__(self, num_qubits: int, local_fields: List[int] = None,
                 couplings: List[int] = None, noise_h: float = 0.0,
                 noise_j: float = 0.0):
        r"""Specifies a 1D Ising model with nearest-neighbor couplings.

        The Hamiltonian has the form:

        H = \sum_i h_i \sigma_i^z + \sum_i J_i \sigma_i^z \sigma_{i+1}^z,

        where the index i runs from 0 to N - 1 (N is the number of qubits),
        \sigma_i^z refers to the Pauli-Z operator of qubit i, h_i is the
        local field of each qubit, J_i is the zz-coupling between qubit i and
        qubit i + 1, and we use the periodic boundary condition such that
        \sigma_{i+1}^z = \sigma_i^z.

        Args:
            num_qubits: Number of qubits in the system.
            local_fields: The coefficients of the local \sigma^z terms,
                in order. Default is 0 for all such terms.
            couplings: The coefficients of the \sigma_i^z \sigma_{i + 1}^z
                terms, in order. Default is -1 for all such terms.
            noise_h: If non-zero, a random Gaussian noise of standard
                deviation noise_h gets added to each local field coefficient.
            noise_j: If non-zero, a random Gaussian noise of standard
                deviation noise_j gets added to each coupling coefficient.
        """
        self._num_qubits = num_qubits
        self._qubit_list = cirq.LineQubit.range(num_qubits)
        self._h_noise = noise_h
        self._j_noise = noise_j

        if local_fields is not None:
            if len(local_fields) != num_qubits:
                raise Exception(
                    'Expected {} local fields, but {} specified'.format(
                        num_qubits, len(local_fields)))
            self._local_fields = local_fields.copy()
        else:
            self._local_fields = [0.0] * num_qubits

        if couplings is not None:
            if len(couplings) != num_qubits:
                raise Exception(
                    'Expected {} couplings, but {} specified'.format(
                        num_qubits, len(couplings)))
            self._couplings = couplings.copy()
        else:
            self._couplings = [-1.0] * num_qubits

        self._noiseless_h = self._local_fields.copy()
        self._noiseless_j = self._couplings.copy()

        for i in range(self._num_qubits):
            self._local_fields[i] += np.random.normal(0, self._h_noise)
            self._couplings[i] += np.random.normal(0, self._j_noise)

    @property
    def size(self) -> int:
        """Number of qubits in the system."""
        return self._num_qubits

    def local_h_fields(self, with_noise: bool = False) -> List[float]:
        """Values of the coefficients of the local z terms.

        Args:
            with_noise: Whether the returned values include the random
                Gaussian noise. Default is no.

        Returns:
            The coefficients for the local z terms.
        """
        if with_noise:
            return self._local_fields
        return self._noiseless_h.copy()

    def j_couplings(self, with_noise: bool = False) -> List[float]:
        """Values of the zz-coupling coefficients.

        Args:
            with_noise: Whether the returned values include the random
                Gaussian noise. Default is no.

        Returns:
            The coefficients for the zz-coupling terms.
        """
        if with_noise:
            return self._couplings
        return self._noiseless_j.copy()

    def randomize_h(self, num_bits: int = 2) -> None:
        """Randomize the coefficients of the local z terms.

        Args:
            num_bits: Defines the resolution of the randomization. Each local z
                coefficient is randomly chosen from np.linspace(-1.0, 1.0,
                num_bits).
        """
        for i in range(self._num_qubits):
            self._noiseless_h[i] = np.random.choice(
                np.linspace(-1.0, 1.0, num_bits))
            self._local_fields[i] = self._noiseless_h[i] + np.random.normal(
                0, self._h_noise)

    def randomize_j(self, num_bits: int = 2) -> None:
        """Randomize the coefficients of the zz coupling terms.

        Args:
            num_bits: Defines the resolution of the randomization. Each local z
                coefficient is randomly chosen from np.linspace(-1.0, 1.0,
                num_bits).
        """
        for i in range(self._num_qubits):
            self._noiseless_j[i] = np.random.choice(
                np.linspace(-1.0, 1.0, num_bits))
            self._couplings[i] = self._noiseless_j[i] + np.random.normal(
                0, self._j_noise)

    def _hadamards(self) -> Iterator[cirq.OP_TREE]:
        for i in range(self._num_qubits):
            yield cirq.H(self._qubit_list[i])

    def _rot_x_layer(self, beta: float) -> Iterator[cirq.OP_TREE]:
        for i in range(self._num_qubits):
            yield cirq.X(self._qubit_list[i]) ** beta

    def _rot_z_layer(self, gamma: float, with_noise: bool
                     ) -> Iterator[cirq.OP_TREE]:
        if with_noise:
            h_s = self._local_fields.copy()
        else:
            h_s = self._noiseless_h.copy()
        for i, h_i in enumerate(h_s):
            yield cirq.Z(self._qubit_list[i]) ** (h_i * gamma / 2.0)

    def _rot_zz_layer(self, gamma: float, with_noise: bool
                      ) -> Iterator[cirq.OP_TREE]:
        if with_noise:
            j_s = self._couplings.copy()
        else:
            j_s = self._noiseless_j.copy()
        for i, j_i in enumerate(j_s):
            yield cirq.ZZ(self._qubit_list[i],
                          self._qubit_list[(i + 1) % self._num_qubits]
                          ) ** (j_i * gamma / 2.0)

    def _qaoa_sequence(self, angle_pairs: Sequence[Tuple[float, float]],
                       with_noise: bool) -> Iterator[cirq.OP_TREE]:
        yield self._hadamards()
        for beta, gamma in angle_pairs:
            yield self._rot_z_layer(gamma, with_noise)
            yield self._rot_zz_layer(gamma, with_noise)
            yield self._rot_x_layer(beta)

    def _calc_energy(self, meas_list: Sequence[int]) -> float:
        h = np.array(self._noiseless_h)
        j = np.array(self._noiseless_j)
        pm_array = 1.0 - 2.0 * np.array(meas_list)
        tot_energy = (np.sum(pm_array * h + 1.0)) / 2.0
        pm_array_shifted = np.concatenate((pm_array[1:], [pm_array[0]]))
        tot_energy += (np.sum(pm_array * pm_array_shifted * j + 1.0)) / 2.0
        return tot_energy

    def get_all_energies(self) -> Tuple[Dict[str, float],
                                        Dict[str, List[float]], np.ndarray]:
        """Computes the energies of all possible bit strings of the noiseless
        Hamiltonian.

        Returns:
            energy_extrema: Maps 'E_max' to the maximum energy and 'E_min' to
                the minimum energy.
            indices: Maps 'max_indices' to the the location(s) of the maximum
                in the returned list of energies, and 'min_indices' to the
                location(s) of the minimum in the returned list of energies.
            all_energies: List of all possible energies, ordered according to
                the bit strings (in the big-endian representation).
        """
        bit_strings = list(itertools.product([0, 1], repeat=self._num_qubits))
        all_energies = []

        for bit_string in bit_strings:
            bit_string = list(bit_string)
            all_energies.append(self._calc_energy(bit_string))

        max_energy = np.max(all_energies)
        min_energy = np.min(all_energies)
        max_indices = np.argwhere(max_energy == all_energies)
        min_indices = np.argwhere(min_energy == all_energies)
        max_indices = [x[0] for x in max_indices]
        min_indices = [x[0] for x in min_indices]

        energy_extrema = {'E_max': max_energy, 'E_min': min_energy}
        indices = {'max_indices': max_indices, 'min_indices': min_indices}
        all_energies = np.asarray(all_energies)

        return energy_extrema, indices, all_energies

    def compute_wavefunction(self, angle_pairs: Sequence[Tuple[float, float]],
                             with_noise: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the wavefunction at the output of a QAOA circuit.

        Refer to Farhi et al., arXiv:1411.4028 (2014) for details on QAOA.

        Args:
            angle_pairs: Pairs of \beta and \gamma angles (i.e. [(\beta_1,
                \gamma_1, \beta_2, \gamma_2 ..).
            with_noise: Whether the Gaussian control noise should be added to
                the angles.

        Returns:
            state_vec: The complex amplitudes of the wavefunction, ordered
                according to the bit strings (in the big-endian notation).
            state_prob: The probability amplitudes of the wavefunction, ordered
                according to the bit strings (in the big-endian notation).
        """
        simulator = cirq.Simulator()
        circuit = cirq.Circuit()
        circuit.append(self._qaoa_sequence(angle_pairs, with_noise))
        result = simulator.simulate(circuit)
        state_vec = np.around(result.final_state, 5)
        state_prob = np.absolute(state_vec) ** 2

        return state_vec, state_prob

    def simulate_qaoa_experiment(
            self, angle_pairs: Sequence[Tuple[float, float]], num_trials: int,
            with_noise: bool = True, readout_error: float = None) -> float:
        """Simulates QAOA experiment with a fixed set of angles.

        Here the output bit strings are sampled from the probability
        distribution of the wavefunction in a number of trials, and the
        average energy of all bit strings is returned. Readout error can also
        be added, in which case each bit in the output bit strings is flipped
        with a fixed probability.

        Args:
            angle_pairs: Pairs of \beta and \gamma angles (i.e. [(\beta_1,
                \gamma_1, \beta_2, \gamma_2 ..).
            num_trials: Number of trials to take. Each trial involves
                sampling one bit string from the output probability
                distribution.
            with_noise: Whether the Gaussian control noise should be added to
                the angles.
            readout_error: A number between 0 and 1. If specified, each bit
            is flipped with the specified probability.

        Returns:
            The average energy of all bit strings from the trials.
        """
        simulator = cirq.Simulator()
        circuit = cirq.Circuit()
        circuit.append(self._qaoa_sequence(angle_pairs, with_noise))
        circuit.append(cirq.measure(*self._qubit_list, key='z'))
        sim_results = simulator.run(circuit, repetitions=num_trials)
        bit_strings = sim_results.measurements['z']
        if readout_error is not None:
            for i in range(num_trials):
                for j in range(self._num_qubits):
                    if np.random.uniform(0, 1.0) < readout_error:
                        bit_strings[i][j] = not bit_strings[i][j]
        result_energies = [self._calc_energy(bits) for bits in bit_strings]
        return float(np.mean(result_energies))
