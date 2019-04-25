import itertools

import cirq
import numpy
import pybobyqa


class QubitRing(object):

    def __init__(self, num_qubits, local_fields=None, couplings=None,
                 noise_h=0.0, noise_j=0.0):
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
            self._local_fields[i] += numpy.random.normal(0, self._h_noise)
            self._couplings[i] += numpy.random.normal(0, self._j_noise)

    @property
    def size(self):
        return self._num_qubits

    @property
    def local_h_fields(self):
        return self._local_fields

    @property
    def j_couplings(self):
        return self._couplings

    def randomize_h(self):
        for i in range(self._num_qubits):
            self._noiseless_h[i] = numpy.random.choice([0.0, 0.0])
            self._local_fields[i] = self._noiseless_h[i] + numpy.random.normal(
                0, self._h_noise)

    def randomize_j(self, num_bits=2):
        for i in range(self._num_qubits):
            self._noiseless_j[i] = numpy.random.choice(
                numpy.linspace(-1.0, 1.0, num_bits))
            self._couplings[i] = self._noiseless_j[i] + numpy.random.normal(0,
                                                                            self._j_noise)

    def _hadamards(self):
        for i in range(self._num_qubits):
            yield cirq.H()(self._qubit_list[i])

    def _rot_x_layer(self, x_half_turn):
        rot = cirq.X ** (x_half_turn * 2.0)
        for i in range(self._num_qubits):
            yield rot(self._qubit_list[i])

    def _rot_z_layer(self, z_half_turn, with_noise):
        if with_noise:
            h_s = self._local_fields.copy()
        else:
            h_s = self._noiseless_h.copy()
        for i, h_i in enumerate(h_s):
            yield cirq.ZZ ** (h_i * z_half_turn)(self._qubit_list[i])

    def _rot_zz_layer(self, zz_half_turn, with_noise):
        if with_noise:
            j_s = self._couplings.copy()
        else:
            j_s = self._noiseless_j.copy()
        for i, j_i in enumerate(j_s):
            yield cirq.ZZ ** (j_i * zz_half_turn
                              )(self._qubit_list[i],
                                self._qubit_list[(i + 1) % self._num_qubits])

    def _qaoa_sequence(self, angle_pairs, with_noise):
        yield self._hadamards()
        for x_angle, z_angle in angle_pairs:
            yield self._rot_z_layer(z_angle, with_noise)
            yield self._rot_zz_layer(z_angle, with_noise)
            yield self._rot_x_layer(x_angle)

    def print_qaoa_circuit(self, angle_pairs, with_noise=True):
        circuit = cirq.Circuit()
        circuit.append(self._qaoa_sequence(angle_pairs, with_noise))
        circuit.append(cirq.measure(*self._qubit_list, key='z'))
        print(circuit)

    def calc_energy(self, meas_list):
        h = numpy.array(self._noiseless_h)
        j = numpy.array(self._noiseless_j)
        pm_array = 1.0 - 2.0 * numpy.array(meas_list)
        tot_energy = (numpy.sum(pm_array * h + 1.0)) / 2.0
        pm_array_shifted = numpy.concatenate((pm_array[1:], [pm_array[0]]))
        tot_energy += (numpy.sum(pm_array * pm_array_shifted * j + 1.0)) / 2.0
        return tot_energy

    def get_all_energies(self):
        bit_strings = list(itertools.product([0, 1], repeat=self._num_qubits))
        all_energies = []

        for bit_string in bit_strings:
            bit_string = list(bit_string)
            all_energies.append(self.calc_energy(bit_string))

        max_energy = numpy.max(all_energies)
        min_energy = numpy.min(all_energies)
        max_indices = numpy.argwhere(max_energy == all_energies)
        min_indices = numpy.argwhere(min_energy == all_energies)
        max_indices = [x[0] for x in max_indices]
        min_indices = [x[0] for x in min_indices]

        results = {'E_max': max_energy, 'E_min': min_energy,
                   'max_indices': max_indices, 'min_indices': min_indices,
                   'energy_list': numpy.array(all_energies)}

        return results

    def simulate_qaoa(self, angle_pairs, with_noise=True):
        simulator = cirq.google.XmonSimulator()
        circuit = cirq.Circuit()
        circuit.append(self._qaoa_sequence(angle_pairs, with_noise))
        result = simulator.simulate(circuit, qubit_order=self._qubit_list)
        state_vec = numpy.around(result.final_state, 5)
        state_prob = numpy.absolute(state_vec) ** 2

        return state_vec, state_prob

    def simulate_qaoa_with_measurements(self, angle_pairs, num_trials,
                                        with_noise=True, readout_error=None):
        simulator = cirq.google.XmonSimulator()
        circuit = cirq.Circuit()
        circuit.append(self._qaoa_sequence(angle_pairs, with_noise))
        circuit.append(cirq.measure(*self._qubit_list, key='z'))
        sim_results = simulator.run(circuit, repetitions=num_trials,
                                    qubit_order=self._qubit_list)
        bit_strings = sim_results.measurements['z']
        if readout_error is not None:
            for i in range(num_trials):
                for j in range(self._num_qubits):
                    if numpy.random.uniform(0, 1.0) < readout_error:
                        bit_strings[i][j] = not bit_strings[i][j]
        result_energies = [self.calc_energy(bit_str) for bit_str in bit_strings]
        return numpy.mean(result_energies)

    def optimal_angle_solver(self, p, num_inits=50, with_noise=False):
        search_analysis = self.get_all_energies()
        e_list = search_analysis['energy_list']

        def cost_function(angles_list):
            angles_list = [(angles_list[i], angles_list[i + 1]) for i in
                           numpy.array(range(p)) * 2]
            _, state_probs = self.simulate_qaoa(angles_list,
                                                with_noise=with_noise)
            return -numpy.sum(e_list * state_probs)

        lower_bounds = numpy.array([0.0] * (2 * p))
        upper_bounds = numpy.array([1.0, 2.0] * p) * 16

        best_cost = None
        best_angles = None

        for i in range(num_inits):
            print(i)
            guess = numpy.random.uniform(0, 2, p * 2)
            guess[0::2] = guess[0::2] / 2.0
            res = pybobyqa.solve(cost_function, guess,
                                 bounds=(lower_bounds, upper_bounds),
                                 maxfun=1000)
            cost = res.f
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_angles = res.x

        best_angles = [(best_angles[i], best_angles[i + 1]) for i in
                       numpy.array(range(p)) * 2]

        return -best_cost, best_angles
