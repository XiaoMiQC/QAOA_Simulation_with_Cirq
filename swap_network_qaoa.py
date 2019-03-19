import numpy as np
import time
from typing import Sequence, Dict, Tuple, Set, List

import cirq


class QAOACircuit:
    def __init__(self, qubits: Sequence[cirq.GridQubit],
                 local_fields: Dict[int, float],
                 interaction_terms: Dict[Tuple[int, int], float]):
        self._num_qubits = len(qubits)
        msg = 'Qubit labels must be smaller than {}.'.format(self._num_qubits)
        self._interaction_terms = {}

        for label in local_fields.keys():
            if label >= self._num_qubits:
                raise RuntimeError(msg)
        for (label_0, label_1), val in interaction_terms.items():
            if label_0 >= self._num_qubits or label_1 >= self._num_qubits:
                raise RuntimeError(msg)
            elif label_0 == label_1:
                raise RuntimeError('Qubit label for each interaction term '
                                   'must be different.')
            l_0, l_1 = min(label_0, label_1), max(label_0, label_1)
            self._interaction_terms[(l_0, l_1)] = val

        self._qubits = qubits
        self._local_fields = local_fields

    def _h_layer(self) -> List[cirq.Gate]:
        gate_seq = []
        for qubit in self._qubits:
            gate_seq.append(cirq.H(qubit))
        return gate_seq

    def _x_layer(self, beta: float) -> List[cirq.Gate]:
        gate_seq = []
        for qubit in self._qubits:
            gate_seq.append(cirq.X(qubit) ** beta)
        return gate_seq

    def _local_field_gates(self, gamma: float) -> List[cirq.Gate]:
        gate_seq = []
        for key, val in self._local_fields.items():
            gate_seq.append(cirq.Z(self._qubits[key]) ** (gamma * val))
        return gate_seq

    def _swap_layer_one_pass(self, gamma: float, start_idx: int,
                             qubit_order: Sequence[int],
                             couplings: Set[Tuple[int, int]]
                             ) -> List[cirq.Gate]:
        gate_seq = []
        locs = range(start_idx, self._num_qubits - 1, 2)
        for loc_0 in locs:
            loc_1 = loc_0 + 1
            l_0, l_1 = qubit_order[loc_0], qubit_order[loc_1]
            l_pair = (min(l_0, l_1), max(l_0, l_1))
            cp = 0
            if l_pair in couplings:
                cp = self._interaction_terms[l_pair] * gamma
                couplings.remove(l_pair)
            gate_seq.extend(_swap_and_zz(self._qubits[loc_0],
                                         self._qubits[loc_1], cp))
            qubit_order[loc_0], qubit_order[loc_1] = l_1, l_0
        return gate_seq

    def _z_layer_forward(self, gamma: float
                         ) -> Tuple[List[int], List[cirq.Gate]]:
        gate_seq = self._local_field_gates(gamma)
        qubit_order = list(range(self._num_qubits))
        couplings = set(self._interaction_terms)
        count = 0
        while len(couplings) > 0:
            time.sleep(0.5)
            gate_seq.extend(self._swap_layer_one_pass(gamma, count % 2,
                                                      qubit_order, couplings))
            count += 1
        return qubit_order, gate_seq

    def _z_layer_backward(self, gamma: float
                          ) -> Tuple[List[int], List[cirq.Gate]]:
        _, gate_seq = self._z_layer_forward(gamma)
        gate_seq.reverse()
        qubit_order = list(range(self._num_qubits))
        return qubit_order, gate_seq

    def _z_layer_direct(self, gamma: float) -> List[cirq.Gate]:
        gate_seq = self._local_field_gates(gamma)
        for (l_0, l_1), val in self._interaction_terms.items():
            gate_seq.append(cirq.ZZ(self._qubits[l_0],
                                    self._qubits[l_1]) ** (val * gamma))
        return gate_seq

    def build(self, beta_seq: Sequence[float], gamma_seq: Sequence[float],
              use_swap: bool = True) -> Tuple[cirq.Circuit, List[int]]:
        p = len(beta_seq)
        if p != len(gamma_seq):
            raise RuntimeError('Number of beta values must be the same as '
                               'the number of gamma values')
        circuit = cirq.Circuit.from_ops(self._h_layer())
        qubit_order = list(range(self._num_qubits))

        if use_swap:
            for step in range(p):
                if step % 2 == 0:
                    qubit_order, gate_seq = self._z_layer_forward(
                        gamma_seq[step])
                    circuit.append(gate_seq)
                else:
                    qubit_order, gate_seq = self._z_layer_backward(
                        gamma_seq[step])
                    circuit.append(gate_seq)
                circuit.append(self._x_layer(beta_seq[step]))

        else:
            for step in range(p):
                circuit.append(self._z_layer_direct(gamma_seq[step]))
                circuit.append(self._x_layer(beta_seq[step]))

        circuit.append(cirq.measure(*self._qubits, key='z'))

        return circuit, qubit_order


def _swap_and_zz(q_0: cirq.GridQubit, q_1: cirq.GridQubit,
                 zz_coeff: float) -> List[cirq.Gate]:
    gate_seq = [cirq.SWAP(q_0, q_1)]
    if zz_coeff != 0:
        gate_seq.append(cirq.ZZ(q_0, q_1) ** zz_coeff)
    return gate_seq


def measure_bits(sampler: cirq.SimulatesSamples, circuit: cirq.Circuit,
                 qubit_order: List[int], repetitions: int) -> np.ndarray:
    pm_mat = np.zeros((repetitions, len(qubit_order)))
    raw_data = sampler.run(circuit, repetitions=repetitions).measurements['z']
    pm_mat[:, qubit_order] = np.asarray(raw_data)
    return 1.0 - 2.0 * pm_mat


def calc_energy(local_fields: Dict[int, float],
                interaction_terms: Dict[Tuple[int, int], float],
                bit_mat: np.ndarray) -> float:
    _, num_qubits = bit_mat.shape

    h_mat = np.zeros(num_qubits)
    for key, val in local_fields.items():
        h_mat[key] = val

    j_mat = np.zeros((num_qubits, num_qubits))
    for (l_i, l_j), val in interaction_terms.items():
        j_mat[l_i, l_j] = val

    h_ave = np.mean(np.einsum('i,ji->j', h_mat, bit_mat))
    j_ave = np.einsum('ij,kj->ik', j_mat, bit_mat)
    j_ave = np.mean(np.einsum('ki,ik->k', bit_mat, j_ave))

    return h_ave + j_ave
