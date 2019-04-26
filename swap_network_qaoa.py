import numpy as np
from typing import Sequence, Dict, Tuple, Set, List

import cirq


class QAOACircuit:
    """Generates QAOA Circuits for an all-to-all connected Ising model."""
    def __init__(self, qubits: Sequence[cirq.GridQubit],
                 local_fields: Dict[int, float],
                 interaction_terms: Dict[Tuple[int, int], float]):
        r"""Specifies the qubits and Hamiltonian of the problem.

        The Hamiltonian of the QAOA problem has the form:

        H_t = \sum_i c_i \sigma_i^z + \sum_{ij} c_{ij} * \sigma_i^z \sigma_j^z,

        where \sigma_i is the Pauli-Z operator on qubit i, c_i and c_{ij} are
        the coefficients for the different terms. The indices i and j must be
        between 0 and N - 1, where N is the total number of qubits. The index
        pairs ij can be between any two qubits, but i must be different from j.

        Args:
            qubits: The grid qubits to run QAOA, in order.
            local_fields: Maps qubit number to its local z term. Specifically,
                {i: c_i} adds a term c_i * \sigma_i^z to the total
                Hamiltonian.
            interaction_terms: Maps tuples (i, j) to zz interaction terms.
                Specifically, {(i, j) : c_ij} adds a term c_{ij} * \sigma_i^z
                \sigma_j^z to the total Hamiltonian.
        """
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
        r"""Builds a quantum circuit running QAOA with a fixed set of angles.

        The circuit to be built is:
        U = \prod_{k = 1}^p \exp(-1j * \beta_k * H_d) \exp(-1j * \gamma_k *
        H_t),

        where H_t is the target Hamiltonian defined at the initialization,
        H_d is the drive Hamiltonian \sum_i \sigma_i^x. The angles \beta_k
        and \gamma_k are variational parameters. In addition to the above,
        the circuit also includes a single layer of Hadamard gates on every
        qubit. For details, refer to Farhi et al., arXiv:1411.4028 (2014).

        The method allows for the circuit to be built in two ways: if
        use_swap is False, two-qubit gates are allowed between any pair of
        qubits. Otherwise two qubit gates are only allowed between qubits
        next to each other and the qubits are assumed to be placed on a line.
        In this case, the qubits are re-shuffled using a network of SWAP
        gates similar to those described in:

        Kivlichan et al., PRL 120, 110501 (2018).

        The SWAP network is modified to use SWAP and ZZ gates as opposed to
        FSIM gates. Also, rather than always perform the full network,
        the procedure does the following:

        1. For k = 0, swap qubit and do zz gates until all zz operations have
        been enacted. Record the qubit order.

        2. For k = 1, do the exact reverse of the above. The qubit order is
        reverted to the original order.

        3. For higher k, perform 1 when k is even and 2 when k is odd.

        Args:
            beta_seq: The variational angles \beta_k, in order. The length of
                beta_seq determines the circuit depth (equal to p in the
                original QAOA paper by Farhi et al.).
            gamma_seq: The variational angles \gamma_k, in order. Must have
                the same length as beta_seq.
            use_swap: Whether to use SWAP network for building the QAOA circuit.

        Returns:
            circuit: The compiled QAOA circuit.
            qubit_order: The order of the qubits at the end of the circuit.
                As an example, for a three-qubit system [2, 0, 1] means the
                third qubit is now the first qubit, the first qubit is now
                the second qubit, and the second qubit is now the third qubit.
        """
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
    """Runs a QAOA circuit a number of times and get all results.

    Args:
        sampler: The simulator or hardware to run the circuit on.
        circuit: The circuit to run.
        qubit_order: The ordering of the qubits in the final bit-string.
            This is only relevant when the SWAP network is used.
        repetitions: The number of trials to repeat the circuit.

    Returns:
        The results of the circuit stored in a 2D array. Each row represents
        the result of one single trial. The only possible values are +1 (
        qubit is in the ground state) and -1 (qubit in the excited state).
    """
    pm_mat = np.zeros((repetitions, len(qubit_order)))
    raw_data = sampler.run(circuit, repetitions=repetitions).measurements['z']
    pm_mat[:, qubit_order] = np.asarray(raw_data)
    return 1.0 - 2.0 * pm_mat


def calc_energy(local_fields: Dict[int, float],
                interaction_terms: Dict[Tuple[int, int], float],
                trial_results: np.ndarray) -> float:
    """Calculate the expectation value of the energy from a collection of
    trial results.

    Args:
        local_fields: Maps qubit number to its local z term. Specifically,
            {i: c_i} adds a term c_i * \sigma_i^z to the total Hamiltonian.
        interaction_terms: Maps tuples (i, j) to zz interaction terms.
            Specifically, {(i, j) : c_ij} adds a term c_{ij} * \sigma_i^z
            \sigma_j^z to the total Hamiltonian.
        trial_results: The trial results of measuring a QAOA circuit, stored in
            a 2D array. Each row represents the result of one single trial.
            The only allowed values are +1 (qubit is in the ground state)
            and -1 (qubit in the excited state)

    Returns:
        The expectation value of the energy of the Hamiltonian.
    """
    _, num_qubits = trial_results.shape

    h_mat = np.zeros(num_qubits)
    for key, val in local_fields.items():
        h_mat[key] = val

    j_mat = np.zeros((num_qubits, num_qubits))
    for (l_i, l_j), val in interaction_terms.items():
        j_mat[l_i, l_j] = val

    h_ave = np.mean(np.einsum('i,ji->j', h_mat, trial_results))
    j_ave = np.einsum('ij,kj->ik', j_mat, trial_results)
    j_ave = np.mean(np.einsum('ki,ik->k', trial_results, j_ave))

    return h_ave + j_ave
