import itertools
import cirq
import openfermioncirq
import numpy as np
import pybobyqa
from matplotlib import pyplot as plt
import scipy.optimize as optimize

def rot_x_layer(length, half_turns):
    """Yields X rotations by half_turns on a square grid of given length."""
    rot = cirq.RotXGate(half_turns=half_turns*2)
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))


def hadamards(length):
    """Yields Hadamards on all qubits"""
    for i in range(length):
        for j in range(length):
            yield cirq.HGate()(cirq.GridQubit(i, j))


def rot_z_layer(h, half_turns):
    """Yields Z rotations by half_turns conditioned on the field h."""
    for (i, j), h_ij in np.ndenumerate(h):
        yield cirq.RotZGate(half_turns=(h_ij * half_turns))(
            cirq.GridQubit(i, j))


def rot_zz_layer(jr, jc, half_turns):
    """Yields rotations about ZZ conditioned on the jr and jc fields."""
    for (i, j), jr_ij in np.ndenumerate(jr):
        yield openfermioncirq.ZZGate(half_turns=(jr_ij * half_turns))(
            cirq.GridQubit(i, j), cirq.GridQubit(i, j + 1))

    for (i, j), jc_ij in np.ndenumerate(jc):
        yield openfermioncirq.ZZGate(half_turns=(jc_ij * half_turns))(
            cirq.GridQubit(i, j), cirq.GridQubit(i + 1, j))


def one_step(h, jr, jc, x_half_turns, z_half_turns):
    length = np.size(h, 0)
    yield hadamards(length)
    yield rot_z_layer(h, z_half_turns)
    yield rot_zz_layer(jr, jc, z_half_turns)
    yield rot_x_layer(length, x_half_turns)


def one_step_nohadamard(h, jr, jc, x_half_turns, z_half_turns):
    length = np.size(h, 0)
    yield rot_z_layer(h, z_half_turns)
    yield rot_zz_layer(jr, jc, z_half_turns)
    yield rot_x_layer(length, x_half_turns)

def angles_trotter_steps(p, time, local_strength):
    step_size = 1.0/float(p)
    angles = [0]*(2*p)
    for idx in range(p):
        angles[2*idx + 1] = (step_size**2)*(idx + 0.5)/np.pi*time
        angles[2*idx] = -step_size*(1.0 - step_size*(idx + 0.5))/np.pi*time*local_strength
    return angles


def calc_energy(pm_meas, h, jr, jc):
    tot_energy = (np.sum(pm_meas * h + 1.0)) / 2.0
    for (i, j), jr_ij in np.ndenumerate(jr):
        tot_energy += (jr_ij * pm_meas[i, j] * pm_meas[i, j + 1] + 1.0) / 2.0
    for (i, j), jc_ij in np.ndenumerate(jc):
        tot_energy += (jc_ij * pm_meas[i, j] * pm_meas[i + 1, j] + 1.0) / 2.0
    return tot_energy


def energy_extrema(length, h, jr, jc):
    possible_config = list(itertools.product([0, 1], repeat = length**2))
    minimum = None
    maximum = None
    minimum_strings = []
    maximum_strings = []
    for measurements in possible_config:
        measurements = list(measurements)
        meas_list_of_lists = [measurements[i * length:(i + 1) * length] for i in
                              range(length)]
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.float)
        tot_energy = calc_energy(pm_meas, h, jr, jc)
        measurements.reverse()
        measurements = np.array(measurements)
        state = np.sum(measurements*2**np.arange(measurements.size))
        if minimum == tot_energy:
            minimum_strings.append(state)
        if maximum == tot_energy:
            maximum_strings.append(state)
        if (minimum is None) or (minimum > tot_energy):
            minimum = tot_energy
            minimum_strings = []
            minimum_strings.append(state)
        if (maximum is None) or (maximum < tot_energy):
            maximum = tot_energy
            maximum_strings = []
            maximum_strings.append(state)
    return minimum_strings, maximum_strings, minimum, maximum


def energy_list(length, h, jr, jc, results):
    measurement_collection = results.measurements['z']
    energy_collection = []
    for measurements in measurement_collection:
        meas_list_of_lists = [measurements[i * length:(i + 1) * length] for i in
                              range(length)]
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.float)
        tot_energy = calc_energy(pm_meas, h, jr, jc)
        energy_collection.append(tot_energy)
    return energy_collection


def energy_expectation(trial_list):
    return sum(trial_list) / float(len(trial_list))


def cost_function(half_turns, h, jr, jc, qubits, num_trials):
    simulator = cirq.google.XmonSimulator()
    length = np.size(h, 0)
    x_half_turns = half_turns[0]
    z_half_turns = half_turns[1]
    circuit = cirq.Circuit()
    circuit.append(one_step(h, jr, jc, x_half_turns, z_half_turns))
    circuit.append(cirq.measure(*qubits, key='z'))
    results = simulator.run(circuit, repetitions=num_trials, qubit_order=qubits)
    e_list = energy_list(length, h, jr, jc, results)
    return energy_expectation(e_list)


def cost_function_multistep(half_turns, h, jr, jc, qubits, num_trials, p):
    simulator = cirq.google.XmonSimulator()
    length = np.size(h, 0)
    circuit = cirq.Circuit()
    circuit.append(hadamards(length))
    for step in range(p):
        x_half_turns = half_turns[0 + step*2]
        z_half_turns = half_turns[1 + step*2]
        circuit.append(one_step_nohadamard(h, jr, jc, x_half_turns, z_half_turns))
    circuit.append(cirq.measure(*qubits, key='z'))
    results = simulator.run(circuit, repetitions=num_trials, qubit_order=qubits)
    e_list = energy_list(length, h, jr, jc, results)
    return energy_expectation(e_list)


def parameter_sweep(x_min, x_max, z_min, z_max, h, jr, jc, qubits, num_points=20, num_trials=1000):
    results_map = np.zeros((num_points, num_points))
    x_values = np.linspace(x_min, x_max, num_points)
    z_values = np.linspace(z_min, z_max, num_points)
    for i, x_half_turns in np.ndenumerate(x_values):
        for j, z_half_turns in np.ndenumerate(z_values):
            half_turns = [x_half_turns, z_half_turns]
            results_map[j, i] = cost_function(half_turns, h, jr, jc, qubits, num_trials)

    print(np.amin(results_map))
    plt.imshow(results_map, origin='lower', extent=[x_min, x_max, z_min, z_max],
               aspect= 0.25)
    plt.xlabel(r"$\beta / \pi$")
    plt.ylabel(r"$\gamma / \pi$")
    plt.colorbar()
    plt.title("Cost function 1")
    plt.show()

def ground_state_prob(half_turns, h, jr, jc, qubits, p):
    simulator = cirq.google.XmonSimulator()
    length = np.size(h, 0)
    minimum_strings, _, _, _ = energy_extrema(length, h, jr, jc)
    circuit = cirq.Circuit()
    circuit.append(hadamards(length))
    for step in range(p):
        x_half_turns = half_turns[0 + step*2]
        z_half_turns = half_turns[1 + step*2]
        circuit.append(one_step_nohadamard(h, jr, jc, x_half_turns, z_half_turns))
    result = simulator.simulate(circuit, qubit_order=qubits)
    state_vec = np.around(result.final_state, 5)
    ground_vec = np.array(state_vec[minimum_strings])
    return np.sum(np.absolute(ground_vec)**2)


def parameter_sweep_ground_prob(x_min, x_max, z_min, z_max, h, jr, jc, qubits, num_points=20):
    simulator = cirq.google.XmonSimulator()
    length = np.size(h, 0)
    minimum_strings, _, _, _ = energy_extrema(length, h, jr, jc)
    results_map = np.zeros((num_points, num_points))
    x_values = np.linspace(x_min, x_max, num_points)
    z_values = np.linspace(z_min, z_max, num_points)
    for i, x_half_turns in np.ndenumerate(x_values):
        for j, z_half_turns in np.ndenumerate(z_values):
            circuit = cirq.Circuit()
            circuit.append(one_step(h, jr, jc, x_half_turns, z_half_turns))
            result = simulator.simulate(circuit, qubit_order=qubits)
            state_vec = np.around(result.final_state, 5)
            ground_vec = np.array(state_vec[minimum_strings])
            results_map[j, i] = np.sum(np.absolute(ground_vec)**2)
    plt.imshow(results_map, origin='lower', extent=[x_min, x_max, z_min, z_max],
               aspect=0.25)
    plt.xlabel(r"$\beta / \pi$")
    plt.ylabel(r"$\gamma / \pi$")
    cb = plt.colorbar()
    plt.title("Ground State Probability")
    plt.show()


def initial_guess_sweep(h, jr, jc, p, qubits, num_trials = 1000):
    best_cost = 1000
    best_angles = None
    initial_guesses = list(itertools.product([[0.15, 0.3], [0.85, 0.3], [0.15, 1.7], [0.85, 1.7], [0.5, 0.3], [0.5, 1.7]], repeat=p))
    for guess in initial_guesses:
        guess = sum(guess, [])
        res = optimize.minimize(cost_function_multistep, guess,
                                args=(h, jr, jc, num_trials, p),
                                method='Nelder-Mead', tol=1e-6)
        cost = cost_function_multistep(res.x, h, jr, jc, qubits, num_trials, p)

        if cost < best_cost:
            best_cost = cost
            best_angles = res.x
        print("Current solution is {}".format(cost))
        print("Current best solution is {}".format(best_cost))

    print("Final best solution is {}".format(best_cost))

    return best_angles

def initial_guess_sweep_bobyqa(h, jr, jc, qubits, p, num_trials = 1000):
    best_cost = 1000
    best_angles = None
    initial_guesses = list(itertools.product([[0.8, 0.3], [0.8, 0.25], [0.8, 0.35], [0.9, 0.3], [0.9, 0.25], [0.9, 0.35], [0.95, 0.3], [0.95, 0.25], [0.95, 0.35], [0.85, 0.3], [0.85, 0.25], [0.85, 0.35]], repeat=p))
    upper_bound = [8]*p*2
    lower_bound = [-8]*p*2
    upper_bound[0::2] = [1.1]*p
    lower_bound[0::2] = [-0.1]*p
    bound = (np.array(lower_bound), np.array(upper_bound))
    result_gnd_prob = []
    result_cost_func = []
    for guess in initial_guesses:
        guess = np.array(sum(guess, []))
        res = pybobyqa.solve(cost_function_multistep, guess,
                                args=(h, jr, jc, qubits, num_trials, p),
                                bounds = bound, maxfun = 1000)
        cost = res.f
        ground_prob = ground_state_prob(res.x, h, jr, jc, qubits, p)

        if cost < best_cost:
            best_cost = cost
            best_angles = res.x
            best_ground_prob = ground_prob
        print("Current solution is {}".format(cost))
        print("Current best solution is {}".format(best_cost))
        print("Current ground state prob is {}".format(ground_prob))
        print("Current best ground state prob is {}".format(best_ground_prob))
        print("")
        result_gnd_prob.append(ground_prob)
        result_cost_func.append(cost)

    print("Final best solution is {}".format(best_cost))

    return best_angles, best_ground_prob, best_cost, result_gnd_prob, result_cost_func


def output_state_prob_energy(half_turns, h, jr, jc, qubits, p, showplots = False, sorted = True):
    length = np.size(h, 0)
    simulator = cirq.google.XmonSimulator()
    circuit = cirq.Circuit()
    circuit.append(hadamards(length))
    for step in range(p):
        x_half_turns = half_turns[0 + step*2]
        z_half_turns = half_turns[1 + step*2]
        circuit.append(one_step_nohadamard(h, jr, jc, x_half_turns, z_half_turns))
    result = simulator.simulate(circuit, qubit_order=qubits)
    state_vec = np.around(result.final_state, 5)
    state_prob = np.absolute(state_vec) ** 2
    l = np.size(state_prob)
    state = np.arange(l)
    energy = []

    for i in state:
        binary_num = [int(x) for x in bin(i)[2:]]
        measurements = [0] * (length**2 - len(binary_num))
        measurements = measurements + binary_num
        meas_list_of_lists = [measurements[i * length:(i + 1) * length] for i in
                              range(length)]
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.float)
        tot_energy = calc_energy(pm_meas, h, jr, jc)
        energy.append(tot_energy)

    energy = np.array(energy)
    state_prob = np.array(state_prob)
    expected_energy = np.sum(energy*state_prob)

    if sorted:
        idx = np.argsort(energy)
        energy = energy[idx]
        state_prob = state_prob[idx]
        return state_prob[0], expected_energy

    if showplots:
        plt.plot(state, state_prob, 'bo', label = "p = 30")
        plt.xlabel("State")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

    return state_prob[np.argmin(energy)], expected_energy


def increase_p(prev_best_angles, h, jr, jc, qubits, p, num_trials):
    next_guess = [0.0]*(2*p + 2)
    prev_x_rots = prev_best_angles[0::2]
    prev_z_rots = prev_best_angles[1::2]
    next_x_guess = [0.0]*(p + 1)
    next_z_guess = [0.0] * (p + 1)
    next_x_guess[-1] = prev_x_rots[-1]
    next_x_guess[0] = prev_x_rots[0]
    next_z_guess[-1] = prev_z_rots[-1]
    next_z_guess[0] = prev_z_rots[0]

    upper_bound = [8]*(p+1)*2
    lower_bound = [-8]*(p+1)*2
    upper_bound[0::2] = [1.1]*(p+1)
    lower_bound[0::2] = [-0.1]*(p+1)
    bound = (np.array(lower_bound), np.array(upper_bound))

    for i in range(1, p):
        next_x_guess[i] = prev_x_rots[i-1]*float(i)/float(p) + prev_x_rots[i]*float(p - i)/float(p)
        next_z_guess[i] = prev_z_rots[i-1] * float(i) / float(p) + prev_z_rots[
            i] * float(p - i) / float(p)
    next_guess[0::2] = next_x_guess
    next_guess[1::2] = next_z_guess
    res = pybobyqa.solve(cost_function_multistep, np.array(next_guess)*1.0,
                         args=(h, jr, jc, qubits, num_trials, p+1), bounds=bound,
                         maxfun=1000)
    cost = res.f
    ground_prob = ground_state_prob(res.x, h, jr, jc, qubits, p)
    return cost, ground_prob, res.x


def qaoa_to_p(h, jr, jc, angles_at_p3, num_increments, qubits, num_trials = 1000):
    cost_functions = [cost_function_multistep(angles_at_p3, h, jr, jc, qubits, num_trials, 3)]
    ratios = [(14.576 - 7.78)/(14.576 - 6.957)]
    ground_probs = [ground_state_prob(angles_at_p3, h, jr, jc, qubits, 3)]
    current_angles = angles_at_p3
    current_p = 3
    for _ in range(num_increments):
        new_cost, new_ground_prob, new_angles = increase_p(current_angles, h, jr, jc, qubits, current_p, num_trials)
        current_angles = new_angles
        current_p += 1
        ground_probs.append(new_ground_prob)
        cost_functions.append(new_cost)
        ratio = (14.576 - new_cost)/(14.576 - 6.957)
        ratios.append(ratio)
        print(ratio)
        print(new_ground_prob)
        print(current_p)
    steps = range(3, 4 + num_increments)
    return ground_probs, ratios, steps, new_angles




