import itertools

import numpy
import pybobyqa
from matplotlib import pyplot
import scipy.optimize as optimize

from ring_of_disagrees import QubitRing


def parameter_sweep_one_step(x_min, x_max, z_min, z_max, ring, num_points=20,
                             noise=False):
    energy_analysis = ring.get_all_energies()
    E_max = energy_analysis['E_max']
    max_indices = energy_analysis['max_indices']
    energy_list = energy_analysis['energy_list']

    results_map = numpy.zeros((num_points, num_points))
    results_map_1 = numpy.zeros((num_points, num_points))
    x_values = numpy.linspace(x_min, x_max, num_points)
    z_values = numpy.linspace(z_min, z_max, num_points)

    for i, x_half_turns in numpy.ndenumerate(x_values):
        for j, z_half_turns in numpy.ndenumerate(z_values):
            print(i,j)
            _, state_probs = ring.simulate_qaoa([(x_half_turns, z_half_turns)], with_noise=noise)
            max_vec = numpy.array(state_probs[max_indices])
            results_map[j, i] = numpy.sum(max_vec)
            energy_mean = numpy.sum(energy_list * state_probs)
            results_map_1[j, i] = (energy_mean - ring.size / 2.0) / (
                    E_max - ring.size / 2.0)

    ratio = (x_max - x_min) / (z_max - z_min)

    fig_1 = pyplot.figure()
    pyplot.imshow(results_map, origin='lower',
                  extent=[x_min, x_max, z_min, z_max], aspect=ratio,
                  figure=fig_1)
    pyplot.xlabel(r"$\beta / \pi$")
    pyplot.ylabel(r"$\gamma / \pi$")
    cb = pyplot.colorbar()
    pyplot.title("Max Energy State(s) Probability (p = 1)")

    fig_2 = pyplot.figure()
    pyplot.imshow(results_map_1, origin='lower',
                  extent=[x_min, x_max, z_min, z_max], aspect=ratio,
                  figure=fig_2)
    pyplot.xlabel(r"$\beta / \pi$")
    pyplot.ylabel(r"$\gamma / \pi$")
    cb = pyplot.colorbar()
    pyplot.title("Approximation Ratio (p = 1)")

    pyplot.show()


def one_d_sweep_one_step(z_min, z_max, x_val, readout_errors, num_qubits,
                         num_points=20):
    ring = QubitRing(num_qubits)
    # energy_analysis = ring.get_all_energies()
    # E_max = energy_analysis['E_max']
    E_max = ring.size * 1.5
    # energy_list = energy_analysis['energy_list']

    results_map = numpy.zeros(num_points)
    results_list = []
    z_values = numpy.linspace(z_min, z_max, num_points)

    for readout_error in readout_errors:
        for i, z_half_turns in numpy.ndenumerate(z_values):
            print(i)
            energy_mean_1 = ring.simulate_qaoa_with_measurements(
                [(x_val, z_half_turns)], 5000, with_noise=False,
                readout_error=readout_error)
            results_map[i] = (energy_mean_1 - ring.size / 2.0) / (
                    E_max - ring.size / 2.0)
        results_list.append(results_map.copy())

    fig_1 = pyplot.figure()
    for i, result in enumerate(results_list):
        pyplot.plot(z_values, result, figure=fig_1,
                    label='Readout Error = {}'.format(readout_errors[i]))
    pyplot.xlabel(r"$\gamma / \pi$")
    pyplot.ylabel('Approximation ratio (p = 1), {} qubits'.format(ring.size))
    pyplot.legend()


def single_angle_qaoa(x_val, z_val, ring):
    energy_analysis = ring.get_all_energies()
    energy_list = energy_analysis['energy_list']
    # E_max = energy_analysis['E_max']
    _, state_probs = ring.simulate_qaoa([(x_val, z_val)], with_noise=True)
    energy_mean = numpy.sum(energy_list * state_probs)
    return (energy_mean - ring.size / 2.0) / (ring.size * 1.5 - ring.size / 2.0)


def error_approx_ratio(noise_level, num_qubits, repeats=500):
    error_list = []
    for _ in range(repeats):
        j_couplings = list(numpy.random.choice([-1.0, 1.0], num_qubits))
        qubit_ring = QubitRing(num_qubits, couplings=j_couplings)
        qubit_ring_noise = QubitRing(num_qubits, couplings=j_couplings,
                                     noise_h=noise_level, noise_j=noise_level)
        approx_rat_0 = single_angle_qaoa(0.125, 0.25, qubit_ring,
                                         qubit_ring_noise)
        approx_rat_1 = single_angle_qaoa(0.125, 0.25, qubit_ring, qubit_ring)
        error_list.append(approx_rat_0 - approx_rat_1)
    return numpy.sqrt(numpy.mean(numpy.array(error_list) ** 2))
#
#
# ring = QubitRing(9)
# ring.print_qaoa_circuit([(0.125, 0.25)])
#
# numpy.random.seed(12)
# num_qubits_0 = 10
# num_qubits_1 = 16
# num_qubits_2 = 22
# ring_noise_0 = QubitRing(num_qubits_0, noise_h=0.0, noise_j=0.0)
# ring_noise_1 = QubitRing(num_qubits_1, noise_h=0.0, noise_j=0.0)
# ring_noise_2 = QubitRing(num_qubits_2, noise_h=0.0, noise_j=0.0)
# ring_noise_0.randomize_j(num_bits = 2001)
# ring_noise_1.randomize_j(num_bits = 2001)
# ring_noise_2.randomize_j(num_bits = 2001)
# e_0 = ring_noise_0.get_all_energies()
# e_1 = ring_noise_1.get_all_energies()
# e_2 = ring_noise_2.get_all_energies()
# e_max_0 = e_0['E_max']
# e_max_1 = e_1['E_max']
# e_max_2 = e_2['E_max']
# e_min_0 = e_0['E_min']
# e_min_1 = e_1['E_min']
# e_min_2 = e_2['E_min']
# readout_errs = numpy.linspace(0, 0.1, 10)
# approx_rat_0 = []
# approx_rat_1 = []
# approx_rat_2 = []
# for err in readout_errs:
#     energy = ring_noise_0.simulate_qaoa_with_measurements([(0.125, 0.25)], 40000, with_noise=False, readout_error = err)
#     ratio_0 = (energy - e_min_0) / (e_max_0 - e_min_0)
#     approx_rat_0.append(ratio_0)
#     energy = ring_noise_1.simulate_qaoa_with_measurements([(0.125, 0.25)],
#                                                           40000,
#                                                           with_noise=False,
#                                                           readout_error=err)
#     ratio_1 = (energy - e_min_1) / (e_max_1 - e_min_1)
#     approx_rat_1.append(ratio_1)
#     energy = ring_noise_2.simulate_qaoa_with_measurements([(0.125, 0.25)],
#                                                           40000,
#                                                           with_noise=False,
#                                                           readout_error=err)
#     ratio_2 = (energy - e_min_2) / (e_max_2 - e_min_2)
#     approx_rat_2.append(ratio_2)
# fig_1 = pyplot.figure()
# pyplot.plot(readout_errs, approx_rat_0, '-o', figure=fig_1, label = '10 qubits')
# pyplot.plot(readout_errs, approx_rat_1, '-o', figure=fig_1, label = '16 qubits')
# pyplot.plot(readout_errs, approx_rat_2, '-o', figure=fig_1, label = '22 qubits')
# pyplot.xlabel('Readout Error')
# pyplot.ylabel('Approximation ratio')
# pyplot.legend()



angles_p1_0 = [(0.6248792357272374, 0.3818259006216306)]
angles_p2_0 = [(0.1878111137535059, 0.31973548595499657),
             (0.5983859191538918, 0.6047804550767809)]
angles_p3_0 = [(0.20946018613969017, 0.29273420394788296),
             (0.6698911360655642, 0.575061131801023),
             (1.0888056680572784, 0.6574350228649777)]
angles_p4_0 = [(0.21546158971336668, 0.23442523540940796),
             (0.6886909888472169, 0.5030519892588232),
             (0.6505403599497521, 0.6120887016501921),
             (0.570609944976628, 0.6551805354678963)]
angles_p5_0 = [(0.7170359543948878, 0.26355754516491364),
             (1.191684375108038, 0.5409778511356115),
             (1.0563486348602702, 0.6235053278643471),
             (0.5970133240892671, 0.02250626868864024),
             (0.5699733827034462, 0.680713665344706)]
angles_list_0 = [angles_p1_0, angles_p2_0, angles_p3_0, angles_p4_0, angles_p5_0]

angles_p1 = [(0.62468593, 0.4934736)]
angles_p2 = [(0.1583621, 0.34615967), (0.57573666, 0.7010879)]
angles_p3 = [(0.19102644, 0.3678837), (0.62694268, 0.67251257),
             (1.06568109, 0.8203547)]
angles_p4 = [(0.18556831, 0.27322502), (0.64285801, 0.57102229),
             (0.59369967, 0.67140039),
             (0.55005325, 0.73263248)]
angles_p5 = [(0.70220405, 0.32089995), (1.15205846, 0.667185),
             (1.06806818, 0.73508268),
             (0.54132326, 0.13009389), (0.54812806, 0.78578933)]
angles_list = [angles_p1, angles_p2, angles_p3, angles_p4, angles_p5]

# x = numpy.arange(5) + 1
# beta_0 = [0.7170359543948878, 1.191684375108038, 1.0563486348602702, 0.5970133240892671, 0.5699733827034462]
# beta_1 = [0.70220405, 1.15205846, 1.06806818, 0.54132326, 0.54812806]
# gamma_0 = [0.26355754516491364, 0.5409778511356115, 0.6235053278643471, 0.02250626868864024, 0.680713665344706]
# gamma_1 = [0.32089995, 0.667185, 0.73508268, 0.13009389, 0.78578933]
# fig_1 = pyplot.figure()
# pyplot.plot(x, beta_0, '-o', figure=fig_1,
#             label='Before Optimization')
# pyplot.plot(x, beta_1, '-o', figure=fig_1,
#             label='After Optimization')
# pyplot.xlabel(r"Step")
# pyplot.ylabel(r'$\beta / \pi$')
# pyplot.legend()
# fig_2 = pyplot.figure()
# pyplot.plot(x, gamma_0, '-o', figure=fig_2,
#             label='Before Optimization')
# pyplot.plot(x, gamma_1, '-o', figure=fig_2,
#             label='After Optimization')
# pyplot.xlabel(r"Step")
# pyplot.ylabel(r'$\gamma / \pi$')
# pyplot.legend()


#
# approx_rat = []
# approx_rat_1 = []
# approx_rat_2 = []
# approx_rat_3 = []
# approx_rat_3_1 = []
# approx_rat_4 = []
# approx_rat_5 = []
# p_s = range(5)
# #
# numpy.random.seed(12)
# ring_22_0 = QubitRing(22, noise_h=0.0, noise_j=0.0)
# ring_22_0.randomize_h()
# ring_22_0.randomize_j(num_bits=2001)
#
# couplings = ring_22_0.j_couplings
# ring_22_1 = QubitRing(22, couplings=couplings, noise_h=0.02, noise_j=0.02)
# ring_22_2 = QubitRing(22, couplings=couplings, noise_h=0.05, noise_j=0.05)
# ring_22_3 = QubitRing(22, couplings=couplings, noise_h=0.1, noise_j=0.1)
# ring_22_4 = QubitRing(22, couplings=couplings, noise_h=0.2, noise_j=0.2)
#
# ring_12 = ring_noise
# energy_analysis_12 = ring_12.get_all_energies()
# E_max_12 = energy_analysis_12['E_max']
# E_min_12 = energy_analysis_12['E_min']
# max_indices_12 = energy_analysis_12['max_indices']
# energy_list_12 = energy_analysis_12['energy_list']
#
# energy_analysis_22 = ring_22_0.get_all_energies()
# E_max_22 = energy_analysis_22['E_max']
# E_min_22 = energy_analysis_22['E_min']
# max_indices_22 = energy_analysis_22['max_indices']
# energy_list_22 = energy_analysis_22['energy_list']
#
# # for p in p_s:
# #     _, state_probs = ring_12.simulate_qaoa(angles_list[p], with_noise=False)
# #     energy = numpy.sum(energy_list_12 * state_probs)
# #     approx_rat.append((energy - E_min_12) / (
# #             E_max_12 - E_min_12))
#
# # for p in p_s:
# #     _, state_probs = ring_22_0.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat.append((energy - E_min_22) / (
# #                         E_max_22 - E_min_22))
# #
# # for p in p_s:
# #     _, state_probs = ring_22_1.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_1.append((energy - E_min_22) / (
# #                         E_max_22 - E_min_22))
# #
# # for p in p_s:
# #     _, state_probs = ring_22_2.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_2.append((energy - E_min_22) / (
# #                         E_max_22 - E_min_22))
# #
# for p in p_s:
#     _, state_probs = ring_22_3.simulate_qaoa(angles_list[p], with_noise=True)
#     energy = numpy.sum(energy_list_22 * state_probs)
#     approx_rat_3.append((energy - E_min_22) / (
#             E_max_22 - E_min_22))
#
# # for p in p_s:
# #     _, state_probs = ring_22_4.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_4.append((energy - E_min_22) / (
# #                         E_max_22 - E_min_22))
#
# #
# # for p in p_s:
# #     _, state_probs = ring_22_2.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_2.append((energy - ring_22_0.size / 2.0) / (
# #                         E_max_22 - ring_22_0.size / 2.0))
#
#
# for p in p_s:
#     print(p)
#
#
#     def cost_function(a_list):
#         a_list = [(a_list[i], a_list[i + 1]) for i in
#                   numpy.array(range(p + 1)) * 2]
#         _, state_probs = ring_22_3.simulate_qaoa(a_list, with_noise=True)
#         return -numpy.sum(energy_list_22 * state_probs)
#
#
#     init_guess_0 = angles_list[p]
#     init_guess = []
#     for a, b in init_guess_0:
#         init_guess.append(a)
#         init_guess.append(b)
#
#     lower_bounds = numpy.array(init_guess) - 0.5
#     upper_bounds = numpy.array(init_guess) + 0.5
#
#     res = pybobyqa.solve(cost_function, numpy.array(init_guess),
#                          bounds=(lower_bounds, upper_bounds), maxfun=100)
#     # res = optimize.minimize(cost_function, init_guess,
#     #                         method='Powell', tol=1e-5)
#
#     # cost = -cost_function(res.x)
#     cost = -res.f
#     rat = (cost - E_min_22) / (
#             E_max_22 - E_min_22)
#     print(rat)
#     print(res.x)
#     approx_rat_3_1.append(rat)
# #
# # for p in p_s:
# #     _, state_probs = ring_22_3.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_3.append((energy - ring_22_0.size / 2.0) / (
# #                         E_max_22 - ring_22_0.size / 2.0))
#
#
# # for p in p_s:
# #     _, state_probs = ring_22_4.simulate_qaoa(angles_list[p], with_noise=True)
# #     energy = numpy.sum(energy_list_22 * state_probs)
# #     approx_rat_4.append((energy - ring_22_0.size / 2.0) / (
# #                         E_max_22 - ring_22_0.size / 2.0))
#
# # for p in p_s:
# #     _, state_probs = ring_12.simulate_qaoa(angles_list[p], with_noise=False)
# #     energy = numpy.sum(energy_list_12 * state_probs)
# #     approx_rat_5.append((energy - ring_12.size / 2.0) / (
# #                         E_max_12 - ring_12.size / 2.0))
#
#
# p_s = numpy.array(p_s) + 1.0
# p_ss = numpy.linspace(1, 5, 100)
# y = (1 + 2 * p_ss) / (2 + 2 * p_ss)
#
# fig_1 = pyplot.figure()
# # pyplot.plot(p_ss, y, '--', figure=fig_1,
# #             label='(2p + 1)/(2p + 2)')
# # pyplot.plot(p_s, approx_rat, '-x', figure=fig_1,
# #             label='{} qubits, no control error'.format(ring_12.size))
# pyplot.plot(p_s, approx_rat_3, '-o', figure=fig_1,
#             label='22 qubits, control error = 0.1')
# pyplot.plot(p_s, approx_rat_3_1, '-o', figure=fig_1,
#                 label='22 qubits, control error = 0.1 (after optimization)')
# # pyplot.plot(p_s, approx_rat, '-x', figure=fig_1,
# #                 label='{} qubits, no control error'.format(ring_12.size))
# # pyplot.plot(p_s, approx_rat, '-o', figure=fig_1,
# #                 label='{} qubits, no control error'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_1, '-o', figure=fig_1,
# #                 label='{} qubits, control error = 0.02'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_2, '-o', figure=fig_1,
# #                 label='{} qubits, control error = 0.05'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_3, '-o', figure=fig_1,
# #                 label='{} qubits, control error = 0.1'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_4, '-o', figure=fig_1,
# #                 label='{} qubits, control error = 0.2'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_2, 'o-', figure=fig_1,
# #                 label='{} qubits, control error = 0.05'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_3, 'o-', figure=fig_1,
# #                 label='{} qubits, control error = 0.1'.format(ring_22_0.size))
# # pyplot.plot(p_s, approx_rat_4, 'o-', figure=fig_1,
# #                 label='{} qubits, control error = 0.2'.format(ring_22_0.size))
# pyplot.xlabel(r"p")
# pyplot.ylabel('Approximation ratio')
# pyplot.legend()

# num_qubits = 14  # j_couplings = list(itertools.product([-1.0, 1.0], repeat=num_qubits))
# approx_rat = []
# noise_level = 0.0
# for i, j_coupling in enumerate(j_couplings):
#     print(i)
#     j_coupling = list(j_coupling)
#     qubit_ring_noise = QubitRing(num_qubits, couplings=j_coupling,
#                                  noise_h=noise_level, noise_j=noise_level)
#     approx_rat.append(single_angle_qaoa(0.125, 0.25, qubit_ring_noise))
# instance = range(2**num_qubits)
# fig_1 = pyplot.figure()
# pyplot.plot(instance, approx_rat, 'ro', figure=fig_1)
# pyplot.legend()
# pyplot.xlabel('Instance Number')
# pyplot.ylabel('Approximation Ratio, 14 Qubits')
# pyplot.ylim(bottom= 0.749, top = 0.751)

numpy.random.seed(8)
x_min = 0.0
x_max = 1.0
z_min = 0.0
z_max = 2.0
ring_noise = QubitRing(20, noise_h=0.0, noise_j=0.0)
# ring_noise.randomize_h()
# ring_noise.randomize_j(num_bits=9)
parameter_sweep_one_step(x_min, x_max, z_min, z_max, ring_noise, num_points=100, noise = False)
# parameter_sweep_one_step(x_min, x_max, z_min, z_max, ring_noise, num_points=30, noise = True)

# numpy.random.seed(12)
# x_min = 0.0
# x_max = 1.0
# z_min = 0.0
# z_max = 2.0
# ring_noise = QubitRing(20, noise_h=0.1, noise_j=0.1)
# ring_noise.randomize_h()
# ring_noise.randomize_j(num_bits=2)
# parameter_sweep_one_step(x_min, x_max, z_min, z_max, ring_noise, num_points=100, noise = False)
# parameter_sweep_one_step(x_min, x_max, z_min, z_max, ring_noise, num_points=100, noise = True)

# z_min = 0
# z_max = 2.0
# x_val = 0.125
# readout_errs = [0.0, 0.01, 0.02, 0.03, 0.05]
# num_qubits = 22
#
# results_list = one_d_sweep_one_step(z_min, z_max, x_val, readout_errs , num_qubits, num_points=100)
#
# noise_levels = numpy.linspace(0.0, 0.3, 10)
# rms_errs_16 = []
# num_qubits = 16
# for noise in noise_levels:
#     print('la')
#     rms_errs_16.append(error_approx_ratio(noise, num_qubits))
#
# rms_errs_12 = []
# num_qubits = 12
# for noise in noise_levels:
#     print('la')
#     rms_errs_12.append(error_approx_ratio(noise, num_qubits))
#
# rms_errs_8 = []
# num_qubits = 8
# for noise in noise_levels:
#     print('la')
#     rms_errs_8.append(error_approx_ratio(noise, num_qubits))


# fig_1 = pyplot.figure()
# pyplot.plot(noise_levels, rms_errs_16, 'ro-', figure=fig_1, label='16 qubits')
# pyplot.plot(noise_levels, rms_errs_12, 'bo-', figure=fig_1, label= '12 qubits')
# pyplot.plot(noise_levels, rms_errs_8, 'go-', figure=fig_1, label= '8 qubits')
# pyplot.legend()
# pyplot.xlabel('RMS amplitude of noise in h and j')
# pyplot.ylabel('RMS fluctuation of best approximation ratio')
# pyplot.yscale("log")
# pyplot.xscale("log")


# x = numpy.logspace(1, 4, num=10).astype(dtype=int)
# y = []
# qubit_ring = QubitRing(num_qubits=16)
# qubit_ring.randomize_j()
# for x_i in x:
#     print(x_i)
#     rms = one_d_sweep_one_step(x_min, x_max, z_val, qubit_ring, x_i, num_points=100)
#     y.append(rms)
#
# p1 = numpy.polyfit(numpy.log(x), numpy.log(y), 1)
# xx = numpy.logspace(1, 4, num=200).astype(dtype=int)
# yy = xx ** p1[0] * numpy.exp(p1[1])
#
# fig_1 = pyplot.figure()
# pyplot.plot(x, y, 'ro', figure=fig_1)
# pyplot.plot(xx, yy, "b-", label="Power law fit, $\propto 1/N**{}$".format(-p1[0]), figure = fig_1)
# pyplot.xlabel(r"Number of Measurements")
# pyplot.ylabel('RMS Error, 16 qubits')
# pyplot.yscale("log")
# pyplot.xscale("log")
# pyplot.legend()
