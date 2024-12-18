import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, mesolve, basis, expect
import os
import json
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # Ensure this package is installed

def define_physical_constants():
    h = 4.1356  # Planck's constant in μeV·ns
    mu_B = 58.0       # Bohr magneton in μeV/T
    g_u = 2.0         # g-factor (assumed)
    dg = 0.005        # g difference between two dots
    B = 0.4           # Magnetic field in Tesla
    t_c_GHz = h * 1     # Tunnel coupling in μeV (since 1 GHz ~4.1357 μeV)
    t_SO = h * 1e-2     # Hyperfine interaction in μeV
    hbar = h / (2 * np.pi)  # μeV·ns

    # Dephasing and relaxation rates
    T1 = 1e8
    T2 = 1000

    gamma_charge = 0
    gamma_relax = 1/T1
    gamma_spin = 1/T2

    gu_muB_B = g_u * mu_B * B    # μeV
    dg_muB_B = dg * mu_B * B     # μeV
    t_c = t_c_GHz                # μeV

    return gu_muB_B, dg_muB_B, t_c, t_SO, mu_B, B, h, gamma_charge, gamma_spin, gamma_relax

def hamiltonian_matrix(epsilon, gu_muB_B, dg_muB_B, t_c, t_SO):
    hbar = 0.658
    # Basis ordering: [|T+>, |T0>, |T->, |S(1,1)>, |S(0,2)>]
    H_matrix = np.array([
        [gu_muB_B,     0,          0,      0,          t_SO],
        [0,            0,          0,      dg_muB_B,   0],
        [0,            0,     -gu_muB_B,    0,         t_SO],
        [0,        dg_muB_B,      0,        0,         t_c],
        [t_SO,          0,      t_SO,      t_c,    -epsilon]
    ], dtype=complex) / hbar

    return Qobj(H_matrix)

def detuning(t, sweep_duration, epsilon_start=75.0, epsilon_end=-2000.0):
    if t < 0:
        return epsilon_start
    elif t > sweep_duration:
        return epsilon_end
    else:
        return epsilon_start + (epsilon_end - epsilon_start) * (t / sweep_duration)

def initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_SO):
    H_initial = hamiltonian_matrix(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_SO)
    eigenvals_initial, eigenstates_initial = H_initial.eigenstates()
    # Initialize in the ground state at epsilon_initial
    return eigenstates_initial[0]

def define_interaction_hamiltonian():
    hbar = 0.658
    # Interaction Hamiltonian in the basis [|T+>, |T0>, |T->, |S(1,1)>, |S(0,2)>]
    H_int_matrix = np.zeros((5,5), dtype=complex)
    # Coupling between |T+> and |T0>
    H_int_matrix[0,1] = np.sqrt(2)
    H_int_matrix[1,0] = np.sqrt(2)
    # Coupling between |T0> and |T->
    H_int_matrix[1,2] = np.sqrt(2)
    H_int_matrix[2,1] = np.sqrt(2)
    # No coupling involving |S(1,1)> and |S(0,2)>
    H_int_qobj = Qobj(H_int_matrix) / hbar
    return H_int_qobj

def simulate_sweep(initial_state, epsilon_initial, epsilon_final, sweep_duration, gu_muB_B, dg_muB_B, t_c, t_SO):
    """Simulate the energy sweep and return the final state."""
    args = {'sweep_duration': sweep_duration}
    H0_qobj = hamiltonian_matrix(0.0, gu_muB_B, dg_muB_B, t_c, t_SO)
    H1_matrix = np.zeros((5, 5), dtype=complex)
    H1_matrix[4, 4] = -1.0  # Detuning affects the |S(0,2)> state
    H1_qobj = Qobj(H1_matrix)
    H_sweep = [H0_qobj, [H1_qobj, lambda t, args: detuning(t, args['sweep_duration'], epsilon_initial, epsilon_final)]]
    opts = {'atol': 1e-9, 'rtol': 1e-7, 'nsteps': int(1e9)}
    result_sweep = mesolve(H_sweep, initial_state, [0, sweep_duration], [], [], args=args, options=opts)
    final_state_after_sweep = result_sweep.states[-1]
    return final_state_after_sweep

def define_basis_states():
    """Define basis states and their labels."""
    basis_states = [basis(5, i) for i in range(5)]
    state_labels = ['|T+⟩', '|T0⟩', '|T-⟩', '|S(1,1)⟩', '|S(0,2)⟩']
    return basis_states, state_labels

def simulate_chirp_excitation(initial_state, H_static, coupling_strength, omega_0, chirp_rate, pulse_duration, c_ops):
    """Simulate chirp-pulse excitation."""
    H_int = define_interaction_hamiltonian()
    args = {
        'omega_0': omega_0,
        'chirp_rate': chirp_rate,
        'coupling_strength': coupling_strength
    }
    H = [H_static, [H_int, chirped_drive]]
    tlist = np.linspace(0, pulse_duration, 750)  # Adjust number of points as needed
    opts = {'atol': 1e-9, 'rtol': 1e-7, 'nsteps': int(1e9)}
    result = mesolve(H, initial_state, tlist, c_ops, [], args=args, options=opts)
    return tlist, result.states

def chirped_drive(t, args):
    """Time-dependent function for chirped pulse excitation."""
    omega_0 = args['omega_0']
    chirp_rate = args['chirp_rate']
    coupling_strength = args['coupling_strength']
    phase = omega_0 * t + 0.5 * chirp_rate * t**2
    return coupling_strength * np.cos(phase)

def prepare_collapse_operators(gamma_charge, gamma_spin, gamma_relax):
    """Prepare the collapse operators for the simulation."""
    c_ops = []
    sqrt_half = 1 / np.sqrt(2)

    if gamma_charge != 0:
        sqrt_gamma_charge = np.sqrt(gamma_charge / 2)
        a_charge = np.diag([sqrt_gamma_charge]*4 + [-sqrt_gamma_charge])
        a_charge_qobj = Qobj(a_charge)
        c_ops.append(a_charge_qobj)

    if gamma_spin != 0:
        sqrt_gamma_spin = np.sqrt(2 * gamma_spin)
        a_spin_dephase = np.zeros((5,5), dtype=complex)
        a_spin_dephase[0,0] = sqrt_gamma_spin
        a_spin_dephase[2,2] = -sqrt_gamma_spin
        a_spin_dephase_qobj = Qobj(a_spin_dephase)
        c_ops.append(a_spin_dephase_qobj)

    if gamma_relax != 0:
        sqrt_gamma_relax = np.sqrt(2 * gamma_relax)
        # Relax to T- from state 1 (L3)
        a_spin_relax1 = np.zeros((5,5), dtype=complex)
        a_spin_relax1[2,1] = sqrt_gamma_relax * sqrt_half
        a_spin_relax1[2,3] = -sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj1 = Qobj(a_spin_relax1)
        c_ops.append(a_spin_relax_qobj1)
        # Relax to T- from state 2 (L4)
        a_spin_relax2 = np.zeros((5,5), dtype=complex)
        a_spin_relax2[2,1] = sqrt_gamma_relax * sqrt_half
        a_spin_relax2[2,3] = sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj2 = Qobj(a_spin_relax2)
        c_ops.append(a_spin_relax_qobj2)
        # Relax to state 1 from T+ (L1)
        a_spin_relax3 = np.zeros((5,5), dtype=complex)
        a_spin_relax3[1,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax3[3,0] = -sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj3 = Qobj(a_spin_relax3)
        c_ops.append(a_spin_relax_qobj3)
        # Relax to state 2 from T+ (L2)
        a_spin_relax4 = np.zeros((5,5), dtype=complex)
        a_spin_relax4[1,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax4[3,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj4 = Qobj(a_spin_relax4)
        c_ops.append(a_spin_relax_qobj4)

    return c_ops

def simulate_for_center_freq(f_center, coupling_strength, delta_f, T_chirp, final_state_after_sweep, H_static, c_ops):
    f_0 = f_center - delta_f
    f_1 = f_center + delta_f

    omega_0 = 2 * np.pi * f_0  # rad/ns
    omega_1 = 2 * np.pi * f_1  # rad/ns
    chirp_rate = (omega_1 - omega_0) / T_chirp  # rad/ns^2

    # Simulate chirp excitation
    tlist, states = simulate_chirp_excitation(
        final_state_after_sweep, H_static, coupling_strength, omega_0, chirp_rate, T_chirp, c_ops
    )

    # Define basis states
    basis_states, state_labels = define_basis_states()

    # Projectors for P(T+) and P(T-)
    proj_T_plus = basis_states[0] * basis_states[0].dag()
    proj_T_minus = basis_states[2] * basis_states[2].dag()

    # Compute P(T+) and P(T-) at final time
    P_T_plus = expect(proj_T_plus, states[-1])
    P_T_minus = expect(proj_T_minus, states[-1])

    P_blocked = P_T_plus + P_T_minus

    return P_blocked

def save_metadata(metadata, directory):
    """Save metadata dictionary as a JSON file."""
    metadata_path = os.path.join(directory, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

def save_data(center_freqs, P_blocked_list, directory):
    """Save simulation data as NumPy binary files."""
    np.save(os.path.join(directory, 'center_freqs.npy'), center_freqs)
    np.save(os.path.join(directory, 'P_blocked_list.npy'), P_blocked_list)
    print(f"Simulation data saved to {directory}")

def main():
    # 1. Define constants
    gu_muB_B, dg_muB_B, t_c, t_SO, mu_B, B, h, gamma_charge, gamma_spin, gamma_relax = define_physical_constants()
    hbar = h / (2 * np.pi)  # μeV·ns

    # 2. Initialize initial state
    epsilon_initial, epsilon_final = 75.0, -2000.0
    initial_state = initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_SO)

    # 3. Simulate the sweep
    sweep_duration = 300.0  # ns
    final_state_after_sweep = simulate_sweep(initial_state, epsilon_initial, epsilon_final, sweep_duration, gu_muB_B, dg_muB_B, t_c, t_SO)

    # 4. Define H_static
    H_static = hamiltonian_matrix(epsilon_final, gu_muB_B, dg_muB_B, t_c, t_SO)

    # 5. Prepare collapse operators
    c_ops = prepare_collapse_operators(gamma_charge, gamma_spin, gamma_relax)

    # 6. Define chirp excitation parameters
    coupling_strength = h * 10e-3  # μeV
    delta_f = 0.13  # GHz
    T_chirp = 1000  # ns
    print(f"Level splitting due to delta_g is: {(dg_muB_B/h)*1e3} [MHz]")

    # 7. Define center frequencies to sweep over
    f_center_start = 11.0  # GHz
    f_center_end = 11.45  # GHz
    num_points = 200
    center_freqs = np.linspace(f_center_start, f_center_end, num_points)

    # 8. Determine number of CPU cores and print it
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    print(f"Utilizing all {num_cores} cores.\n")

    # 9. Start parallel simulations with progress bar using joblib
    print("Starting parallel simulations...")

    # Define a wrapper function to pass to joblib
    def simulate_for_freq_wrapper(f_center):
        return simulate_for_center_freq(f_center, coupling_strength, delta_f, T_chirp, final_state_after_sweep, H_static, c_ops)

    # Use tqdm_joblib to get a progress bar
    with tqdm_joblib(tqdm(desc="Center Frequencies", total=len(center_freqs))) as progress_bar:
        P_blocked_list = Parallel(n_jobs=num_cores)(
            delayed(simulate_for_freq_wrapper)(f_center) for f_center in center_freqs
        )

    print("Parallel simulations completed.\n")

    # 10. Create a directory to save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"simulation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in directory: {results_dir}\n")

    # 11. Prepare metadata
    metadata = {
        'timestamp': timestamp,
        'simulation_parameters': {
            'epsilon_initial': epsilon_initial,
            'epsilon_final': epsilon_final,
            'sweep_duration_ns': sweep_duration,
            'chirp_parameters': {
                'coupling_strength_μeV': coupling_strength,
                'delta_f_GHz': delta_f,
                'T_chirp_ns': T_chirp
            },
            'decoherence_rates': {
                'gamma_charge': gamma_charge,
                'gamma_spin': gamma_spin,
                'gamma_relax': gamma_relax
            },
            'center_frequency_range_GHz': {
                'start': f_center_start,
                'end': f_center_end,
                'num_points': num_points
            }
        }
    }

    # 12. Save simulation data and metadata
    save_data(center_freqs, P_blocked_list, results_dir)
    save_metadata(metadata, results_dir)

    # 13. Plot P_blocked vs center frequency
    plt.figure(figsize=(10,6))
    plt.plot(center_freqs, P_blocked_list, 'o-')
    plt.xlabel('Center Frequency (GHz)', fontsize=14)
    plt.ylabel('$P_{\\text{blocked}}$', fontsize=14)
    plt.title('$P_{\\text{blocked}}$ vs Center Frequency', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(results_dir, "P_blocked_vs_center_frequency.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}\n")

    plt.show()

if __name__ == "__main__":
    main()
