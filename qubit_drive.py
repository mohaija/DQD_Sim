import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, mesolve, basis, Options, expect
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing
import json
from datetime import datetime
import os  # Added for directory operations

def define_physical_constants():
    h = 4.1356  # Planck's constant in μeV·ns
    mu_B = 58.0       # Bohr magneton in μeV/T
    g_u = 2.0         # g-factor (assumed)
    dg = 0.005        # g difference between two dots
    B = 0.4           # Magnetic field in Tesla
    t_c_GHz = h * 1     # Tunnel coupling in μeV (since 1 GHz ~4.1357 μeV)
    t_HF = h * 1e-2     # Hyperfine interaction in μeV
    hbar = 0.658

    # Both example dephasing rates are at 0.2 MHz (5 [μs])
    T1 = 1e8
    T2 = 1e3
    gamma_charge = 0
    gamma_spin = 1/T2
    gamma_relax = 1/T1

    gu_muB_B = g_u * mu_B * B    # μeV
    dg_muB_B = dg * mu_B * B     # μeV
    t_c = t_c_GHz                # μeV

    return gu_muB_B, dg_muB_B, t_c, t_HF, mu_B, B, h, gamma_charge, gamma_spin, gamma_relax

def hamiltonian_matrix(epsilon, gu_muB_B, dg_muB_B, t_c, t_HF):
    hbar = 0.658
    # Basis ordering: [|T+>, |T0>, |T->, |S(1,1)>, |S(0,2)>]
    H_matrix = np.array([
        [gu_muB_B,     0,          0,      0,          t_HF],
        [0,            0,          0,      0.5*dg_muB_B,   0],
        [0,            0,     -gu_muB_B,    0,         t_HF],
        [0,        0.5*dg_muB_B,      0,        0,         t_c],
        [t_HF,          0,      t_HF,      t_c,    -epsilon]
    ], dtype=complex) / hbar

    return Qobj(H_matrix)

def define_transformation_matrix():
    sqrt_half = 1 / np.sqrt(2)
    U_matrix = np.array([
        [1,          0,          0,        0, 0],  # |T+⟩ remains the same
        [0,    sqrt_half,  sqrt_half,        0, 0],  # |↑↓⟩ expressed in terms of |T0⟩ and |S(1,1)⟩
        [0,          0,          0,        1, 0],  # |T-⟩ remains the same
        [0,    sqrt_half, -sqrt_half,        0, 0],  # |↓↑⟩ expressed in terms of |T0⟩ and |S(1,1)⟩
        [0,          0,          0,        0, 1],  # |S(0,2)⟩ remains the same
    ], dtype=complex)
    U_qobj = Qobj(U_matrix)
    return U_qobj

def detuning(t, sweep_duration, epsilon_start=75.0, epsilon_end=-2000.0):
    if t < 0:
        return epsilon_start
    elif t > sweep_duration:
        return epsilon_end
    else:
        return epsilon_start + (epsilon_end - epsilon_start) * (t / sweep_duration)

def initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_HF):
    H_initial = hamiltonian_matrix(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_HF)
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

def simulate_sweep(initial_state, epsilon_initial, epsilon_final, sweep_duration, gu_muB_B, dg_muB_B, t_c, t_HF):
    """Simulate the energy sweep and return the final state."""
    args = {'sweep_duration': sweep_duration}
    H0_qobj = hamiltonian_matrix(0.0, gu_muB_B, dg_muB_B, t_c, t_HF)
    H1_matrix = np.zeros((5, 5), dtype=complex)
    H1_matrix[4, 4] = -1.0  # Detuning affects the |S(0,2)> state
    H1_qobj = Qobj(H1_matrix)
    H_sweep = [H0_qobj, [H1_qobj, lambda t, args: detuning(t, args['sweep_duration'], epsilon_initial, epsilon_final)]]
    opts = Options(atol=1e-9, rtol=1e-7, nsteps=1e9)
    print("Starting sweep simulation...")
    result_sweep = mesolve(H_sweep, initial_state, [0, sweep_duration], [], [], args=args, options=opts)
    final_state_after_sweep = result_sweep.states[-1]
    print("Sweep simulation completed.\n")
    return final_state_after_sweep

def define_basis_states():
    """Define basis states and their labels."""
    basis_states = [basis(5, i) for i in range(5)]
    state_labels = ['|T+⟩', '|T0⟩', '|T-⟩', '|S(1,1)⟩', '|S(0,2)⟩']
    return basis_states, state_labels

def analyze_eigenstates(H_static, basis_states, state_labels):
    """Calculate and display eigenvalues and eigenstates at epsilon_final."""
    hbar = 0.658
    eigenvals_static_freq, eigenstates_static = H_static.eigenstates()
    eigenvals_static = eigenvals_static_freq * hbar
    # Sort eigenvalues and eigenstates
    sorted_indices = np.argsort(eigenvals_static)
    eigenvals_sorted = eigenvals_static[sorted_indices]
    eigenstates_sorted = [eigenstates_static[i] for i in sorted_indices]

    # Print eigenvalues and their corresponding eigenstates overlaps with basis states
    print("Eigenstates at epsilon = -2000 μeV:")
    for idx, (E, state) in enumerate(zip(eigenvals_sorted, eigenstates_sorted)):
        overlaps = [np.abs(state.overlap(basis_states[i]))**2 for i in range(5)]
        overlaps_str = ', '.join([f"{state_labels[i]}: {overlaps[i]:.3f}" for i in range(5) if overlaps[i] > 0.01])
        print(f"Eigenstate {idx}: Energy = {E.real:.6f} μeV, Overlaps: {overlaps_str}")
    print()
    return eigenvals_sorted, eigenstates_sorted

def compute_overlaps(final_state_after_sweep, eigenstates_sorted):
    """Compute overlaps of the final state with eigenstates at epsilon_final."""
    print("Overlaps of final state after sweep with eigenstates at epsilon = -2000 μeV:")
    overlaps_initial_eigenstates = []
    for idx, state in enumerate(eigenstates_sorted):
        overlap = final_state_after_sweep.overlap(state)
        probability = np.abs(overlap)**2
        overlaps_initial_eigenstates.append(probability)
        print(f"Overlap with eigenstate {idx}: magnitude={np.abs(overlap):.4f}, Probability={probability:.6f}")
    print()
    return overlaps_initial_eigenstates

def identify_T_plus_eigenstate(basis_states, eigenstates_sorted, eigenvals_sorted):
    """Identify the eigenstate corresponding to |T+⟩."""
    T_plus_state = basis_states[0]  # |T+⟩
    overlaps_T_plus = [np.abs(eigenstate.overlap(T_plus_state))**2 for eigenstate in eigenstates_sorted]
    T_plus_index = np.argmax(overlaps_T_plus)
    E_T_plus = eigenvals_sorted[T_plus_index].real
    print(f"Eigenstate corresponding to |T+⟩ is Eigenstate {T_plus_index} with Energy = {E_T_plus:.6f} μeV\n")
    return T_plus_index, E_T_plus

def calculate_drive_parameters(E_initial, E_T_plus, hbar):
    """Calculate energy difference and drive frequency."""
    delta_E = E_T_plus - E_initial  # μeV
    drive_energy = abs(delta_E)  # μeV
    print(f"Calculated energy difference (delta_E) between |T+⟩ and final state: {delta_E:.6f} μeV")
    print(f"Translated drive energy: {drive_energy:.6f} μeV (E = ℏω)\n")
    return delta_E, drive_energy

def compute_probabilities(drive_frequency, H_static, H_int, initial_state, tlist, c_ops, opts, proj_ops, save_states, drive_strength):
    args = {'drive_frequency': drive_frequency, 'drive_strength': drive_strength}
    H = [H_static,
         [H_int, lambda t, args: args['drive_strength'] * np.cos(args['drive_frequency'] * t)]]

    if save_states:
        opts.store_states = True
        result = mesolve(H, initial_state, tlist, [], e_ops=[], args=args, options=opts)
        expectation_values = np.array([[expect(op, state) for state in result.states] for op in proj_ops])
    else:
        result = mesolve(H, initial_state, tlist, [], e_ops=proj_ops, args=args, options=opts)
        expectation_values = np.array(result.expect)  # Shape: (num_states, num_times)

    return expectation_values

def simulate_rabi_oscillations_parallel(initial_state, H_static, drive_strength, drive_frequencies, pulse_durations, basis_states, num_cores, gamma_charge, gamma_spin, gamma_relax, save_states=False):
    """Simulate Rabi oscillations in parallel over drive frequencies."""
    # Define interaction Hamiltonian and projection operators
    H_int = define_interaction_hamiltonian()

    # Define superposition states
    sqrt_half = 1 / np.sqrt(2)
    S_11_state = basis_states[3]  # |S(1,1)>
    T0_state = basis_states[1]    # |T0>
    psi_plus = (S_11_state + T0_state).unit()
    psi_minus = (S_11_state - T0_state).unit()

    # Create projection operators for basis states and superposition states
    proj_ops = [state * state.dag() for state in basis_states]  # Projections onto basis states
    proj_ops += [psi_plus * psi_plus.dag(), psi_minus * psi_minus.dag()]  # Projections onto superposition states

    # Update state labels
    state_labels = ['|T+⟩', '|T0⟩', '|T-⟩', '|S(1,1)⟩', '|S(0,2)⟩', '|↑↓⟩', '|↓↑⟩']

    opts = Options(atol=1e-9, rtol=1e-7, nsteps=1e9)

    # Construct collapse operators (same as before)
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
    # 4 Relaxing operators
    if gamma_relax != 0: # Relax to T- from state 1 (L3)
        sqrt_gamma_relax = np.sqrt(2 * gamma_relax)
        a_spin_relax1 = np.zeros((5,5), dtype=complex)
        a_spin_relax1[2,1] = sqrt_gamma_relax * sqrt_half
        a_spin_relax1[2,3] = -sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj1 = Qobj(a_spin_relax1)
        c_ops.append(a_spin_relax_qobj1)
    if gamma_relax != 0: # Relax to T- from state 2 (L4)
        sqrt_gamma_relax = np.sqrt(2 * gamma_relax)
        a_spin_relax2 = np.zeros((5,5), dtype=complex)
        a_spin_relax2[2,1] = sqrt_gamma_relax * sqrt_half
        a_spin_relax2[2,3] = sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj2 = Qobj(a_spin_relax2)
        c_ops.append(a_spin_relax_qobj2)
    if gamma_relax != 0: # Relax to state 1 from T+ (L1)
        sqrt_gamma_relax = np.sqrt(2 * gamma_relax)
        a_spin_relax3 = np.zeros((5,5), dtype=complex)
        a_spin_relax3[1,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax3[3,0] = -sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj3 = Qobj(a_spin_relax3)
        c_ops.append(a_spin_relax_qobj3)
    if gamma_relax != 0: # Relax to state 2 from T+ (L2)
        sqrt_gamma_relax = np.sqrt(2 * gamma_relax)
        a_spin_relax4 = np.zeros((5,5), dtype=complex)
        a_spin_relax4[1,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax4[3,0] = sqrt_gamma_relax * sqrt_half
        a_spin_relax_qobj4 = Qobj(a_spin_relax4)
        c_ops.append(a_spin_relax_qobj4)

    # Run in parallel over drive frequencies with progress bar
    print("Simulating Rabi oscillations...")

    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()
        print(f"Using all available cores ({num_cores})")
    else:
        print(f"Using {num_cores} cores")

    with tqdm_joblib(tqdm(desc="Drive Frequencies", total=len(drive_frequencies))) as progress_bar:
        probabilities_list = Parallel(n_jobs=num_cores)(
            delayed(compute_probabilities)(
                df, H_static, H_int, initial_state, pulse_durations, c_ops, opts, proj_ops, save_states, drive_strength)
            for df in drive_frequencies)

    # Stack the results into an array
    expectations = np.array(probabilities_list)  # Shape: (num_drive_frequencies, num_states, num_times)
    return expectations, state_labels

def plot_rabi_oscillations(pulse_durations, drive_energies, expectations, h, state_indices=None, state_labels=None, timestamp='', results_dir=''):
    """Plot the Rabi oscillations for specified state indices."""
    if state_indices is None:
        state_indices = [0]  # Default to initial state

    for idx in state_indices:
        probabilities = expectations[:, idx, :]  # Extract probabilities for the state
        plt.figure(figsize=(12, 8))
        X, Y = np.meshgrid(pulse_durations, drive_energies)
        pcm = plt.pcolormesh(X, Y, probabilities, shading='auto', cmap='viridis')
        plt.xlabel('Pulse Duration (ns)', fontsize=14)
        plt.ylabel('Drive Energy (μeV)', fontsize=14)
        state_label = state_labels[idx] if state_labels else f'State {idx}'
        plt.title(f'Rabi Oscillations for {state_label}', fontsize=16)
        cbar = plt.colorbar(pcm, label=f'Probability of {state_label}')
        plt.tight_layout()
        # Save the plot
        filename = f'rabi_oscillations_{state_label.replace("|", "").replace("⟩", "").replace("/", "")}_{timestamp}.png'
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved as {filepath}")
        plt.close()

def plot_probability_vs_pulse_duration(pulse_durations, drive_energies, expectations, target_E, h, state_indices=None, state_labels=None, timestamp='', results_dir=''):
    """Plot probability vs. pulse duration at a specific drive energy for specified states."""
    if state_indices is None:
        state_indices = [0]  # Default to initial state

    # Find the index of the drive energy closest to target_E μeV
    idx_E = np.abs(drive_energies - target_E).argmin()
    closest_E = drive_energies[idx_E]
    corresponding_freq = drive_energies[idx_E] / h  # Convert to GHz

    print(f"Selected drive energy: {closest_E:.2f} μeV (Index: {idx_E}), Frequency: {corresponding_freq:.4f} GHz")

    for idx in state_indices:
        # Extract the probabilities for this drive energy across all pulse durations
        probabilities_at_E = expectations[idx_E, idx, :]

        # Create the plot
        plt.figure(figsize=(10, 6))
        state_label = state_labels[idx] if state_labels else f'State {idx}'
        plt.plot(pulse_durations, probabilities_at_E, label=f'{state_label}')
        plt.axhline(0.5, color='red', linestyle='--', label='P = 0.5')
        plt.xlabel('Pulse Duration (ns)', fontsize=14)
        plt.ylabel(f'Probability of {state_label}', fontsize=14)
        plt.title(f'Probability vs. Pulse Duration at E = {closest_E:.2f} μeV', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        filename = f'probability_vs_pulse_duration_{state_label.replace("|", "").replace("⟩", "").replace("/", "")}_E_{closest_E:.2f}_{timestamp}.png'
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved as {filepath}")
        plt.close()

def main():
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create results directory
    results_dir = f'simulation_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in directory: {results_dir}")

    # 1. Define constants
    gu_muB_B, dg_muB_B, t_c, t_HF, mu_B, B, h, gamma_charge, gamma_spin, gamma_relax = define_physical_constants()
    hbar = h / (2 * np.pi)  # μeV·ns

    # 2. Initialize initial state: defines the initial state as the ground state of the Hamiltonian at detuning = epsilon_initial

    epsilon_initial, epsilon_final = 75.0, -2000.0
    initial_state = initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, t_HF)

    # 3. Simulate the sweep
    sweep_duration = 300.0  # ns
    final_state_after_sweep = simulate_sweep(initial_state, epsilon_initial, epsilon_final, sweep_duration, gu_muB_B, dg_muB_B, t_c, t_HF)

    # 4. Analyze eigenstates at epsilon_final
    basis_states, _ = define_basis_states()
    H_static = hamiltonian_matrix(epsilon_final, gu_muB_B, dg_muB_B, t_c, t_HF)
    eigenvals_sorted, eigenstates_sorted = analyze_eigenstates(H_static, basis_states, state_labels=['|T+⟩', '|T0⟩', '|T-⟩', '|S(1,1)⟩', '|S(0,2)⟩'])

    # 5. Compute overlaps of final state after sweep with eigenstates
    overlaps_initial_eigenstates = compute_overlaps(final_state_after_sweep, eigenstates_sorted)

    # 6. Calculate expectation value of energy for final state
    E_initial = expect(H_static, final_state_after_sweep).real
    print(f"Expectation value of energy for the final state after sweep: {E_initial:.6f} μeV\n")

    # 7. Identify the eigenstate corresponding to |T+⟩
    T_plus_index, E_T_plus = identify_T_plus_eigenstate(basis_states, eigenstates_sorted, eigenvals_sorted)

    # 8. Calculate energy difference and drive frequency (THIS IS IRRELEVANT FOR NOW AS THE DRIVING ENERGY IS CHOSEN MANUALLY BELOW)
    delta_E, drive_energy = calculate_drive_parameters(E_initial, E_T_plus, hbar)

    # 9. Define rabi frequency and simulation parameters
    drive_strength = h * 2e-3  # Adjust as needed (0.012μeV corresponds to 3[MHz])
    pulse_durations = np.linspace(0, 2000, 250)  # ns
    delta_energy = h * 0.031  
    num_energies = 250
    drive_energy = h * 11.224
    drive_energies = np.linspace(drive_energy - delta_energy, drive_energy + delta_energy, num_energies)  # μeV

    # 10. Simulate Rabi oscillations
    num_cores = -1  # Use all available cores
    save_states = False  # Set to True to save quantum states during simulation
    save_data = True  # Set to True to save the data and metadata
    print("Starting Rabi oscillations simulation over energy and pulse duration range...")
    expectations, state_labels = simulate_rabi_oscillations_parallel(
        final_state_after_sweep, H_static, drive_strength, drive_energies / hbar, pulse_durations, basis_states, num_cores, gamma_charge, gamma_spin, gamma_relax, save_states)
    print("Rabi oscillations simulation completed.\n")

    # Save data and metadata if requested
    if save_data:
        metadata = {
            'gu_muB_B': gu_muB_B,
            'dg_muB_B': dg_muB_B,
            't_c': t_c,
            't_HF': t_HF,
            'mu_B': mu_B,
            'B': B,
            'h': h,
            'gamma_charge': gamma_charge,
            'gamma_spin': gamma_spin,
            'gamma_relax': gamma_relax,
            'epsilon_initial': epsilon_initial,
            'epsilon_final': epsilon_final,
            'sweep_duration': sweep_duration,
            'drive_strength': drive_strength,
            'drive_energy': drive_energy,
            'delta_energy': delta_energy,
            'num_energies': num_energies,
            'num_cores': num_cores,
            'state_labels': state_labels,
        }
        # Convert numpy arrays to lists for JSON serialization
        metadata_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metadata.items()}
        # Save metadata as JSON
        metadata_filename = os.path.join(results_dir, f'metadata_{timestamp}.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata_serializable, f, indent=4)
        print(f"Metadata has been saved to '{metadata_filename}'.")

        # Save data arrays with timestamp
        np.save(os.path.join(results_dir, f'expectations_{timestamp}.npy'), expectations)
        np.save(os.path.join(results_dir, f'drive_energies_{timestamp}.npy'), drive_energies)
        np.save(os.path.join(results_dir, f'pulse_durations_{timestamp}.npy'), pulse_durations)
        print("Simulation data arrays have been saved with timestamped filenames.")

    # 11. Plot results
    # Now the state_labels include the superposition states |↑↓⟩ and |↓↑⟩
    # We can specify the indices of these states for plotting
    psi_plus_index = 5  # Index of |↑↓⟩ in state_labels
    psi_minus_index = 6  # Index of |↓↑⟩ in state_labels

    # Plot probabilities for the superposition states
    print(f"Plotting Rabi oscillations for {state_labels[psi_plus_index]} and {state_labels[psi_minus_index]}")
    plot_rabi_oscillations(pulse_durations, drive_energies, expectations, h, state_indices=[psi_plus_index, psi_minus_index], state_labels=state_labels, timestamp=timestamp, results_dir=results_dir)

    # Plot probability vs. pulse duration for a specific drive energy and states
    target_E = drive_energy
    print(f"Plotting probability vs. pulse duration for {state_labels[psi_plus_index]} and {state_labels[psi_minus_index]} at E = {target_E} μeV")
    plot_probability_vs_pulse_duration(pulse_durations, drive_energies, expectations, target_E=target_E, h=h, state_indices=[psi_plus_index, psi_minus_index], state_labels=state_labels, timestamp=timestamp, results_dir=results_dir)

if __name__ == "__main__":
    main()
