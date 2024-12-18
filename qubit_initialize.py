import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, mesolve, basis, Options
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # Ensure this package is installed
import multiprocessing

def define_physical_constants():
    """
    Define and return the physical constants used in the simulation.
    """
    h = 4.135  # Planck's constant in μeV·GHz⁻¹
    mu_B = 58.0         # Bohr magneton in μeV/T
    g_u = 2.0           # g-factor (assumed)
    dg = 0.005         # g difference between two dots
    B = 0.4             # Magnetic field in Tesla (100 mT)
    t_c = h * 1       # Tunnel coupling in 1 GHz
    Delta_SO = h * 1e-2

    gu_muB_B = g_u * mu_B * B  # μeV
    dg_muB_B = dg * mu_B * B  # μeV

    return gu_muB_B, dg_muB_B, t_c, Delta_SO, h

def hamiltonian_matrix(epsilon, gu_muB_B, dg_muB_B, t_c, Delta_SO, h):
    hbar = 0.658  # Planck's constant in μeV·GHz⁻¹
    """
    Construct and return the Hamiltonian matrix as a Qobj.
    """
    H_matrix = np.array([
        [gu_muB_B,     0,          0,      0,      Delta_SO],
        [0,            0,          0,      dg_muB_B,      0],
        [0,            0,     -gu_muB_B,    0,      Delta_SO],
        [0,            dg_muB_B,          0,      0,    t_c],
        [Delta_SO,            0,           Delta_SO,   t_c,  -epsilon]
    ], dtype=complex)
    
    return Qobj(H_matrix)

def calculate_eigenvalues(epsilon_values, gu_muB_B, dg_muB_B, t_c, Delta_SO, h):
    """
    Calculate and return sorted eigenvalues for each epsilon in epsilon_values.
    """
    all_eigenvalues = []
    for epsilon in epsilon_values:
        H = hamiltonian_matrix(epsilon, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)
        eigenvals = H.eigenenergies()
        all_eigenvalues.append(np.sort(eigenvals.real))
    
    return np.array(all_eigenvalues)

def plot_eigenvalues(epsilon_values, all_eigenvalues):
    """
    Plot energy eigenvalues versus detuning ε.
    """
    plt.figure(figsize=(10, 7))
    for i in range(all_eigenvalues.shape[1]):
        plt.plot(epsilon_values, all_eigenvalues[:, i], label=f'Eigenvalue {i+1}')
    plt.xlabel('Detuning ε (μeV)', fontsize=14)
    plt.ylabel('Frequency Eigenvalues (GHz)', fontsize=14)
    plt.title('Frequency Eigenvalues vs Detuning for Singlet-Triplet Hamiltonian', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def detuning(t, sweep_duration, epsilon_start=75.0, epsilon_end=-150.0):
    """
    Define the detuning ε(t) as a function of time t.
    """
    if t < 0:
        return epsilon_start
    elif t > sweep_duration:
        return epsilon_end
    else:
        return epsilon_start + (epsilon_end - epsilon_start) * (t / sweep_duration)

def initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, Delta_SO, h):
    """
    Initialize and return the ground state at the initial detuning ε_initial.
    """
    H_initial = hamiltonian_matrix(epsilon_initial, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)
    _, eigenstates_initial = H_initial.eigenstates()
    return eigenstates_initial[0]

def define_final_eigenstates(epsilon_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, h):
    """
    Calculate and return the eigenstates at the final detuning ε_final.
    """
    H_final = hamiltonian_matrix(epsilon_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)
    eigenvals_final, eigenstates_final = H_final.eigenstates()
    sorted_indices = np.argsort(eigenvals_final)
    eigenstates_sorted = [eigenstates_final[i] for i in sorted_indices]
    return eigenstates_sorted

def simulate_single_sweep(sweep_duration, initial_state, eigenstates_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, epsilon_initial, epsilon_final, h):
    """
    Simulate a single sweep and return the probabilities of ending in each final eigenstate.
    """
    args = {'sweep_duration': sweep_duration}
    H0_qobj = hamiltonian_matrix(0.0, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)
    H1_matrix = np.zeros((5, 5), dtype=complex)
    H1_matrix[4, 4] = -1.0
    H1_qobj = Qobj(H1_matrix)
    H = [H0_qobj, [H1_qobj, lambda t, args: detuning(t, args['sweep_duration'], epsilon_initial, epsilon_final)]]

    opts = Options(atol=1e-10, rtol=1e-8, nsteps=1e9)
    result = mesolve(H, initial_state, [0, sweep_duration], [], [], args=args, options=opts)
    final_state = result.states[-1]

    probabilities = []
    for final_eigenstate in eigenstates_final:
        overlap = final_eigenstate.overlap(final_state)
        probabilities.append(np.abs(overlap) ** 2)
    
    return probabilities

def simulate_sweep_parallel(sweep_durations, initial_state, eigenstates_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, epsilon_initial, epsilon_final, h, num_cores=-1):
    """
    Simulate sweeps in parallel and return the probabilities matrix.
    """
    # Prepare arguments for parallel execution
    parallel = Parallel(n_jobs=num_cores, backend="multiprocessing")
    tasks = (delayed(simulate_single_sweep)(
                sweep_duration, initial_state, eigenstates_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, epsilon_initial, epsilon_final, h
             ) for sweep_duration in sweep_durations)
    
    # Execute tasks with a progress bar
    print("Starting parallel sweep simulations...")
    with tqdm_joblib(tqdm(desc="Sweeps", total=len(sweep_durations))):
        results = parallel(tasks)
    print("Parallel sweep simulations completed.\n")
    
    return np.array(results)

def plot_probabilities(sweep_durations, probabilities, eigenstates_final):
    """
    Plot the probabilities of ending in each final eigenstate versus sweep duration.
    """
    plt.figure(figsize=(12, 8))
    for state_idx in range(probabilities.shape[1]):
        plt.plot(sweep_durations, probabilities[:, state_idx], label=f'Final State {state_idx+1}')
    plt.xscale('log')
    plt.xlabel('Sweep Duration (ns)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Final State Probabilities vs Sweep Duration', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Define physical constants
    gu_muB_B, dg_muB_B, t_c, Delta_SO, h = define_physical_constants()

    # 2. Calculate and plot eigenvalues over a range of epsilon
    epsilon_min, epsilon_max, num_points = -100.0, 100.0, 400
    epsilon_values = np.linspace(epsilon_min, epsilon_max, num_points)
    all_eigenvalues = calculate_eigenvalues(epsilon_values, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)
    plot_eigenvalues(epsilon_values, all_eigenvalues)

    # 3. Initialize sweep parameters
    epsilon_initial, epsilon_final = 75.0, -2000.0
    sweep_durations = np.logspace(-2, 3, 300)  # Sweep durations from 0.01 ns to 1000 ns

    # 4. Initialize the initial state
    initial_state = initialize_initial_state(epsilon_initial, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)

    # 5. Define final eigenstates at epsilon_final
    eigenstates_final = define_final_eigenstates(epsilon_final, gu_muB_B, dg_muB_B, t_c, Delta_SO, h)

    # 6. Simulate sweeps in parallel
    num_cores = -1  # Use all available cores; set to an integer to specify the number of cores
    probabilities = simulate_sweep_parallel(
        sweep_durations, initial_state, eigenstates_final, gu_muB_B, dg_muB_B, t_c, Delta_SO,
        epsilon_initial, epsilon_final, h, num_cores=num_cores
    )

    # 7. Plot the results
    plot_probabilities(sweep_durations, probabilities, eigenstates_final)

if __name__ == "__main__":
    main()
