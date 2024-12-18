import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from matplotlib.collections import LineCollection  # For colored line plotting
import matplotlib.colors as mcolors  # For color normalization

# Set this variable to True to enable saving results, False to disable
SAVE_RESULTS = False

def main():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if SAVE_RESULTS:
        save_dir = f"simulation_output_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"All outputs will be saved to the directory: {save_dir}")
    else:
        save_dir = None

    # Simulation parameters
    params = {
        # Physical constants
        'mu_B': 58.0,   # Bohr magneton in μeV/T
        'g': 2,         # g-factor
        'B': 0.4,       # Magnetic field in Tesla
        'h': 4.1356,    # μeV·ns

        # Rabi frequency
        'Omega_R': 2 * np.pi * 8e-3,  # Rabi frequency (rad/ns)

        # Hamiltonian parameters
        'delta_f': 75e-3,       # Total frequency range around f_0 (GHz)
        'pulse_duration': 5000, # Pulse duration in ns

        # Relaxation and dephasing times
        'T1': 4e3,   # Relaxation time in ns
        'T2': 2e3,   # Coherence time in ns
    }

    # Derived parameters
    params['hbar'] = params['h'] / (2 * np.pi)
    params['omega_0'] = params['g'] * params['mu_B'] * params['B'] / params['hbar']  # Resonance frequency in GHz
    params['f_0'] = params['g'] * params['mu_B'] * params['B'] / params['h']         # Resonance frequency in GHz
    print(params['f_0'])
    params['f_start'] = params['f_0'] - params['delta_f'] / 2   # Start frequency of chirp (GHz)
    print(params['f_start'])
    params['f_end'] = params['f_0'] + params['delta_f'] / 2     # End frequency of chirp (GHz)
    print(params['f_end'])
    params['alpha'] = (params['f_end'] - params['f_start']) / params['pulse_duration']  # Chirp rate (GHz/ns)
    params['gamma1'] = 1 / params['T1']      # Relaxation rate
    params['gamma_phi'] = 1 / params['T2']   # Dephasing rate

    # Time array
    params['tlist'] = np.linspace(0, params['pulse_duration'], 2000)

    # Operators
    operators = {
        'sx': sigmax(),
        'sy': sigmay(),
        'sz': sigmaz(),
        'si': qeye(2),
    }

    # Initial state (ground state)
    psi0 = basis(2, 1)

    # Hamiltonian
    H0 = 0.5 * params['omega_0'] * operators['sz']
    H1 = 0.5 * params['Omega_R'] * operators['sx']

    f_start = params['f_start']
    alpha = params['alpha']
    t0 = params['pulse_duration'] / 2
    sigma = params['pulse_duration'] / 5  # Adjust sigma as needed

    # Time-dependent drive with Gaussian envelope
    envelope = 'gaussian'  # or 'constant'

    if envelope == 'constant':
        def drive(t, args):
            return np.cos(2 * np.pi * f_start * t + np.pi * alpha * t**2)
    elif envelope == 'gaussian':
        def drive(t, args):
            envelope = np.exp(- (t - t0)**2 / (2 * sigma**2))
            return envelope * np.cos(2 * np.pi * f_start * t + np.pi * alpha * t**2)
    else:
        raise ValueError("Invalid envelope type. Choose 'constant' or 'gaussian'.")

    H = [H0, [H1, drive]]

    # Collapse operators
    c_relax = np.sqrt(params['gamma1']) * sigmam()          # Relaxation (decay from |1> to |0>)
    c_dephase = np.sqrt(2 * params['gamma_phi']) * operators['sz']  # Dephasing
    c_ops = [c_relax, c_dephase]

    # Solve the master equation
    e_ops = [operators['sx'], operators['sy'], operators['sz']]
    opts = Options(store_states=True, atol=1e-9, rtol=1e-7, nsteps=1e9)
    result = mesolve(H, psi0, params['tlist'], [], e_ops, options=opts)

    # Compute the expectation values in the rotating frame
    expect_x = result.expect[0]
    expect_y = result.expect[1]
    expect_z = result.expect[2]

    phi_t = 2 * np.pi * (params['f_start'] * params['tlist'] + 0.5 * params['alpha'] * params['tlist']**2)
    #phi_t = 2 * np.pi * (params['f_0'] * params['tlist'])

    cos_phi_t = np.cos(phi_t)
    sin_phi_t = np.sin(phi_t)
    expect_x_rot = cos_phi_t * expect_x + sin_phi_t * expect_y
    expect_y_rot = -sin_phi_t * expect_x + cos_phi_t * expect_y
    expect_z_rot = expect_z

    # Save simulation parameters and final state if enabled
    if SAVE_RESULTS:
        # Save parameters
        params_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}
        params_path = os.path.join(save_dir, 'simulation_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params_serializable, f, indent=4)
        print(f"Simulation parameters saved to {params_path}")

        # Save final state
        final_state = result.states[-1] if hasattr(result, 'states') and result.states else result.final_state
        final_state_path = os.path.join(save_dir, 'final_state.pkl')
        with open(final_state_path, 'wb') as f:
            pickle.dump(final_state, f)
        print(f"Final state saved to {final_state_path}")

    # -------------------------
    # Modified Plot Section
    # -------------------------

    # Plot evolution of state probabilities only
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=200)  # Changed from (2,1) to (1,1)

    ax1.plot(params['tlist'], (1 - expect_z) / 2, label="Probability in state |↓⟩", linewidth=1)
    ax1.plot(params['tlist'], (1 + expect_z) / 2, label="Probability in state |↑⟩", linewidth=1)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.set_title("Evolution of State Probabilities in Basis States")
    ax1.grid(True)

    plt.tight_layout()

    if SAVE_RESULTS:
        prob_plot_path = os.path.join(save_dir, 'state_probabilities.png')
        plt.savefig(prob_plot_path)
        print(f"Probability plot saved to {prob_plot_path}")
    plt.show()

    # Plot Bloch sphere trajectory
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    b = Bloch(fig=fig, axes=ax)
    b.point_size = [4]
    b.point_color = ['blue']
    b.line_color = ['red']
    b.vector_color = ['green']
    b.point_marker = ['o']
    b.frame_width = 1
    b.add_points([expect_x_rot, expect_y_rot, expect_z_rot], meth='l')
    b.add_vectors([expect_x_rot[-1], expect_y_rot[-1], expect_z_rot[-1]])
    b.make_sphere()
    if SAVE_RESULTS:
        bloch_plot_path = os.path.join(save_dir, 'bloch_sphere.png')
        plt.savefig(bloch_plot_path)
        print(f"Bloch sphere plot saved to {bloch_plot_path}")
    plt.show()

    # Compute eigenenergies and eigenstates as a function of detuning
    delta_start = 2 * np.pi * (params['f_start'] - params['f_0'])  # Convert to rad/ns
    delta_end = 2 * np.pi * (params['f_end'] - params['f_0'])      # Convert to rad/ns
    delta_list = np.linspace(delta_start, delta_end, 1000)
    delta_list_GHz = delta_list / (2 * np.pi)
    eigenenergies = []
    eigenstates_expectations = {'sx': [], 'sy': [], 'sz': []}

    for delta in delta_list:
        H_RWA = 0.5 * delta * operators['sz'] + 0.5 * params['Omega_R'] * operators['sx']
        evals, estates = H_RWA.eigenstates()
        eigenenergies.append(evals / (2 * np.pi))  # Convert to GHz
        for est in estates:
            eigenstates_expectations['sx'].append(expect(operators['sx'], est))
            eigenstates_expectations['sy'].append(expect(operators['sy'], est))
            eigenstates_expectations['sz'].append(expect(operators['sz'], est))

    eigenenergies = np.array(eigenenergies)
    for key in eigenstates_expectations:
        eigenstates_expectations[key] = np.array(eigenstates_expectations[key]).reshape(len(delta_list), 2)

    # -------------------------
    # Modified Eigenenergies Plot
    # -------------------------
    # Plot eigenenergies with color mapping according to ⟨σ_z⟩
    fig, ax = plt.subplots(figsize=(8,6))

    for idx in range(2):  # For each eigenstate
        x = 4.13566*delta_list_GHz
        y = eigenenergies[:, idx] * 1e3
        z = eigenstates_expectations['sz'][:, idx]  # ⟨σ_z⟩ values

        # Create segments for LineCollection
        points = np.array([x, y]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection with colormap based on ⟨σ_z⟩
        lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(-1,1))
        lc.set_array(z[:-1])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    ax.set_xlim(4.13566*delta_list_GHz.min(), 4.13566*delta_list_GHz.max())
    ax.set_ylim(eigenenergies.min()*1e3, eigenenergies.max()*1e3)

    ax.set_xlabel('Detuning Δ (μeV)')
    ax.set_ylabel('Eigenenergies (MHz)')
    ax.set_title('Eigenenergies vs Detuning')
    ax.grid(True)

    # Add colorbar indicating ⟨σ_z⟩
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('⟨σ_z⟩')

    plt.tight_layout()

    if SAVE_RESULTS:
        eigenenergy_colormap_plot_path = os.path.join(save_dir, 'eigenenergies_colormap.png')
        plt.savefig(eigenenergy_colormap_plot_path)
        print(f"Eigenenergy colormap plot saved to {eigenenergy_colormap_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
