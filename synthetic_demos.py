import matplotlib.pyplot as plt
import numpy as np

from colect.mixture import GaussianMixtureModel
from colect.kmp import KMP

# Demonstation database settings
H = 5 # Number of demonstrations
N = 500 # Number of points in the demonstrations

# Demonstration settings
T = 5 # Demo duration [s]
A0 = 0.1 # sin offset
A = 0.01 # sin amplitude
f = 1 # sin frequency [Hz]

# Constant coordinates
x0 = 0.5*np.ones(N).reshape(-1,1)
y0 = 0.5*np.ones(N).reshape(-1,1)
quat = np.array([0, 0, 0, 1]) # Vertical ee
quat = np.tile(quat, (N, 1))

def sinusoidal_profile():

    def noisy_sin(frequency, amplitude, amplitude_offset, total_time, num_points):
        time = np.linspace(0, total_time, num_points)
        # Calculate the sine wave
        sine_wave = amplitude_offset + amplitude * np.sin(2 * np.pi * frequency * time)
        # Generate random standard deviation for Gaussian noise
        gaussian_noise_std = np.random.uniform(0.01, 0.015)  # You can adjust the range based on your preference
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(0, gaussian_noise_std, num_points)
        # Add noise to the sine wave
        return (sine_wave + gaussian_noise ).reshape(-1,1)

    demos = []
    t = np.linspace(0, 5, N)
    for h in range(H):
        # Generate sin wave
        z = noisy_sin(f, A, A0, T, N)
        demo = np.hstack((x0, y0, z, quat))
        demos.append(demo)
        if h == 1:
            plt.plot(t, demo[:,2], color="grey", linewidth=0.5, label="Demonstrations")
        else:
            plt.plot(t, demo[:,2], color="grey", linewidth=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Position z - [m]")
    plt.grid()
    return demos

def extract_input_data(datasets: np.ndarray, field: str, dt: float = 0.1) -> np.ndarray:

    # Extract the shape of each demonstration
    H = len(datasets)  # Number of demonstrations
    N = len(datasets[0])  # Length of each demonstration

    # Input vector for time-based GMM
    x_gmr = dt * np.arange(1, N + 1).reshape(1, -1)
    X = np.tile(x_gmr, H).reshape(1, -1)

    # Depending on the requested field, set up the output vectors
    demos = np.vstack(datasets)
    Y = demos[:,2].reshape(-1,1).T

    # Add the outputs to the dataset
    X = np.vstack((X, Y))

    #if field != "force":
    # Compute the derivatives of the outputs
    Y = np.split(Y, H, axis=1)
    for y in Y:
        y[0, :] = np.gradient(y[0, :]) / dt
    dY = np.hstack(Y)

    # Add the derivatives to the dataset
    X = np.vstack((X, dY))

    # The shape of the dataset should be (n_samples, n_features)
    return X.T

# Generate the artificial demonstrations
demos = sinusoidal_profile()

# Inputs/outputs for GMM/GMR/KMP
gmm_dt = 0.1
x_gmr = gmm_dt * np.arange(0, N).reshape(1, -1)
X_pos = extract_input_data(demos, 'position')
kmp_dt = 0.1
x_kmp = kmp_dt * np.arange(0, N).reshape(1, -1)

# GMM/GMR on the position
gmm = GaussianMixtureModel(n_components=14, n_demos=H)
gmm.fit(X_pos)
mu_pos, sigma_pos = gmm.predict(x_gmr)

# KMP on the position
kmp = KMP(l=0.5, sigma_f=5, verbose=True)
kmp.fit(x_gmr, mu_pos, sigma_pos)
mu_pos_kmp, sigma_pos_kmp = kmp.predict(x_kmp)

t_gmr = np.linspace(0, T, N)
plt.plot(t_gmr, mu_pos[0,:], label="GMR")
plt.plot(t_gmr, mu_pos_kmp[0,:], label="KMP", color="red")
plt.fill_between(x=t_gmr, y1=mu_pos[0, :]+np.sqrt(sigma_pos[0, 0, :]), y2=mu_pos[0, :]-np.sqrt(sigma_pos[0, 0, :]),color="blue",alpha=0.35)    
plt.fill_between(x=t_gmr, y1=mu_pos_kmp[0, :]+np.sqrt(sigma_pos_kmp[0, 0, :]), y2=mu_pos_kmp[0, :]-np.sqrt(sigma_pos_kmp[0, 0, :]),color="red",alpha=0.35)    
plt.legend()
plt.show()

np.save("sinusoidal_traj", mu_pos_kmp[0,:])
