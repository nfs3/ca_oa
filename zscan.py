"""

1. F (Transition Matrix) = [1]:

    This matrix assumes the normalized intensity remains constant over time. This might be appropriate if you expect minimal or negligible changes in the normalized intensity during the
    filtering process. However, if you anticipate some variations in the intensity, the matrix might need adjustments to reflect those changes.

2. H (Observation Matrix) = [1]:

    This matrix indicates that only the normalized intensity is directly measured, which aligns with the provided code snippet.

3. initial_state_mean = 51.38:

    This is your initial estimate for the normalized intensity. It's difficult to assess without context about the data and expected behavior. However, ensure this value aligns with your
    understanding of the initial system state.

4. sigma_n (Initial State Covariance) = 2639:

    This value represents a high initial uncertainty in the normalized intensity. Since it's on the diagonal of the initial state covariance matrix, it specifically reflects the uncertainty in
    the initial normalized intensity value. You might want to consider:
        Reviewing the spread or variability in the data points around the initial normalized intensity to see if the uncertainty is realistically high.
        Revisiting your initial guess for the normalized intensity and potentially adjusting it based on prior knowledge or data analysis.

5. sigma_m (Observation Covariance) = 0.8:

    This value indicates a medium level of uncertainty in your measurements. The choice seems reasonable unless you have specific knowledge about the expected noise level in your measurements.
    You may want to adjust this value based on your data analysis or sensor specifications.


6. Q (Process Covariance) = 0.001:

    This value represents a low level of uncertainty in the process noise. This suggests you expect minimal variations in the system's state transitions, which might be appropriate
    depending on the nature of your system.

"""

###

import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.optimize import curve_fit

wave_length = 660e-9
w0 = 13e-6
k = 2 * np.pi / wave_length
z0 = k * w0 * w0 / 2

def z_formulae(z, phi):
    x = z/z0
    return 1 + phi * 4 * x / ((1 + x*x) * (9 + x*x))

# Read CSV data
df = pd.read_csv("/content/drive/MyDrive/datum/193_ca_oa.csv", header=0)
df_x = pd.read_csv("/content/drive/MyDrive/datum/193_calculated.csv", header=0)

# Extract relevant data
high = df_x.x.max()       # Maximum x value
low = df_x.x.min()        # Minimum x value
x = np.linspace(low, high, df.shape[0])  # Create evenly spaced x values

# Process data
df.columns = ['Closed Aperture', 'Open Aperture']  # Set column names
df['Normalized'] = df['Closed Aperture'] / df['Open Aperture']  # Calculate normalized signal
df['Normalized'] = df['Normalized'] / df['Normalized'].mean()  # Normalize to mean 1
df['x'] = x + 0.0037  # Add offset to x values


# Assuming 'Closed Aperture' is a valid column in df
closed_aperture_list = list(df['Closed Aperture'])

kf = KalmanFilter(transition_matrices=[1],
                   observation_matrices=[1],
                   initial_state_mean=51.38,
                   initial_state_covariance=2639,
                   observation_covariance=0.8,
                   transition_covariance=.01)

# Apply Kalman filter (filtering only)
filtered_data, _ = kf.filter(closed_aperture_list)


# Normalize the filtered data
normalized_filtered_data = filtered_data / filtered_data.mean()

# Extract the 1-dimensional arrays for x and normalized data
x_data = df['x'].to_numpy()
normalized_data = df['Normalized'].to_numpy()  # Assuming the normalized data is in the 'Normalized' column

# Fit z_formulae to the normalized data
popt, pcov = curve_fit(z_formulae, x_data, normalized_data)

# Normalize the Y-axis limits for plotting
min_y, max_y = plt.ylim()
plt.ylim(min_y + 0.3, max_y + 0.5)
# Plot the results
#plt.plot(x_data, filtered_data, label='KF Filter Estimate')  # Use x_data consistently
plt.plot(x_data, normalized_filtered_data, label='Normalized KF Filter Estimate')
plt.plot(x_data, z_formulae(x_data, *popt), label='Fitted z_formulae')


# Find the peak and valley points of the normalized filtered data
normalized_filtered_peak_index = np.argmax(normalized_filtered_data)
normalized_filtered_valley_index = np.argmin(normalized_filtered_data)

normalized_filtered_peak_x_value = x_data[normalized_filtered_peak_index]
normalized_filtered_peak_y_value = normalized_filtered_data[normalized_filtered_peak_index]

normalized_filtered_valley_x_value = x_data[normalized_filtered_valley_index]
normalized_filtered_valley_y_value = normalized_filtered_data[normalized_filtered_valley_index]

# Plot the peak and valley points of the normalized filtered data
plt.plot(normalized_filtered_peak_x_value, normalized_filtered_peak_y_value, 'o', color='blue', markersize=5, label='Normalized Filtered Peak')
plt.annotate(f"f_Peak: ({normalized_filtered_peak_x_value:.4f}, {normalized_filtered_peak_y_value[0]:.4f})",
             xy=(normalized_filtered_peak_x_value, normalized_filtered_peak_y_value),
             xytext=(normalized_filtered_peak_x_value + 0.007, normalized_filtered_peak_y_value + 0))

plt.plot(normalized_filtered_valley_x_value, normalized_filtered_valley_y_value, 'o', color='purple', markersize=5, label='Normalized Filtered Valley')
plt.annotate(f"f_Valley: ({normalized_filtered_valley_x_value:.4f}, {normalized_filtered_valley_y_value[0]:.4f})",
             xy=(normalized_filtered_valley_x_value, normalized_filtered_valley_y_value),
             xytext=(normalized_filtered_valley_x_value + 0.007, normalized_filtered_valley_y_value - 0))

print(popt)
# Add labels and title
plt.xlabel("X-axis (from df['x'])")
plt.ylabel("Normalized Closed Aperture")
plt.title("KF Filter Estimate with Peak, Valley, and Fitted z_formulae")
plt.legend(loc="lower left")
plt.show()

print(popt)