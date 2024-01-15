import pandas as pd
import matplotlib.pyplot as plt

# Load training datasets (A), ideal functions (C), and selected ideal functions
training_data = pd.read_csv('training_data.csv')
ideal_functions = pd.read_csv('ideal_functions.csv')
selected_ideal_functions = pd.read_csv('selected_ideal_functions.csv')

# Create subplots for data visualization
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Data Visualization')

# Plot the training datasets
for i in range(4):
    ax = axs[i // 2, i % 2]
    ax.scatter(training_data['x'], training_data[f'y{i+1}'], label=f'Training Dataset {i+1}', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel(f'y{i+1}')
    ax.set_title(f'Training Dataset {i+1}')

# Plot the ideal functions
for i in range(50):
    ax = axs[i // 25, i % 2]
    ax.plot(ideal_functions['y50'], ideal_functions[f'y{i+1}'], label=f'Ideal Function {i+1}', linestyle='--', alpha=0.7)

# Plot the selected ideal functions (best fit)

# Add legends and adjust layout
for ax in axs.ravel():
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplot layout
plt.show()