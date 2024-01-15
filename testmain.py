import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load training data (A), test data (B), and ideal functions (C)
training_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('test_data.csv')
ideal_functions = pd.read_csv('ideal_functions.csv')

# Define a function to fit an ideal function to training data
def fit_ideal_function(x, y):
    # Define the function to fit (e.g., a polynomial)
    def ideal_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Perform the curve fitting
    popt, _ = curve_fit(ideal_function, x, y)
    
    return ideal_function, popt

# Choose the four ideal functions that minimize the sum of squared deviations
chosen_ideal_functions = []

for i in range(4):
    x_train = training_data['x']
    y_train = training_data[f'y{i + 1}']
    
    ideal_function, params = fit_ideal_function(x_train, y_train)
    
    chosen_ideal_functions.append({'function': ideal_function, 'parameters': params})

# Map test data to the chosen ideal functions and calculate deviations
test_results = []

for _, row in test_data.iterrows():
    x_test = row['x']
    y_test = row['y']
    
    best_fit = None
    min_deviation = float('inf')
    
    for ideal_func in chosen_ideal_functions:
        predicted_y = ideal_func['function'](x_test, *ideal_func['parameters'])
        deviation = abs(y_test - predicted_y)
        
        if deviation < min_deviation:
            min_deviation = deviation
            best_fit = ideal_func
        
    test_results.append({
        'x (test func)': x_test,
        'y (test func)': y_test,
        'Chosen Ideal Function': best_fit['function'].__name__,
        'Deviation': min_deviation
    })

# Save test results to a CSV file
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv('test_results.csv', index=False)

