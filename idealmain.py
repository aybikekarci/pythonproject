import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Load training data from CSV files
def load_training_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    return data_frames

# Load ideal functions from CSV file
def load_ideal_functions(file_path):
    ideal_functions = pd.read_csv(file_path)
    return ideal_functions

# Define a function to fit an ideal function to training data using least-squares
def fit_ideal_function(x, y, ideal_function):
    try:
        params, _ = curve_fit(ideal_function, x, y)
        return params
    except Exception as e:
        print(f"Error while fitting: {str(e)}")
        return None

# Choose the best fit ideal function for each training dataset
def choose_best_fit_ideal_functions(training_data, ideal_functions):
    best_fit_ideals = []

    for i, training_set in enumerate(training_data):
        x_train, y_train = training_set['x'], training_set['y']
        best_fit_params = None
        best_fit_function = None

        for func_name in ideal_functions.columns[1:]:
            ideal_function = getattr(np, func_name)
            params = fit_ideal_function(x_train, y_train, ideal_function)

            if params is not None:
                fit_residuals = np.sum((y_train - ideal_function(x_train, *params))**2)
                if best_fit_params is None or fit_residuals < best_fit_params[0]:
                    best_fit_params = (fit_residuals, params)
                    best_fit_function = func_name

        best_fit_ideals.append((i + 1, best_fit_function, best_fit_params))

    return best_fit_ideals

# Unit tests
def test_fit_ideal_function():
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 1

    def linear_func(x, a, b):
        return a * x + b

    params = fit_ideal_function(x, y, linear_func)
    assert np.allclose(params, [2, 1], atol=1e-2)

def test_choose_best_fit_ideal_functions():
    # Create mock training data and ideal functions
    training_data = [pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]}),
                     pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]}),
                     pd.DataFrame({'x': [1, 2, 3], 'y': [1, 8, 27]}),
                     pd.DataFrame({'x': [1, 2, 3], 'y': [1, 16, 81]})]

    ideal_functions = pd.DataFrame({'x': [1, 2, 3], 'Ideal1': [1, 2, 3], 'Ideal2': [1, 4, 9]})

    best_fit_ideals = choose_best_fit_ideal_functions(training_data, ideal_functions)
    assert best_fit_ideals == [(1, 'Ideal1', (0.0, [1.0, 1.0])), 
                               (2, 'Ideal2', (0.0, [1.0, 1.0])),
                               (3, 'Ideal2', (0.0, [1.0, 1.0])),
                               (4, 'Ideal2', (0.0, [1.0, 1.0]))]

if __name__ == "__main__":
    # Load training data and ideal functions
    training_data = load_training_data(["training_data.csv"])
    ideal_functions = load_ideal_functions("ideal_functions.csv")


    # Run unit tests
    test_fit_ideal_function()
    print("Unit tests passed.")