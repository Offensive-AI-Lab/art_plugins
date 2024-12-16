import pandas as pd
import numpy as np

# Define constraint functions as given in the example
class FactoryCat:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def method(self, vector):
        vector = integer(vector)
        return bound(vector, self.min, self.max)

    def get_method(self):
        return self.method

class Bound:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def method(self, vector):
        return bound(vector, self.min, self.max)

    def get_method(self):
        return self.method

def bound(vector, v_min=-np.inf, v_max=np.inf):
    return np.clip(vector, v_min, v_max)

def integer(vector, v_min=0, v_max=0):
    return np.round(vector)

def binary(vector, v_min=0, v_max=0):
    vector = integer(vector)
    return bound(vector, 0, 1)

def positive(v_min=0, v_max=np.inf):
    return Bound(v_min, v_max).get_method()

def negative(v_min=-np.inf, v_max=0):
    return Bound(v_min, v_max).get_method()

def categorical(v_min=0, v_max=0):
    return FactoryCat(v_min, v_max).get_method()

def normalized(vector, v_min=0, v_max=1):
    return bound(vector, v_min, v_max)

def normalized_negative(vector, v_min=-1, v_max=1):
    return bound(vector, v_min, v_max)

# Function to determine constraints based on data statistics
def generate_constraints(data):
    constraints = {}
    for col in data.columns:
        min_val, max_val = data[col].min(), data[col].max()

        if data[col].nunique() == 2 and set(data[col].dropna().unique()).issubset({0, 1}):
            # Binary feature
            constraints[col] = [binary]
        elif data[col].dtype == 'object' or data[col].nunique() < 10:
            # Categorical feature
            min_cat = 0
            max_cat = data[col].nunique() - 1
            constraints[col] = [categorical(min_cat, max_cat)]
        elif pd.api.types.is_integer_dtype(data[col]):
            if min_val >= 0:
                # Positive integer feature
                constraints[col] = [positive(0), integer]
            elif min_val > 0:
                # Positive integer feature (greater than 0)
                constraints[col] = [positive(1), integer]
            else:
                # Negative integer feature
                constraints[col] = [negative(), integer]
        elif pd.api.types.is_float_dtype(data[col]):
            if min_val >= 0 and max_val <= 1:
                # Normalized positive feature (0 to 1)
                constraints[col] = [normalized]
            elif min_val >= -1 and max_val <= 1:
                # Normalized negative feature (-1 to 1)
                constraints[col] = [normalized_negative]
            elif min_val >= 0:
                # Positive continuous feature
                constraints[col] = [positive(0)]
            elif min_val > 0:
                # Positive integer feature (greater than 0)
                constraints[col] = [positive(1)]
            else:
                # Negative continuous feature
                constraints[col] = [negative()]
        else:
            # Default case for other data types
            constraints[col] = [bound]
    
    return constraints