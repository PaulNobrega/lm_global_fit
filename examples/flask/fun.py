import numpy as np # Ensure numpy is imported

# --- Model Function Definitions (Vectorized) ---

def gaussian_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gaussian model: A * exp(-0.5 * ((x - xc) / w)^2) (Vectorized)"""
    if len(params) != 3: raise ValueError("Gaussian model expects 3 parameters: [amp, center, stddev]")
    amp, center, stddev = params
    if stddev == 0: return np.full_like(x, np.nan) # Return NaN array if stddev is zero

    # Calculate exponent using array operations
    exponent = -0.5 * ((x - center) / stddev)**2

    # Handle potential overflow/underflow element-wise
    # Use np.errstate to temporarily ignore warnings during exp calculation
    with np.errstate(over='ignore', under='ignore'):
        result = amp * np.exp(exponent)

    # Replace Inf/-Inf resulting from overflow with 0.0 (or amp if needed?)
    # Or handle based on sign of amp? Let's use 0.0 for simplicity.
    result[~np.isfinite(result)] = 0.0
    # Replace values where exponent was too small (underflow) with 0.0
    result[np.abs(exponent) > 700] = 0.0 # Re-apply threshold check

    return result

def linear_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear model through origin: y = m*x (Vectorized)"""
    if len(params) != 1: raise ValueError("Linear model (y=m*x) expects 1 parameter: [slope]")
    slope = params[0]
    return slope * x # NumPy handles element-wise multiplication

def constant_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Constant model: c (Vectorized)"""
    if len(params) != 1: raise ValueError("Constant model expects 1 parameter: [offset]")
    offset = params[0]
    # Return an array of the same shape as x, filled with the offset
    return np.full_like(x, offset, dtype=float)

def exponential_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exponential model: A * exp(-rate * x) (Vectorized)"""
    if len(params) != 2: raise ValueError("Exponential model expects 2 parameters: [amplitude, rate]")
    amplitude, rate = params

    # Calculate exponent using array operations
    exponent = -rate * x

    # Handle potential overflow/underflow element-wise
    with np.errstate(over='ignore', under='ignore'):
        result = amplitude * np.exp(exponent)

    # Replace Inf/-Inf/NaN resulting from overflow/underflow
    result[~np.isfinite(result)] = 0.0
    result[np.abs(exponent) > 700] = 0.0 # Re-apply threshold check

    return result

# --- Model Mapping (Remains the same) ---
MODEL_MAP = {
    "gaussian": gaussian_model,
    "linear": linear_model,
    "constant": constant_model,
    "exponential": exponential_model
}