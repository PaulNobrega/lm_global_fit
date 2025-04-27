# lm_global_fit: Levenberg-Marquardt Global Fitter

A Python library for performing non-linear least squares curve fitting on multiple datasets simultaneously (global fitting) using the Levenberg-Marquardt algorithm. It supports parameter fixing, linking parameters across datasets using shared IDs, composite models (sum of multiple model functions per dataset), calculation of common goodness-of-fit statistics, confidence intervals (with bootstrap fallback), and model extrapolation.

This library is a high-fidelity translation and enhancement of the original [GlobalFit.JS](https://github.com/PaulNobrega/globalfitJS) library, which itself was adapted from the Fortran project Savuka. This Python version leverages NumPy and SciPy for numerical operations and offers parallel processing capabilities.

**Version:** 1.0.0

## Key Features & Enhancements

*   **Python Implementation:** Runs natively in Python environments.
*   **NumPy/SciPy Backend:** Utilizes NumPy for efficient array operations and SciPy for robust SVD (`scipy.linalg.svd`) and statistical functions (`scipy.stats.t`). Uses `numpy.linalg.solve` / `numpy.linalg.pinv` for solving linear systems within LM.
*   **Parallel Independent Fitting:** `lm_fit_independent` can fit datasets in parallel using Python's `multiprocessing` module, significantly speeding up analysis of large numbers of independent datasets.
*   **Parallel Bootstrap CI:** The bootstrap confidence interval calculation (used as a fallback) can also leverage `multiprocessing` for faster execution.
*   **Vectorized Model Support:** Encourages defining model functions that operate directly on NumPy arrays for improved performance, although it includes fallbacks for non-vectorized functions.
*   **Model Extrapolation:** Added `model_x_range` option to calculate and plot fitted curves, component curves, and confidence intervals beyond the original data range.
*   **Global Fitting:** Fit multiple datasets simultaneously with shared or independent parameters.
*   **Levenberg-Marquardt Algorithm:** Robust and widely used algorithm for non-linear least squares.
*   **Parameter Fixing:** Keep specific parameters constant during the fit using a `fixMap`.
*   **Parameter Linking:** Force parameters across different datasets or models to share the same fitted value using a `linkMap` with shared string/number IDs.
*   **Composite Models:** Define the model for a dataset as the sum of multiple individual model functions.
*   **Goodness-of-Fit Statistics:** Calculates Degrees of Freedom, Reduced Chi-Squared, AIC, AICc, and BIC.
*   **Covariance & Errors:** Returns the covariance matrix and standard errors for active parameters. Uses `abs()` for variance if numerically negative and logs a warning.
*   **Covariance Regularization:** Applies a small regularization factor during covariance matrix calculation for improved numerical stability.
*   **Custom Logging & Progress:** Provides options (`onLog`, `onProgress` callbacks) for users to handle verbose output and track progress.
*   **Parameter Constraints:** Supports simple box constraints (min/max) and custom constraint functions.
*   **Robust Cost Functions:** Optional use of L1 (Absolute Residual) or Lorentzian cost functions for outlier resistance.
*   **Confidence Intervals:** Calculates confidence intervals for fitted model curves using the covariance matrix and Student's t-distribution (Delta method).
*   **Bootstrap Fallback for Confidence Intervals:** Automatically falls back to multiprocessing-enabled bootstrapping when the covariance matrix yields negative variances or fails inversion.
*   **Simulation Functionality:** Generate synthetic datasets using `simulate_from_params` with support for Gaussian and Poisson noise.

## Advantages

*   **Improved Parameter Estimation:** Global fitting uses information from all datasets simultaneously, often leading to more precise and reliable parameter estimates, especially for shared parameters.
*   **Model Discrimination:** Allows testing hypotheses where certain parameters are expected to be the same across different experimental conditions (datasets).
*   **Flexibility:** Handles complex scenarios with multiple model components contributing to the overall signal for each dataset.
*   **Performance:** Leverages NumPy for vectorized calculations and `multiprocessing` for parallel execution of independent fits and bootstrap CIs.
*   **Python Ecosystem:** Integrates naturally with other scientific Python libraries (NumPy, SciPy, Matplotlib, etc.).
*   **Numerical Stability:** Uses SVD-based inversion (via SciPy) for the covariance matrix with regularization.
*   **Fallback Mechanisms:** Includes robust fallback mechanisms like bootstrapping for confidence intervals.

## Installation

Install the package directly from PyPI:

```bash
pip install lm_global_fit
```

## Usage Example

```python
import numpy as np
from lm_global_fit import (
    lm_fit_global,
    lm_fit,
    lm_fit_independent,
    simulate_from_params
)

# --- 1. Define Model Functions ---
# Must accept params=np.array([...]) and x=np.array([...]), return np.array([...]) if vectorized
# Or accept params=np.array([...]) and x=np.array([xValue]), return np.array([yValue]) if not vectorized
def gaussian_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gaussian model: A * exp(-0.5 * ((x - xc) / w)^2) (Vectorized)"""
    if len(params) != 3: raise ValueError("Gaussian model expects 3 parameters: [amp, center, stddev]")
    amp, center, stddev = params
    if stddev == 0: return np.full_like(x, np.nan)
    exponent = -0.5 * ((x - center) / stddev)**2
    with np.errstate(over='ignore', under='ignore'): result = amp * np.exp(exponent)
    result[~np.isfinite(result)] = 0.0
    result[np.abs(exponent) > 700] = 0.0
    return result

def linear_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear model through origin: y = m*x (Vectorized)"""
    if len(params) != 1: raise ValueError("Linear model (y=m*x) expects 1 parameter: [slope]")
    slope = params[0]
    return slope * x

# --- 2. Prepare Data ---
data_in = {
    'x': [
        [1, 2, 3, 4, 5, 6],  # Dataset 0
        [0, 1, 2, 3, 4, 5, 6, 7]  # Dataset 1
    ],
    'y': [
        [5.1, 8.2, 9.9, 10.1, 8.5, 5.3],  # Noisy Gaussian
        [1.9, 4.1, 5.9, 8.1, 10.0, 12.1, 13.8, 16.2]  # Noisy Linear
    ],
    'ye': [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Errors for DS0
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  # Errors for DS1
    ]
}

# --- 3. Define Model Structure ---
model_function_in = [
    [gaussian_model],  # Dataset 0: Gaussian only
    [linear_model]     # Dataset 1: Linear only
]

# --- 4. Initial Parameter Guesses ---
initial_parameters_in = [
    [[9.0, 3.5, 1.0]],  # DS0: [amp, center, stddev]
    [[2.0]]             # DS1: [slope]
]

# --- 5. Define Options ---
fit_options = {
    'maxIterations': 200,
    'logLevel': 'info',
    'confidenceInterval': 0.95,
    'calculateFittedModel': {'numPoints': 100},
    'bootstrapFallback': True,
    'numBootstrapSamples': 100
}

# --- 6. Run the Global Fit ---
result = lm_fit_global(data_in, model_function_in, initial_parameters_in, fit_options)

# --- 7. Process Results ---
if result.get('error'):
    print(f"Fit failed: {result['error']}")
else:
    print(f"Converged: {result['converged']} in {result['iterations']} iterations.")
    print(f"Final Chi^2: {result['chiSquared']:.5e}")
    print(f"Reduced Chi^2: {result['reducedChiSquared']:.5f}")
    print(f"Active Parameters: {result['p_active']}")
```

## API Reference

### `lm_fit_global(data, model_function, initial_parameters, options)`

Performs the global fit. *(Note: Current implementation is synchronous)*

**Parameters:**

*   `data` (`Dict[str, List[Sequence[float]]]`): Dictionary containing the experimental data. Requires keys:
    *   `'x'`: List where each element is a sequence (list, tuple, np.array) of independent variable values for a dataset.
    *   `'y'`: List of sequences of dependent variable values.
    *   `'ye'`: List of sequences of error/uncertainty values (std devs). **Must not contain zeros or negative values.**
*   `model_function` (`List[List[Callable[[np.ndarray, np.ndarray], np.ndarray]]]`): List of lists of model functions. `model_function[dsIdx]` is a list of functions for dataset `dsIdx`.
    *   Each individual function `func = model_function[dsIdx][paramGroupIdx]` represents a component of the total model for that dataset.
    *   It will be called as `func(params_array, x_array)`, where `params_array` is the corresponding NumPy parameter array and `x_array` is a NumPy array of x-values.
    *   Each function **must** return a NumPy array of calculated y-values corresponding to the input `x_array`. Vectorized functions (operating on the whole `x_array` at once) are preferred for performance. Non-vectorized functions (expecting a single x-value in `x_array`) will work but trigger a slower loop-based evaluation for curve generation.
    *   The results of all functions in `model_function[dsIdx]` are summed internally to get the final model value for dataset `dsIdx`.
*   `initial_parameters` (`List[List[List[float]]]`): Nested list of initial parameter guesses. Structure must align with `model_function`. `initial_parameters[dsIdx][paramGroupIdx]` is a list of numbers.
*   `options` (`Dict[str, Any]`, optional): Configuration object for the fit. See `DEFAULT_OPTIONS` for keys and defaults. Key options include:
    *   `fixMap` (`List[List[List[bool]]]`, optional): Defines fixed parameters (True = fixed). Structure matches `initial_parameters`.
    *   `linkMap` (`List[List[List[Optional[Union[str, int]]]]]`, optional): Defines parameter linking using shared IDs. Structure matches `initial_parameters`.
    *   `constraints` (`List[List[List[Optional[Dict[str, float]]]]]`, optional): Box constraints (`{'min': val, 'max': val}`). Structure matches `initial_parameters`.
    *   `constraintFunction` (`Callable[[ParametersNpType], ParametersNpType]`, optional): Custom function applied after box constraints. Takes and returns the nested list of NumPy arrays structure.
    *   `confidenceInterval` (`float`, optional): Level for CIs (e.g., 0.95).
    *   `bootstrapFallback` (`bool`, default: `True`): Use bootstrap if standard CI fails.
    *   `numBootstrapSamples` (`int`, default: `200`): Number of bootstrap samples.
    *   `calculateFittedModel` (`bool | Dict`, default: `False`): Calculate smooth fitted curves. Use `{'numPoints': N}` to specify points.
    *   `calculateComponentModels` (`bool`, default: `False`): Calculate smooth component curves.
    *   `model_x_range` (`List[Optional[Tuple[float, float]]]`, optional): List defining the calculation range `(min, max)` for each dataset's curves/CIs. `None` uses data range.
    *   `num_workers` (`int`, optional): Number of parallel workers for bootstrap CI. Defaults to `cpu_count()`. Set to 1 for sequential.
    *   `onLog`, `onProgress`, `logLevel`, `maxIterations`, `errorTolerance`, etc.

**Returns:**

*   `Dict[str, Any]`: A dictionary containing the fitting results (`ResultType`). Key fields:
    *   `p_active` (`List[float]`): Final active parameter values.
    *   `p_reconstructed` (`List[List[List[float]]]`): Full parameter structure with final values.
    *   `finalParamErrors` (`List[List[List[Optional[float]]]]`): Standard errors for all parameters (0 for fixed, propagated for slaves, `None` if NaN).
    *   `chiSquared` (`Optional[float]`): Final cost function value (`None` if NaN/Inf).
    *   `covarianceMatrix` (`Optional[List[List[float]]]`): Covariance matrix for active parameters (`None` if failed).
    *   `parameterErrors` (`List[Optional[float]]`): Standard errors for active parameters (`None` if NaN).
    *   `iterations` (`int`): Iterations performed.
    *   `converged` (`bool`): Convergence status.
    *   `activeParamLabels` (`List[str]`): Labels for active parameters.
    *   `error` (`Optional[str]`): Error message if fit failed.
    *   `totalPoints`, `degreesOfFreedom`, `reducedChiSquared`, `aic`, `aicc`, `bic`: Goodness-of-fit stats (`None` if calculation failed).
    *   `residualsPerSeries` (`Optional[List[np.ndarray]]`): List of residual arrays.
    *   `fittedModelCurves` (`Optional[List[Dict[str, np.ndarray]]]`): List of curve dictionaries (`{'x': np.array, 'y': np.array}`).
    *   `ci_lower`, `ci_upper` (`Optional[List[Dict[str, np.ndarray]]]`): Lists of CI bound dictionaries.
    *   `fittedModelComponentCurves` (`Optional[List[List[Dict[str, np.ndarray]]]]`): Nested list of component curve dictionaries.

### `lm_fit(data, model_function, initial_parameters, options)`

Convenience wrapper for fitting a **single** dataset.

*   Accepts `data` as `{x: Sequence[float], y: Sequence[float], ye: Sequence[float]}`.
*   Accepts `model_function` as `Callable | List[Callable]`.
*   Accepts `initial_parameters` as `Sequence[float] | List[Sequence[float]]`.
*   Accepts `options` like `lm_fit_global`, but maps/constraints should be in single-dataset format (e.g., `fixMap = [[False, True], [False]]`). `model_x_range` should be a single tuple `(min, max)` or `None`.
*   Returns the same result dictionary structure as `lm_fit_global`.

### `lm_fit_independent(data, model_function, initial_parameters, options, num_workers)`

Fits multiple datasets **independently**, potentially in parallel.

*   Accepts `data`, `model_function`, `initial_parameters` in the same multi-dataset format as `lm_fit_global`.
*   Accepts most `options` like `lm_fit_global`. `linkMap` is ignored. `fixMap`, `constraints`, `model_x_range` apply per-dataset if provided in the full nested structure.
*   `num_workers` (`int`, optional): Overrides `options['num_workers']` for parallel execution.
*   Returns a **list** of result dictionaries, one for each dataset fit.

### `simulate_from_params(data_x, model_functions, parameters, options)`

Generates simulated data.

*   Accepts `data_x` (list of x-sequences), `model_functions`, `parameters` (list-of-list-of-list).
*   `options` can include `noiseType` ('gaussian', 'poisson', 'none' or list) and `noiseStdDev` (number or list).
*   Returns `Dict[str, List[np.ndarray]]` containing keys `'x'` and `'y'` with lists of NumPy arrays.

## Notes & Considerations

*   **Dependencies:** Requires Python 3, NumPy, and SciPy. Matplotlib is needed for the example plots.
*   **Parallelism:** `lm_fit_independent` and bootstrap CI use `multiprocessing`. Performance gains depend on the number of datasets/samples, the complexity of model evaluations, and system overhead. Ensure model functions and custom constraints are picklable if using parallelism.
*   **Vectorization:** Define model functions to accept and return NumPy arrays for best performance, especially during curve/CI generation.
*   **Error Estimation:** Parameter errors are based on the covariance matrix derived from the Jacobian (first derivatives). Warnings are issued for negative variances (using `abs()`) or non-finite results.
*   **Covariance Matrix:** Regularization (`covarianceLambda`) is applied for stability.
*   **Robust Cost Functions:** Using `robustCostFunction: 1` or `2` changes the meaning of `chiSquared`. It's no longer strictly the sum of squared normalized residuals. The parameter values obtained will be Maximum Likelihood Estimates under the assumed noise distribution (double-exponential or Lorentzian, respectively), but interpreting the absolute value of the final "chiSquared" for goodness-of-fit requires care. Reduced Chi-Squared is less meaningful in these cases. AIC/BIC based on this modified chi-squared are still useful for *comparing* models fit with the *same* robust cost function.
*   **Model Function Signature:** Ensure model functions adhere to the expected signature `func(params_array, x_array)` returning a NumPy array.

## MIT License

Copyright (c) 2025 R. Paul Nobrega (Original JS), [Your Name/Year for Python Port]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
