# globalfit_py_v2.py

import sys
import time
import warnings
import queue # For multiprocessing queue
from typing import (List, Dict, Tuple, Callable, Optional, Union, Sequence,
                    Any, cast)
from functools import partial
from multiprocessing import Pool, Manager, cpu_count
import copy # For deep copying options/params
import traceback # For detailed error logging
import numpy as np
from scipy.linalg import svd
from scipy.stats import t as student_t

# Type definitions matching the expected structures
# Using nested Lists for input structures like JS for API fidelity
# Internal operations will convert to NumPy arrays where appropriate
DataType = Dict[str, List[np.ndarray]] # {'x': [np.array,...], 'y': ..., 'ye': ...}
ModelFunctionType = Callable[[np.ndarray, np.ndarray], np.ndarray]
ModelFunctionsType = List[List[ModelFunctionType]]
ParametersType = List[List[List[float]]] # Nested list structure for input/output
ParametersNpType = List[List[np.ndarray]] # Nested numpy arrays for internal use
ConstraintsType = List[List[List[Optional[Dict[str, float]]]]]
FixMapType = List[List[List[bool]]]
LinkMapType = List[List[List[Optional[Union[str, int]]]]]
OptionsType = Dict[str, Any]
ResultType = Dict[str, Any]

# --- Define default no-op functions for pickling ---
def _default_log_func(message: str, level: str) -> None:
    """Default no-op logging function."""
    pass # Do nothing

def _default_progress_func(progress_data: Dict[str, Any]) -> None:
    """Default no-op progress function."""
    pass # Do nothing
# ---

# Default Options (mirroring JS defaults)
DEFAULT_OPTIONS = {
    'calculateFittedModel': False,
    'constraintFunction': None,
    'constraints': None,
    'covarianceLambda': 1e-9,
    'epsilon': 1e-8,
    'errorTolerance': 1e-6,
    'fixMap': None,
    'gradientTolerance': 1e-6,
    'lambdaInitial': 1e-3,
    'lambdaIncreaseFactor': 10.0,
    'lambdaDecreaseFactor': 10.0,
    'linkMap': None,
    'logLevel': 'info',
    'onLog': _default_log_func, # Use the named function
    'onProgress': _default_progress_func, # Use the named function
    'maxIterations': 100,
    'robustCostFunction': None,
    'confidenceInterval': None,
    'numBootstrapSamples': 200,
    'bootstrapFallback': True,
    'calculateComponentModels': False,
    'model_x_range': None,
    'num_workers': None, # Add num_workers for bootstrap/independent control
}

# Logging levels
LOG_LEVELS = {'none': 0, 'error': 1, 'warn': 2, 'info': 3, 'debug': 4}
_MIN_ERROR_VALUE = 1e-12 # Minimum error value to avoid division by zero

# Helper class for parameter handling
class ParameterManager:
    """Manages parameter flattening, reconstruction, fixing, and linking."""
    def __init__(self, initial_params_struct: ParametersType, fix_map: Optional[FixMapType], link_map: Optional[LinkMapType]):
        self.initial_params_struct = initial_params_struct
        self.num_datasets = len(initial_params_struct)
        # Ensure param groups are numpy arrays for shape calculation
        self.param_shapes = [[np.asarray(pg).shape for pg in ds] for ds in initial_params_struct]
        self.total_params_flat = sum(sum(np.asarray(pg).size for pg in ds) for ds in initial_params_struct) # Use size for numpy

        self._param_meta = [] # List of dicts: {'ds': ds, 'pg': pg, 'pi': pi, 'initial': val, 'fixed': bool, 'link_id': id, 'master_idx': flat_idx | None, 'active_idx': active_idx | None}
        self._active_indices_flat = [] # Indices in the flattened full param list that are active
        self._active_map = {} # Maps flat_idx -> active_idx
        self._param_labels = [] # e.g., "ds0_pg1_pi0"

        self._parse_structure(fix_map, link_map)

    def _parse_structure(self, fix_map: Optional[FixMapType], link_map: Optional[LinkMapType]):
        flat_idx = 0
        active_idx_counter = 0
        link_masters = {} # link_id -> master_flat_idx

        for ds_idx, ds_params in enumerate(self.initial_params_struct):
            for pg_idx, param_group in enumerate(ds_params):
                # Ensure param_group is treated as a list/iterable of numbers
                param_group_list = list(np.asarray(param_group).ravel()) # Flatten just in case
                for pi_idx, initial_val in enumerate(param_group_list):
                    label = f"ds{ds_idx}_pg{pg_idx}_pi{pi_idx}"
                    self._param_labels.append(label)

                    is_fixed = False
                    try: # Safe access to potentially sparse maps
                        if fix_map and ds_idx < len(fix_map) and fix_map[ds_idx] and pg_idx < len(fix_map[ds_idx]) and fix_map[ds_idx][pg_idx] and pi_idx < len(fix_map[ds_idx][pg_idx]):
                             is_fixed = bool(fix_map[ds_idx][pg_idx][pi_idx])
                    except (IndexError, TypeError): pass # Ignore malformed fixMap parts

                    link_id = None
                    try: # Safe access
                        if link_map and ds_idx < len(link_map) and link_map[ds_idx] and pg_idx < len(link_map[ds_idx]) and link_map[ds_idx][pg_idx] and pi_idx < len(link_map[ds_idx][pg_idx]):
                            link_val = link_map[ds_idx][pg_idx][pi_idx]
                            if link_val is not None and str(link_val).strip() != '':
                                link_id = str(link_val).strip()
                    except (IndexError, TypeError): pass # Ignore malformed linkMap parts

                    meta = {
                        'ds': ds_idx, 'pg': pg_idx, 'pi': pi_idx,
                        'initial': float(initial_val), 'fixed': is_fixed,
                        'link_id': link_id, 'master_idx': None, 'active_idx': None,
                        'label': label
                    }

                    is_slave = False
                    if link_id is not None:
                        if link_id in link_masters:
                            master_flat_idx = link_masters[link_id]
                            # Check if master is fixed (use already processed meta)
                            if master_flat_idx < len(self._param_meta) and self._param_meta[master_flat_idx]['fixed']:
                                meta['fixed'] = True # Slave inherits fixed status from master
                            meta['master_idx'] = master_flat_idx
                            is_slave = True
                        else:
                            # This is the first time we see this link_id, it's the master (unless fixed)
                            if not is_fixed:
                                link_masters[link_id] = flat_idx
                            # If it *is* fixed, it cannot be a master for linking active params

                    if not meta['fixed'] and not is_slave: # Use meta['fixed'] as it might have been updated
                        # This parameter is active
                        meta['active_idx'] = active_idx_counter
                        self._active_indices_flat.append(flat_idx)
                        self._active_map[flat_idx] = active_idx_counter
                        active_idx_counter += 1
                    # No need for the elif check for fixed masters, as subsequent slaves will correctly point to the fixed master via master_idx

                    self._param_meta.append(meta)
                    flat_idx += 1

        self.num_active_params = len(self._active_indices_flat)
        self.active_param_labels = [self._param_meta[idx]['label'] for idx in self._active_indices_flat]


    def get_initial_active_params(self) -> np.ndarray:
        """Returns the initial values of only the active parameters."""
        if not self._active_indices_flat:
            return np.array([], dtype=float)
        return np.array([self._param_meta[idx]['initial'] for idx in self._active_indices_flat])

    def reconstruct(self, active_params: np.ndarray) -> ParametersNpType:
        """Reconstructs the full nested parameter structure from active parameters."""
        # Check length first
        if len(active_params) != self.num_active_params:
             raise ValueError(f"Input active_params length ({len(active_params)}) does not match expected number of active parameters ({self.num_active_params}).")

        full_params_flat = np.zeros(self.total_params_flat)

        for i, meta in enumerate(self._param_meta):
            current_active_idx = meta['active_idx'] # The index (0, 1, 2...) if active, else None

            if current_active_idx is not None:
                # It's an active parameter. Use its active_idx directly to index active_params.
                if 0 <= current_active_idx < len(active_params):
                   full_params_flat[i] = active_params[current_active_idx]
                else:
                    # This indicates an internal inconsistency in active_idx assignment during parsing
                    raise IndexError(f"Internal error: active_idx {current_active_idx} is out of bounds for input active_params vector (length {len(active_params)}). Check ParameterManager parsing.")

            elif meta['master_idx'] is not None:
                # It's a slave parameter
                master_flat_idx = meta['master_idx']
                if not (0 <= master_flat_idx < len(self._param_meta)):
                     raise IndexError(f"Internal error: Master index {master_flat_idx} out of bounds.")

                master_meta = self._param_meta[master_flat_idx]
                master_active_idx = master_meta['active_idx'] # Master's active index (if active)

                if master_active_idx is not None:
                     # Master is active. Use its active_idx to get value from input vector.
                     if 0 <= master_active_idx < len(active_params):
                         full_params_flat[i] = active_params[master_active_idx]
                     else:
                         # This indicates an internal inconsistency
                         raise IndexError(f"Internal error: master_active_idx {master_active_idx} is out of bounds for input active_params vector (length {len(active_params)}). Check ParameterManager parsing.")
                else:
                     # Master is fixed or another slave (points to fixed original)
                     full_params_flat[i] = master_meta['initial'] # Use master's initial (fixed) value

            else:
                # It's fixed and not a slave
                full_params_flat[i] = meta['initial']

        # Now reshape flat array back into nested structure of numpy arrays
        reconstructed_structure: ParametersNpType = []
        current_flat_idx = 0
        for ds_idx in range(self.num_datasets):
            ds_list = []
            for pg_idx in range(len(self.param_shapes[ds_idx])):
                # Use size for potentially multi-dimensional param groups if needed later
                num_params_in_group = np.prod(self.param_shapes[ds_idx][pg_idx]).item() # Get total elements
                original_shape = self.param_shapes[ds_idx][pg_idx]
                group_params_flat = full_params_flat[current_flat_idx : current_flat_idx + num_params_in_group]
                # Reshape back to original shape (important if params were > 1D)
                group_params = group_params_flat.reshape(original_shape)
                ds_list.append(group_params)
                current_flat_idx += num_params_in_group
            reconstructed_structure.append(ds_list)

        return reconstructed_structure

    # ... (keep get_active_indices, get_num_active, get_active_param_labels, get_full_param_labels) ...
    def get_active_indices(self) -> List[int]: return self._active_indices_flat
    def get_num_active(self) -> int: return self.num_active_params
    def get_active_param_labels(self) -> List[str]: return self.active_param_labels
    def get_full_param_labels(self) -> List[str]: return self._param_labels


# --- Core Fitting Logic Helpers ---

def _calculate_residuals_and_model(
    params_struct: ParametersNpType,
    data: DataType,
    model_functions: ModelFunctionsType,
    debug_iteration: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]], int]:
    """Calculates residuals, model values, component values, and total points."""
    residuals_list = []
    models_list = []
    component_models_list = []
    total_points = 0
    num_datasets = len(data['x'])

    for ds_idx in range(num_datasets):
        x_data = data['x'][ds_idx]
        y_data = data['y'][ds_idx]
        ye_data = data['ye'][ds_idx]
        n_pts = len(x_data)
        total_points += n_pts

        ds_model_y = np.zeros_like(y_data, dtype=float)
        ds_component_models = []

        if ds_idx >= len(model_functions) or ds_idx >= len(params_struct):
             raise ValueError(f"Data index {ds_idx} out of bounds for models/params.")
        param_groups = params_struct[ds_idx]
        model_groups = model_functions[ds_idx]
        if len(param_groups) != len(model_groups):
             raise ValueError(f"Mismatch param groups/model groups for dataset {ds_idx}.")

        for pg_idx, model_func in enumerate(model_groups):
            pg_params = param_groups[pg_idx]
            component_y = np.zeros_like(y_data, dtype=float)
            try:
                 # Vectorized call if model supports it, otherwise loop
                 # Assuming models take (params, x_array) and return y_array
                 # Check if model is likely vectorized (more robust checks possible)
                 is_vectorized = getattr(model_func, 'is_vectorized', False) # Optional flag
                 if is_vectorized:
                      component_y = model_func(pg_params, x_data)
                 else:
                      for i, x_point in enumerate(x_data):
                           res = model_func(pg_params, np.array([x_point]))
                           component_y[i] = res[0]

                 if not np.all(np.isfinite(component_y)):
                      non_finite_idx = np.where(~np.isfinite(component_y))[0]
                      raise ValueError(f"Non-finite value returned by model {pg_idx} at x={x_data[non_finite_idx[0]]}...")

            except Exception as e:
                 raise RuntimeError(f"Error evaluating model {pg_idx} for dataset {ds_idx}: {e}") from e

            ds_component_models.append(component_y)
            ds_model_y += component_y

        residuals = (y_data - ds_model_y) / ye_data
        if not np.all(np.isfinite(residuals)):
             non_finite_idx = np.where(~np.isfinite(residuals))[0]
             raise ValueError(f"Non-finite residual calculated for dataset {ds_idx} at index {non_finite_idx[0]}. y={y_data[non_finite_idx[0]]}, model={ds_model_y[non_finite_idx[0]]}, ye={ye_data[non_finite_idx[0]]}")

        residuals_list.append(residuals)
        models_list.append(ds_model_y)
        component_models_list.append(ds_component_models)

    return residuals_list, models_list, component_models_list, total_points


def _calculate_cost(residuals_list: List[np.ndarray], robust_cost_func: Optional[int]) -> float:
    """Calculates the cost (Chi-Squared or robust equivalent)."""
    if not residuals_list: return 0.0
    try:
        # Filter out potential empty arrays before concatenating
        valid_residuals = [r for r in residuals_list if isinstance(r, np.ndarray) and r.size > 0]
        if not valid_residuals: return 0.0 # Return 0 if all datasets were empty
        all_residuals = np.concatenate(valid_residuals)
    except ValueError as e:
        # Handle case where concatenation fails (e.g., inconsistent shapes if logic error upstream)
        print(f"Error concatenating residuals in _calculate_cost: {e}", file=sys.stderr)
        return np.inf # Return Inf to indicate failure

    cost = 0.0
    try:
        # Ensure residuals are finite before calculation
        if not np.all(np.isfinite(all_residuals)):
             warnings.warn("Non-finite values found in residuals during cost calculation.", RuntimeWarning)
             return np.nan # Return NaN if residuals are bad

        if robust_cost_func == 1:
            cost = np.sum(np.abs(all_residuals))
        elif robust_cost_func == 2:
            # Check for potential overflow before log
            term = 0.5 * all_residuals**2
            if np.any(term > 700): # Avoid exp overflow leading to log(inf)
                 warnings.warn("Large residuals encountered in Lorentzian cost function, may lead to Inf.", RuntimeWarning)
            term = 1.0 + term
            # Check for non-positive before log (shouldn't happen with term=1+0.5*res^2)
            if np.any(term <= 0): return np.nan
            cost = np.sum(np.log(term))
        else:
            cost = np.sum(all_residuals**2)

        # Final check if cost itself became non-finite (e.g., sum overflow)
        if not np.isfinite(cost):
             warnings.warn(f"Calculated cost is non-finite ({cost}).", RuntimeWarning)
             return np.nan

    except Exception as e:
         warnings.warn(f"Error during cost calculation: {e}", RuntimeWarning)
         return np.nan # Return NaN on any calculation error

    return cost

def _calculate_jacobian(
    param_manager: ParameterManager,
    current_params_active: np.ndarray,
    data: DataType,
    model_functions: ModelFunctionsType,
    epsilon: float,
    base_residuals_list: List[np.ndarray] # Residuals corresponding to current_params_active
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates Jacobian dr/dp using forward difference."""
    num_active_params = param_manager.get_num_active()
    base_residuals_flat = np.concatenate(base_residuals_list) if base_residuals_list else np.array([])
    total_points = len(base_residuals_flat)
    jacobian = np.zeros((total_points, num_active_params), dtype=float)

    if num_active_params == 0 or total_points == 0: return jacobian, base_residuals_flat

    params_perturbed_active = current_params_active.copy()

    for j in range(num_active_params): # Iterate through active parameters
        param_original_value = params_perturbed_active[j]
        h = epsilon * abs(param_original_value) if param_original_value != 0 else epsilon
        if h == 0: h = epsilon # Ensure h is non-zero if epsilon is non-zero

        params_perturbed_active[j] += h # Perturb one active parameter
        params_struct_perturbed = param_manager.reconstruct(params_perturbed_active)

        try:
            residuals_perturbed_list, _, _, _ = _calculate_residuals_and_model(
                params_struct_perturbed, data, model_functions
            )
            residuals_perturbed_flat = np.concatenate(residuals_perturbed_list) if residuals_perturbed_list else np.array([])

            # Forward difference: d(residual)/d(param) = (res(p+h) - res(p)) / h
            # Jacobian J_ij = -d(residual_i)/d(param_j) -> use -(res(p+h)-res(p))/h
            jacobian_col = -(residuals_perturbed_flat - base_residuals_flat) / h

            # Check for non-finite values in the calculated column
            if not np.all(np.isfinite(jacobian_col)):
                 warnings.warn(f"Non-finite values calculated in Jacobian column {j}. Setting column to zero.", RuntimeWarning)
                 jacobian[:, j] = 0.0
            else:
                 jacobian[:, j] = jacobian_col

        except Exception as e:
             warnings.warn(f"Could not evaluate Jacobian column {j} due to error: {e}. Setting column to zero.", RuntimeWarning)
             jacobian[:, j] = 0.0
        finally:
            params_perturbed_active[j] = param_original_value # Restore original value

    return jacobian, base_residuals_flat

def _apply_constraints(
    params_struct: ParametersNpType,
    constraints: Optional[ConstraintsType],
    constraint_func: Optional[Callable[[ParametersNpType], ParametersNpType]]
) -> ParametersNpType:
    """Applies box constraints and then the custom constraint function."""
    # Ensure input is mutable (it should be list of numpy arrays)
    params_constrained = [ [pg.copy() for pg in ds] for ds in params_struct]

    # 1. Box Constraints
    if constraints:
        # Basic structure check (optional but recommended)
        if len(constraints) == len(params_constrained):
            for ds_idx, ds_constraints in enumerate(constraints):
                if len(ds_constraints) == len(params_constrained[ds_idx]):
                    for pg_idx, pg_constraints in enumerate(ds_constraints):
                        if len(pg_constraints) == len(params_constrained[ds_idx][pg_idx]):
                            for pi_idx, constraint in enumerate(pg_constraints):
                                if constraint and isinstance(constraint, dict):
                                    val = params_constrained[ds_idx][pg_idx][pi_idx]
                                    min_val = constraint.get('min')
                                    max_val = constraint.get('max')
                                    if min_val is not None: val = max(val, min_val)
                                    if max_val is not None: val = min(val, max_val)
                                    params_constrained[ds_idx][pg_idx][pi_idx] = val
                        # else: Log warning about structure mismatch?

    # 2. Custom Constraint Function
    if constraint_func and callable(constraint_func):
        try:
            # Pass the potentially box-constrained parameters
            params_constrained = constraint_func(params_constrained)
            # TODO: Add validation that the returned structure matches expected?
        except Exception as e:
            warnings.warn(f"Error applying custom constraint function: {e}. Proceeding without custom constraint.", RuntimeWarning)

    return params_constrained

def _calculate_covariance_and_errors(
    jacobian: np.ndarray,
    residuals_flat: np.ndarray,
    num_active_params: int,
    total_points: int,
    cost_final: float,
    covar_lambda: float, # Regularization factor
    robust_cost_func: Optional[int],
    log_func: Callable[[str, str], None],
    log_level_threshold: int
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Calculates the covariance matrix and standard errors for active parameters.
    Returns (covariance_matrix, parameter_errors, scale_factor)
    """
    covariance_matrix = None
    parameter_errors = np.full(num_active_params, np.nan, dtype=float)
    scale_factor = 1.0 # aka reduced chi-squared for standard LS

    if num_active_params == 0 or total_points == 0:
        if log_level_threshold >= LOG_LEVELS['warn']:
             log_func("Covariance calculation skipped: No active parameters or data points.", 'warn')
        return covariance_matrix, parameter_errors, scale_factor

    degrees_of_freedom = total_points - num_active_params
    if degrees_of_freedom <= 0:
        if log_level_threshold >= LOG_LEVELS['warn']:
            log_func(f"Covariance calculation skipped: Degrees of freedom <= 0 ({total_points} points, {num_active_params} active params).", 'warn')
        return covariance_matrix, parameter_errors, scale_factor # Cannot compute reliably

    # Estimate scale factor (Reduced Chi-Squared for standard LS)
    # For robust cost funcs, this isn't strictly reduced chi-sq, but needed for error scaling
    scale_factor = cost_final / degrees_of_freedom

    try:
        # Calculate Hessian approximation: H = J^T * J
        hessian_approx = jacobian.T @ jacobian

        # Apply regularization BEFORE inversion for stability
        if covar_lambda > 0:
            hessian_approx += covar_lambda * np.diag(np.diag(hessian_approx))
            # Alternative: Add to identity: hessian_approx += covar_lambda * np.identity(num_active_params)

        # Calculate covariance matrix: inv(H) * scale_factor
        # Use SVD-based pseudoinverse for robustness against singularity
        try:
            u, s, vh = svd(hessian_approx)
            # Define threshold for singular values to consider zero
            threshold = np.finfo(float).eps * max(hessian_approx.shape) * s[0] # Based on numpy.linalg.pinv
            s_inv = np.array([1.0 / si if si > threshold else 0.0 for si in s])
            hessian_inv = vh.T @ np.diag(s_inv) @ u.T

            covariance_matrix = hessian_inv * scale_factor

            # Extract variances (diagonal elements)
            variances = np.diag(covariance_matrix)

            # Calculate standard errors: sqrt(abs(variance))
            # Use abs() and warn if variance is negative (as per JS implementation)
            negative_variance_indices = np.where(variances < 0)[0]
            if len(negative_variance_indices) > 0:
                if log_level_threshold >= LOG_LEVELS['warn']:
                    log_func(f"Warning: Negative variances encountered for active parameters at indices {negative_variance_indices}. Using absolute values for standard error calculation. This may indicate an ill-conditioned problem or poor fit.", 'warn')
                variances = np.abs(variances)

            # Check for non-finite variances before sqrt
            non_finite_variances = ~np.isfinite(variances)
            if np.any(non_finite_variances):
                 if log_level_threshold >= LOG_LEVELS['warn']:
                     log_func(f"Warning: Non-finite variances encountered at indices {np.where(non_finite_variances)[0]}. Setting corresponding standard errors to NaN.", 'warn')
                 variances[non_finite_variances] = np.nan # Propagate NaN

            parameter_errors = np.sqrt(variances)

        except np.linalg.LinAlgError as svd_err:
            if log_level_threshold >= LOG_LEVELS['error']:
                 log_func(f"SVD calculation failed during covariance matrix computation: {svd_err}. Cannot compute errors.", 'error')
            covariance_matrix = None # Indicate failure
            parameter_errors = np.full(num_active_params, np.nan, dtype=float)
        except Exception as e:
             if log_level_threshold >= LOG_LEVELS['error']:
                  log_func(f"Unexpected error during covariance matrix calculation: {e}. Cannot compute errors.", 'error')
             covariance_matrix = None
             parameter_errors = np.full(num_active_params, np.nan, dtype=float)


    except Exception as e:
        if log_level_threshold >= LOG_LEVELS['error']:
            log_func(f"Error during covariance matrix calculation: {e}", 'error')
        covariance_matrix = None # Ensure it's None on any failure
        parameter_errors = np.full(num_active_params, np.nan, dtype=float)

    return covariance_matrix, parameter_errors, scale_factor


def _reconstruct_full_errors(param_manager: ParameterManager, active_errors: np.ndarray) -> List[List[List[Optional[float]]]]:
    """Reconstructs the full error structure, filling NaN/0 for non-active."""
    full_errors_flat = np.full(param_manager.total_params_flat, np.nan) # Default to NaN
    active_idx_map = {meta['active_idx']: i for i, meta in enumerate(param_manager._param_meta) if meta['active_idx'] is not None}

    for i, meta in enumerate(param_manager._param_meta):
        if meta['active_idx'] is not None:
            # Active parameter
            active_err_idx = meta['active_idx']
            if active_err_idx < len(active_errors):
                 full_errors_flat[i] = active_errors[active_err_idx]
            # else: remains NaN (should not happen if active_errors is correct)
        elif meta['fixed']:
            # Fixed parameter
            full_errors_flat[i] = 0.0
        elif meta['master_idx'] is not None:
            # Slave parameter - inherit error from master if master is active
            master_meta = param_manager._param_meta[meta['master_idx']]
            if master_meta['active_idx'] is not None:
                 master_active_idx = master_meta['active_idx']
                 if master_active_idx < len(active_errors):
                     full_errors_flat[i] = active_errors[master_active_idx]
                 # else: master error is NaN, so slave error remains NaN
            else:
                 # Master is fixed, so slave error is 0
                 full_errors_flat[i] = 0.0
        # else: remains NaN (should not happen)


    # Reshape flat array back into nested list structure (matching ParametersType)
    reconstructed_structure = []
    current_flat_idx = 0
    for ds_idx in range(param_manager.num_datasets):
        ds_list = []
        for pg_idx in range(len(param_manager.param_shapes[ds_idx])):
            num_params_in_group = param_manager.param_shapes[ds_idx][pg_idx][0]
            group_errors = full_errors_flat[current_flat_idx : current_flat_idx + num_params_in_group]
            # Convert numpy types to standard Python floats/None for JSON compatibility etc.
            group_errors_list = [float(e) if np.isfinite(e) else (0.0 if e == 0 else None) for e in group_errors]
            ds_list.append(group_errors_list)
            current_flat_idx += num_params_in_group
        reconstructed_structure.append(ds_list)

    # Cast to expected output type structure
    final_structure: List[List[List[Optional[float]]]] = cast(List[List[List[Optional[float]]]], reconstructed_structure)
    return final_structure

def _determine_x_range(
    series_x_data: np.ndarray,
    model_x_range_series: Optional[Tuple[float, float]],
    series_index: int,
    log_func: Callable[[str, str], None],
    log_level_threshold: int
) -> Optional[Tuple[float, float]]:
    """Helper to determine the x-range for curve calculation."""
    x_min, x_max = None, None
    print(f"!!! ENTER _determine_x_range DS {series_index} with input range: {model_x_range_series}") # Log entry

    # 1. Try user-provided range first
    if model_x_range_series is not None:
        print(f"!!! DEBUG DS {series_index}: Provided range is not None.")
        if isinstance(model_x_range_series, (tuple, list)) and len(model_x_range_series) == 2:
            print(f"!!! DEBUG DS {series_index}: Provided range is tuple/list of 2.")
            try:
                min_r_raw = model_x_range_series[0]
                max_r_raw = model_x_range_series[1]
                print(f"!!! DEBUG DS {series_index}: Raw values [{min_r_raw}, {max_r_raw}]")
                min_r, max_r = float(min_r_raw), float(max_r_raw)
                print(f"!!! DEBUG DS {series_index}: Parsed float values [{min_r}, {max_r}]")

                # Check conditions individually
                cond1 = min_r <= max_r
                cond2 = np.isfinite(min_r)
                cond3 = np.isfinite(max_r)
                print(f"!!! DEBUG DS {series_index}: Checks: min<=max? {cond1}, isfinite(min)? {cond2}, isfinite(max)? {cond3}")

                if cond1 and cond2 and cond3:
                    x_min, x_max = min_r, max_r
                    print(f"!!! DEBUG DS {series_index}: SUCCESSFULLY set range from user input: [{x_min}, {x_max}]")
                    if log_level_threshold >= LOG_LEVELS['debug']:
                        log_func(f"Using provided x-range [{x_min:.3g}, {x_max:.3g}] for series {series_index} curve.", "debug")
                else: # Condition failed
                    print(f"!!! DEBUG DS {series_index}: User range validation FAILED.")
                    if log_level_threshold >= LOG_LEVELS['warn']:
                        log_func(f"Provided model_x_range for series {series_index} is invalid (min={min_r} > max={max_r} or non-finite). Falling back to data range.", "warn")
            except (ValueError, TypeError) as e: # Failed float conversion
                print(f"!!! DEBUG DS {series_index}: User range parsing FAILED (ValueError/TypeError: {e}).")
                if log_level_threshold >= LOG_LEVELS['warn']:
                    log_func(f"Provided model_x_range for series {series_index} is not numeric. Falling back to data range.", "warn")
        else: # Not a tuple/list of length 2
            print(f"!!! DEBUG DS {series_index}: User range type/length check FAILED.")
            if log_level_threshold >= LOG_LEVELS['warn']:
                log_func(f"Provided model_x_range for series {series_index} is not a tuple/list of two numbers. Falling back to data range.", "warn")

    # 2. Fallback to data range if needed
    if x_min is None or x_max is None:
        print(f"!!! DEBUG DS {series_index}: Falling back to data range.")
        if log_level_threshold >= LOG_LEVELS['debug']:
            log_func(f"Determining x-range from data for series {series_index} curve.", "debug")
        if series_x_data.size > 0:
            try:
                x_min_data = np.min(series_x_data)
                x_max_data = np.max(series_x_data)
                if np.isfinite(x_min_data) and np.isfinite(x_max_data):
                     x_min, x_max = x_min_data, x_max_data
                else:
                     if log_level_threshold >= LOG_LEVELS['error']:
                         log_func(f"Non-finite values found in x-data for series {series_index}. Cannot determine data range.", "error")
                     return None
            except (ValueError, TypeError): # Should be caught by np.isfinite, but belt-and-suspenders
                 if log_level_threshold >= LOG_LEVELS['error']:
                     log_func(f"Non-numeric X data found in series {series_index}. Cannot determine data range.", "error")
                 return None
        else:
            x_min, x_max = 0.0, 0.0 # Default for empty data

    if x_min is None or x_max is None: # Should only happen if data range failed
         if log_level_threshold >= LOG_LEVELS['error']:
             log_func(f"Could not determine valid x-range for series {series_index}.", "error")
         return None

        # --- ADD DEBUG ---
    if log_level_threshold >= LOG_LEVELS['debug']:
        log_func(f"==> _determine_x_range RETURNING for DS {series_index}: [{x_min}, {x_max}] (Input was: {model_x_range_series})", "debug")
        # --- END DEBUG ---
    print(f"!!! _determine_x_range FINAL for DS {series_index}: Returning [{x_min}, {x_max}]")
    print(f"!!! _determine_x_range FINAL for DS {series_index}: Returning [{x_min}, {x_max}] (Input was: {model_x_range_series})")    
    return x_min, x_max

def _generate_fitted_curves(
    params_struct: ParametersNpType,
    data: DataType,
    model_functions: ModelFunctionsType,
    calculate_option: Union[bool, Dict[str, int]],
    model_x_range_list: Optional[List[Optional[Tuple[float, float]]]],
    log_func: Callable[[str, str], None],
    log_level_threshold: int
) -> List[Dict[str, np.ndarray]]:
    """Generates smooth fitted curves for each dataset using the specified range."""
    fitted_curves = []
    num_datasets = len(data['x'])
    num_points = 200
    if isinstance(calculate_option, dict) and 'numPoints' in calculate_option:
        num_points = calculate_option['numPoints']

    for ds_idx in range(num_datasets):
        x_data = data['x'][ds_idx]
        current_series_range_opt = model_x_range_list[ds_idx] if model_x_range_list and len(model_x_range_list) > ds_idx else None
        x_range = _determine_x_range(x_data, current_series_range_opt, ds_idx, log_func, log_level_threshold)

        if x_range is None:
             warnings.warn(f"Could not determine valid x-range for fitted curve for dataset {ds_idx}. Skipping.", RuntimeWarning)
             fitted_curves.append({'x': np.array([]), 'y': np.array([])})
             continue
        x_min_plot, x_max_plot = x_range
        x_smooth = np.linspace(x_min_plot, x_max_plot, num_points)
        y_smooth = np.zeros_like(x_smooth, dtype=float)

        param_groups = params_struct[ds_idx]
        model_groups = model_functions[ds_idx]
        calc_ok = True

        for pg_idx, model_func in enumerate(model_groups):
            if not calc_ok: break
            pg_params = param_groups[pg_idx]
            try:
                 # --- Try vectorized call first ---
                 y_comp = model_func(pg_params, x_smooth)
                 if not np.all(np.isfinite(y_comp)): raise ValueError("Non-finite value in component")
                 y_smooth += y_comp
                 # --- End vectorized call ---
            except (TypeError, ValueError, IndexError): # Catch errors indicating non-vectorized or other issues
                 # --- Fallback to loop ---
                 warnings.warn(f"Model function {model_func.__name__} (ds={ds_idx}, pg={pg_idx}) might not be vectorized or failed. Falling back to loop.", RuntimeWarning)
                 try:
                      for i, x_point in enumerate(x_smooth):
                           res = model_func(pg_params, np.array([x_point]))
                           if not np.isfinite(res[0]): raise ValueError(f"Non-finite value at x={x_point}")
                           y_smooth[i] += res[0] # Add to existing sum
                 except Exception as e_loop:
                      warnings.warn(f"Could not generate fitted curve (loop fallback) for dataset {ds_idx}, component {pg_idx}: {e_loop}", RuntimeWarning)
                      y_smooth[:] = np.nan; calc_ok = False; break
                 # --- End Fallback ---
            except Exception as e_vec: # Catch other errors during vectorized call
                 warnings.warn(f"Could not generate fitted curve (vectorized) for dataset {ds_idx}, component {pg_idx}: {e_vec}", RuntimeWarning)
                 y_smooth[:] = np.nan; calc_ok = False; break

        fitted_curves.append({'x': x_smooth, 'y': y_smooth})

    return fitted_curves

def _generate_component_curves(
    params_struct: ParametersNpType,
    data: DataType,
    model_functions: ModelFunctionsType,
    calculate_fitted_model_option: Union[bool, Dict[str, int]],
    model_x_range_list: Optional[List[Optional[Tuple[float, float]]]],
    log_func: Callable[[str, str], None],
    log_level_threshold: int
) -> List[List[Dict[str, np.ndarray]]]:
    """Generates smooth curves for each component using the specified range."""
    component_curves_all_ds = []
    num_datasets = len(data['x'])
    num_points = 200
    if isinstance(calculate_fitted_model_option, dict) and 'numPoints' in calculate_fitted_model_option:
        num_points = calculate_fitted_model_option['numPoints']

    for ds_idx in range(num_datasets):
        ds_component_curves = []
        x_data = data['x'][ds_idx]
        current_series_range_opt = model_x_range_list[ds_idx] if model_x_range_list and len(model_x_range_list) > ds_idx else None
        x_range = _determine_x_range(x_data, current_series_range_opt, ds_idx, log_func, log_level_threshold)

        if x_range is None:
             warnings.warn(f"Could not determine valid x-range for component curves for dataset {ds_idx}. Skipping.", RuntimeWarning)
             num_models_expected = len(model_functions[ds_idx]) if ds_idx < len(model_functions) else 0
             component_curves_all_ds.append([{'x': np.array([]), 'y': np.array([])}] * num_models_expected)
             continue
        x_min_plot, x_max_plot = x_range
        x_smooth = np.linspace(x_min_plot, x_max_plot, num_points)

        param_groups = params_struct[ds_idx]
        model_groups = model_functions[ds_idx]

        for pg_idx, model_func in enumerate(model_groups):
            pg_params = param_groups[pg_idx]
            component_y = np.zeros_like(x_smooth, dtype=float) # Initialize component y
            calc_ok = True
            try:
                 # --- Try vectorized call first ---
                 component_y = model_func(pg_params, x_smooth)
                 if not np.all(np.isfinite(component_y)): raise ValueError("Non-finite value in component")
                 # --- End vectorized call ---
            except (TypeError, ValueError, IndexError): # Fallback to loop
                 warnings.warn(f"Model function {model_func.__name__} (ds={ds_idx}, pg={pg_idx}) might not be vectorized or failed. Falling back to loop for component curve.", RuntimeWarning)
                 try:
                      for i, x_point in enumerate(x_smooth):
                           res = model_func(pg_params, np.array([x_point]))
                           if not np.isfinite(res[0]): raise ValueError(f"Non-finite value at x={x_point}")
                           component_y[i] = res[0] # Assign component result directly
                 except Exception as e_loop:
                      warnings.warn(f"Could not generate component curve (loop fallback) for dataset {ds_idx}, component {pg_idx}: {e_loop}", RuntimeWarning)
                      component_y[:] = np.nan; calc_ok = False
            except Exception as e_vec: # Catch other errors during vectorized call
                 warnings.warn(f"Could not generate component curve (vectorized) for dataset {ds_idx}, component {pg_idx}: {e_vec}", RuntimeWarning)
                 component_y[:] = np.nan; calc_ok = False

            ds_component_curves.append({'x': x_smooth, 'y': component_y})
        component_curves_all_ds.append(ds_component_curves)

    return component_curves_all_ds

# --- NEW: Core LM Optimization Loop ---
def _run_lm_optimization_loop(
    param_manager: ParameterManager,
    initial_params_active: np.ndarray,
    data_np: DataType,
    model_function: ModelFunctionsType,
    options: OptionsType, # Pass the full options dict
    log_func: Callable[[str, str], None],
    progress_func: Callable[[Dict[str, Any]], None],
    log_level_threshold: int
) -> Tuple[np.ndarray, float, bool, int, Optional[np.ndarray], Optional[List[np.ndarray]], Optional[str]]:
    """
    Performs the core Levenberg-Marquardt iterations.

    Returns:
        Tuple: (final_active_params, final_cost, converged_status, iterations_done,
                final_jacobian, final_residuals_list, error_message)
    """
    params_active = initial_params_active.copy()
    num_active_params = param_manager.get_num_active()

    # Extract options needed for the loop
    lm_lambda = options['lambdaInitial']
    max_iterations = options['maxIterations']
    error_tolerance = options['errorTolerance']
    gradient_tolerance = options['gradientTolerance']
    epsilon = options['epsilon']
    lambda_increase = options['lambdaIncreaseFactor']
    lambda_decrease = options['lambdaDecreaseFactor']
    robust_cost = options['robustCostFunction']
    constraints = options['constraints']
    constraint_func = options['constraintFunction']

    converged = False
    iteration = 0
    error_message = None
    cost = np.inf # Cost of the currently accepted parameter set
    cost_prev = np.inf # Cost of the previously accepted parameter set
    final_jacobian = None
    final_residuals_list = None

    if log_level_threshold >= LOG_LEVELS['debug']:
        log_func(f"Starting LM loop. Initial active params: {params_active}", 'debug')

    while iteration < max_iterations:
        iteration += 1
        cost_at_iter_start = np.nan # Cost corresponding to params_active at start of iter

        try:
            # 1. Calculate current residuals and cost
            params_struct = param_manager.reconstruct(params_active)
            residuals_list, _, _, _ = _calculate_residuals_and_model(params_struct, data_np, model_function, debug_iteration=iteration)
            current_cost = _calculate_cost(residuals_list, robust_cost)
            cost_at_iter_start = current_cost # Store cost for current params

            if not np.isfinite(cost_at_iter_start):
                 error_message = "Cost function resulted in non-finite value (NaN or Inf)."
                 if log_level_threshold >= LOG_LEVELS['error']: log_func(error_message, 'error')
                 break

            # Update accepted cost if this is the first iteration or if the previous step was accepted
            if iteration == 1 or cost_at_iter_start < cost:
                 cost = cost_at_iter_start
                 final_residuals_list = residuals_list # Store residuals for accepted params

            if log_level_threshold >= LOG_LEVELS['debug']: log_func(f"Iteration {iteration}, Current Cost: {cost:.6e}, Lambda: {lm_lambda:.3e}", 'debug')

            # Call progress callback
            try:
                progress_func({
                     'iteration': iteration, 'chiSquared': cost, 'lambda': lm_lambda,
                     'activeParameters': params_active.copy()
                })
            except Exception as prog_e: warnings.warn(f"Error in onProgress callback: {prog_e}", RuntimeWarning)

            # 2. Check convergence (compare cost with cost_prev from *last accepted* step)
            if iteration > 1 and np.isfinite(cost_prev):
                 cost_change = abs(cost - cost_prev)
                 # Use relative tolerance check
                 if cost_change < error_tolerance * (1 + abs(cost_prev)):
                     if log_level_threshold >= LOG_LEVELS['info']: log_func(f"Converged: Chi-squared change ({cost_change:.2e}) below tolerance.", 'info')
                     converged = True; break

            # 3. Calculate Jacobian and Gradient
            if num_active_params > 0:
                 # Use residuals_list calculated at the start of this iteration
                 jacobian, residuals_flat = _calculate_jacobian(param_manager, params_active, data_np, model_function, epsilon, residuals_list)
                 final_jacobian = jacobian # Store Jacobian for accepted params

                 # Gradient: g = -J^T * r (Note: Jacobian already includes the minus sign from dr/dp)
                 # So, g = J^T @ r where J = -dr/dp
                 # Let's recalculate gradient based on standard LM formula: need -J^T r where J = d(model)/dp
                 # Our _calculate_jacobian returns dr/dp which is - (1/ye) * d(model)/dp
                 # So, J^T r = (dr/dp)^T r
                 # The LM update is (H + lambda D) dp = -g = J^T r
                 # Where H = J^T J = (dr/dp)^T (dr/dp)
                 # Let's stick to the Jacobian being dr/dp for now.
                 # Then H = J^T J
                 # And the RHS of the LM equation is -J^T r
                 gradient_lm = -jacobian.T @ residuals_flat # This is the RHS vector for LM update

                 # Check gradient convergence (norm of the gradient vector)
                 # Using max component norm as per original JS logic
                 max_grad_comp = np.max(np.abs(gradient_lm)) if len(gradient_lm)>0 else 0.0
                 if max_grad_comp < gradient_tolerance:
                     if log_level_threshold >= LOG_LEVELS['info']: log_func(f"Converged: Maximum gradient component ({max_grad_comp:.2e}) below tolerance.", 'info')
                     converged = True; break
            else: # No active parameters
                 if log_level_threshold >= LOG_LEVELS['info']: log_func("No active parameters to optimize. 'Converged'.", 'info')
                 converged = True; break

            # 4. Calculate Hessian Approximation and Parameter Step
            hessian_approx = jacobian.T @ jacobian # H = J^T J where J = dr/dp

            # Augment diagonal (Levenberg-Marquardt)
            diag_elements = np.diag(hessian_approx).copy()
            diag_elements[diag_elements <= 1e-12] = 1e-12 # Floor for damping
            damping_matrix = lm_lambda * np.diag(diag_elements)
            augmented_hessian = hessian_approx + damping_matrix

            try:
                # Solve (H + lambda D) dp = -J^T r = gradient_lm
                delta_p = np.linalg.solve(augmented_hessian, gradient_lm)
            except np.linalg.LinAlgError:
                 if log_level_threshold >= LOG_LEVELS['warn']: log_func(f"Iteration {iteration}: Linear solve failed. Trying pseudoinverse.", 'warn')
                 try: delta_p = np.linalg.pinv(augmented_hessian) @ gradient_lm
                 except np.linalg.LinAlgError:
                      error_message = f"Iteration {iteration}: Failed to solve for parameter step even with pseudoinverse."; break

            # 5. Evaluate prospective parameters and cost
            params_active_new = params_active + delta_p
            params_struct_new = param_manager.reconstruct(params_active_new)
            params_struct_new = _apply_constraints(params_struct_new, constraints, constraint_func)
            # Flatten back *only active* params after constraint
            temp_flat_all_new = np.concatenate([pg.ravel() for ds in params_struct_new for pg in ds])
            params_active_new = temp_flat_all_new[param_manager.get_active_indices()] if num_active_params > 0 else np.array([])

            residuals_new_list, _, _, _ = _calculate_residuals_and_model(params_struct_new, data_np, model_function, debug_iteration=iteration+0.5)
            cost_new = _calculate_cost(residuals_new_list, robust_cost)

            # 6. Accept or reject step, update lambda
            if cost_new < cost and np.isfinite(cost_new):
                if log_level_threshold >= LOG_LEVELS['debug']: log_func(f"Iteration {iteration}: Step accepted. Cost decreased from {cost:.6e} to {cost_new:.6e}", 'debug')
                cost_prev = cost # Store previously accepted cost
                params_active = params_active_new # Accept new parameters
                # cost = cost_new # Cost will be recalculated at start of next iter
                lm_lambda /= lambda_decrease
            else:
                if log_level_threshold >= LOG_LEVELS['debug']:
                     reason = "non-finite cost" if not np.isfinite(cost_new) else f"cost increased/stalled ({cost_new:.6e} >= {cost:.6e})"
                     log_func(f"Iteration {iteration}: Step rejected ({reason}). Increasing lambda.", 'debug')
                lm_lambda *= lambda_increase
                cost_prev = cost # Keep previous cost for convergence check
                if lm_lambda > 1e10: # Check for runaway lambda
                     if log_level_threshold >= LOG_LEVELS['warn']: log_func("Lambda reached maximum limit. Fit may be stalled.", 'warn')

        except Exception as e:
             error_message = f"Error during LM iteration {iteration}: {e}"
             if log_level_threshold >= LOG_LEVELS['error']:
                  log_func(error_message, 'error')
                  log_func(traceback.format_exc(), 'debug')
             break

    # --- End of Loop ---
    if error_message is None and iteration >= max_iterations and not converged:
         if log_level_threshold >= LOG_LEVELS['warn']:
              log_func(f"Fit did not converge within {max_iterations} iterations.", 'warn')

    # Return final state (use cost which holds the last accepted cost)
    return params_active, cost, converged, iteration, final_jacobian, final_residuals_list, error_message

# --- Bootstrap Worker and Orchestrator ---
def _bootstrap_worker(
    sample_index: int,
    # Arguments passed via functools.partial:
    original_data_np: DataType,
    model_function_all: ModelFunctionsType,
    original_fit_params_struct: ParametersNpType, # Use final params from original fit as guess
    options_all: OptionsType,
    param_manager_orig: ParameterManager, # Need structure info
    x_smooth_list_all: List[np.ndarray], # Pre-calculated smooth x-grids
    log_queue: Optional[queue.Queue] = None # Use standard queue for MP
) -> Tuple[int, Optional[List[Optional[np.ndarray]]]]: # Return list of optional arrays
    """
    Worker function for a single bootstrap sample fit.
    Fits resampled data and returns the fitted Y values on the smooth grid.
    """
    # Need to import numpy here if run in separate process
    import numpy as np
    import copy
    import time # For potential sleeps if queue fails

    worker_options = copy.deepcopy(options_all)
    num_datasets = len(original_data_np['x'])
    num_active_params = param_manager_orig.get_num_active()

    # Custom logging for this worker process
    log_level = worker_options.get('logLevel', 'info')
    log_level_threshold = LOG_LEVELS.get(log_level.lower(), LOG_LEVELS['info'])
    def worker_log(message, level):
         if LOG_LEVELS.get(level.lower(), LOG_LEVELS['none']) <= log_level_threshold:
             log_entry = (sample_index, message, level) # Use sample index for context
             if log_queue:
                  try: log_queue.put(log_entry)
                  except Exception as qe:
                       # Fallback print if queue fails in worker
                       print(f"[BS Worker {sample_index} QueueErr] {level.upper()}: {message} (QErr: {qe})", file=sys.stderr)


    # Set worker options
    worker_options['onLog'] = worker_log
    worker_options['onProgress'] = _default_progress_func # Disable progress
    worker_options['confidenceInterval'] = None # Disable nested CI
    worker_options['calculateFittedModel'] = False # Don't need curve dict
    worker_options['calculateComponentModels'] = False
    worker_options['logLevel'] = 'error' # Only log errors from sub-fits by default

    try:
        # 1. Create Resampled Data
        resampled_data_np: DataType = {'x': [], 'y': [], 'ye': []}
        for ds_idx in range(num_datasets):
            n_points = len(original_data_np['x'][ds_idx])
            if n_points == 0:
                 resampled_data_np['x'].append(np.array([])); resampled_data_np['y'].append(np.array([])); resampled_data_np['ye'].append(np.array([]))
                 continue
            # Use numpy's random choice for efficiency
            indices = np.random.choice(n_points, n_points, replace=True)
            resampled_data_np['x'].append(original_data_np['x'][ds_idx][indices])
            resampled_data_np['y'].append(original_data_np['y'][ds_idx][indices])
            resampled_ye = np.maximum(_MIN_ERROR_VALUE, original_data_np['ye'][ds_idx][indices])
            resampled_data_np['ye'].append(resampled_ye)

        # 2. Prepare Initial Guess (use final parameters from original fit)
        temp_flat_all_orig = np.concatenate([pg.ravel() for ds in original_fit_params_struct for pg in ds])
        initial_params_active_bs = temp_flat_all_orig[param_manager_orig.get_active_indices()] if num_active_params > 0 else np.array([])

        # 3. Run LM Optimization on Resampled Data
        bs_params_active, bs_cost, bs_converged, bs_iter, _, _, bs_error = _run_lm_optimization_loop(
            param_manager=param_manager_orig,
            initial_params_active=initial_params_active_bs,
            data_np=resampled_data_np,
            model_function=model_function_all,
            options=worker_options,
            log_func=worker_log,
            progress_func=_default_progress_func,
            log_level_threshold=log_level_threshold
        )

        if bs_error or not bs_converged:
             worker_log(f"Bootstrap sample {sample_index} fit failed or did not converge (Error: {bs_error}, Converged: {bs_converged}).", 'debug')
             return sample_index, None # Indicate failure

        # 4. Calculate Fitted Y values on the smooth grid
        bs_final_params_struct = param_manager_orig.reconstruct(bs_params_active)
        bs_fitted_y_smooth_list: List[Optional[np.ndarray]] = [] # Ensure type hint
        for ds_idx in range(num_datasets):
            x_smooth = x_smooth_list_all[ds_idx]
            if x_smooth is None or x_smooth.size == 0:
                 bs_fitted_y_smooth_list.append(None)
                 continue

            y_smooth = np.zeros_like(x_smooth, dtype=float)
            param_groups = bs_final_params_struct[ds_idx]
            model_groups = model_function_all[ds_idx]
            calc_ok = True
            for pg_idx, model_func in enumerate(model_groups):
                 if not calc_ok: break
                 pg_params = param_groups[pg_idx]
                 try: # Vectorized calculation if possible
                      y_comp = model_func(pg_params, x_smooth)
                      if not np.all(np.isfinite(y_comp)): raise ValueError("Non-finite value in component")
                      y_smooth += y_comp
                 except (TypeError, ValueError): # Fallback to loop if vectorized fails or model not designed for it
                      try:
                           for i, x_point_val in enumerate(x_smooth):
                                model_result = model_func(pg_params, np.array([x_point_val]))
                                if not np.isfinite(model_result[0]): raise ValueError("Non-finite value")
                                y_smooth[i] += model_result[0]
                      except Exception as eval_e:
                           worker_log(f"Bootstrap sample {sample_index}, ds={ds_idx}: Error evaluating fitted model on smooth grid: {eval_e}", 'warn')
                           y_smooth[:] = np.nan; calc_ok = False; break

            bs_fitted_y_smooth_list.append(y_smooth if calc_ok else None)

        return sample_index, bs_fitted_y_smooth_list

    except Exception as worker_e:
        worker_log(f"Unexpected error in bootstrap worker {sample_index}: {worker_e}", 'error')
        worker_log(traceback.format_exc(), 'debug')
        return sample_index, None

def _run_bootstrap_ci(
    param_manager: ParameterManager,
    final_params_active: np.ndarray,
    original_data_np: DataType,
    model_function_all: ModelFunctionsType,
    options: OptionsType, # Full options dict
    log_func: Callable[[str, str], None],
    log_level_threshold: int,
    x_smooth_list_all: List[Optional[np.ndarray]] # Use Optional list
) -> Tuple[Optional[List[Dict[str, np.ndarray]]], Optional[List[Dict[str, np.ndarray]]]]:
    """Orchestrates the bootstrap CI calculation using multiprocessing."""

    num_bootstrap_samples = options['numBootstrapSamples']
    confidence_level = options['confidenceInterval']
    num_datasets = len(original_data_np['x'])
    num_workers = options.get('num_workers', None)

    if log_level_threshold >= LOG_LEVELS['info']:
        log_func(f"Starting bootstrap CI calculation with {num_bootstrap_samples} samples...", 'info')

    # --- Setup ---
    # Stores collected y-values: collected_y[ds_idx][x_idx] = [y_samp1, y_samp2, ...]
    collected_y_values: List[Optional[List[List[float]]]] = []
    for ds_idx in range(num_datasets):
        x_smooth = x_smooth_list_all[ds_idx]
        if x_smooth is not None and x_smooth.size > 0:
             collected_y_values.append([[] for _ in range(len(x_smooth))])
        else:
             collected_y_values.append(None) # Mark dataset as skipped

    original_fit_params_struct = param_manager.reconstruct(final_params_active)
    successful_samples = 0

    # Determine number of workers
    max_workers = cpu_count()
    workers = max_workers if num_workers is None else max(1, min(num_workers, max_workers))
    use_parallel = workers > 1 and num_datasets > 0 # Also check num_datasets

    # --- Run Workers ---
    pool_results = [] # Store results outside context managers
    if use_parallel:
        with Manager() as manager:
            log_queue = manager.Queue()
            # Pass necessary arguments that are picklable
            worker_partial = partial(
                _bootstrap_worker,
                original_data_np=original_data_np,
                model_function_all=model_function_all,
                original_fit_params_struct=original_fit_params_struct,
                options_all=options, # Pass full options
                param_manager_orig=param_manager,
                x_smooth_list_all=x_smooth_list_all,
                log_queue=log_queue
            )
            with Pool(processes=workers) as pool:
                async_result = pool.map_async(worker_partial, range(num_bootstrap_samples))
                # Process logs while waiting
                while not async_result.ready():
                     try:
                         log_entry = log_queue.get(timeout=0.1)
                         s_idx, msg, level = log_entry
                         log_func(f"[BS Sample {s_idx}] {msg}", level)
                     except queue.Empty: pass
                     except Exception as q_err: print(f"Error processing BS log queue: {q_err}", file=sys.stderr); time.sleep(0.1)

                pool_results = async_result.get() # Get results list: [(idx, data), ...]

                # Drain remaining logs
                while not log_queue.empty():
                     try:
                         log_entry = log_queue.get_nowait()
                         s_idx, msg, level = log_entry
                         log_func(f"[BS Sample {s_idx}] {msg}", level)
                     except queue.Empty: break
                     except Exception as q_err: print(f"Error draining BS log queue: {q_err}", file=sys.stderr)

    else: # Sequential execution
        worker_partial = partial(
            _bootstrap_worker,
            original_data_np=original_data_np,
            model_function_all=model_function_all,
            original_fit_params_struct=original_fit_params_struct,
            options_all=options,
            param_manager_orig=param_manager,
            x_smooth_list_all=x_smooth_list_all,
            log_queue=None
        )
        for i in range(num_bootstrap_samples):
             if log_level_threshold >= LOG_LEVELS['debug'] and i % 50 == 0 and i > 0:
                  log_func(f"Bootstrap progress: {i}/{num_bootstrap_samples}", 'debug')
             pool_results.append(worker_partial(i)) # Append tuple (idx, data)

    # --- Collect Results ---
    for _, result_data in pool_results: # result_data is List[Optional[np.ndarray]]
         if result_data is not None:
             successful_samples += 1
             for ds_idx in range(num_datasets):
                  # Check if this dataset and x-point list exists and matches length
                  if collected_y_values[ds_idx] is not None and \
                     ds_idx < len(result_data) and \
                     result_data[ds_idx] is not None and \
                     len(result_data[ds_idx]) == len(collected_y_values[ds_idx]):

                       y_smooth_sample = result_data[ds_idx]
                       for x_idx in range(len(y_smooth_sample)):
                            if np.isfinite(y_smooth_sample[x_idx]):
                                 collected_y_values[ds_idx][x_idx].append(y_smooth_sample[x_idx])


    # --- Post-processing and Percentile Calculation ---
    if log_level_threshold >= LOG_LEVELS['info']:
        log_func(f"Bootstrap finished. Successful samples: {successful_samples}/{num_bootstrap_samples}", 'info')

    if successful_samples < min(10, 0.1 * num_bootstrap_samples):
        if log_level_threshold >= LOG_LEVELS['error']: log_func(f"Bootstrap Error: Very few successful samples ({successful_samples}). Cannot reliably calculate CI bands.", 'error')
        return None, None
    if successful_samples < 0.5 * num_bootstrap_samples:
        if log_level_threshold >= LOG_LEVELS['warn']: log_func("Bootstrap Warning: Less than 50% successful samples. Results may be less reliable.", 'warn')

    ci_lower_final = []
    ci_upper_final = []
    lower_quantile = (1.0 - confidence_level) / 2.0
    upper_quantile = 1.0 - lower_quantile

    for ds_idx in range(num_datasets):
        x_smooth = x_smooth_list_all[ds_idx]
        # Handle cases where x_smooth or collected values are None/empty
        if x_smooth is None or x_smooth.size == 0 or collected_y_values[ds_idx] is None:
             nan_curve = np.array([])
             ci_lower_final.append({'x': nan_curve, 'y': nan_curve})
             ci_upper_final.append({'x': nan_curve, 'y': nan_curve})
             continue

        num_x_points = len(x_smooth)
        y_lower = np.full(num_x_points, np.nan)
        y_upper = np.full(num_x_points, np.nan)

        for x_idx in range(num_x_points):
            y_dist = collected_y_values[ds_idx][x_idx]
            if len(y_dist) >= 2: # Need at least 2 points for percentile
                 try:
                     # Use numpy percentile for potentially better handling of edge cases
                     y_lower[x_idx] = np.percentile(y_dist, lower_quantile * 100.0)
                     y_upper[x_idx] = np.percentile(y_dist, upper_quantile * 100.0)
                 except IndexError: # Should not happen with np.percentile but safety
                      pass # Keep NaN
            # else: remains NaN

        ci_lower_final.append({'x': x_smooth, 'y': y_lower})
        ci_upper_final.append({'x': x_smooth, 'y': y_upper})

    return ci_lower_final, ci_upper_final

def _calculate_confidence_intervals(
        param_manager: ParameterManager,
        final_params_active: np.ndarray,
        jacobian: Optional[np.ndarray],  # Can be None
        covariance_matrix: Optional[np.ndarray],
        data: DataType,
        model_functions: ModelFunctionsType,
        confidence_level: float,
        total_points: int,
        num_active_params: int,
        calculate_fitted_model_option: Union[bool, Dict[str, int]],
        log_func: Callable[[str, str], None],
        log_level_threshold: int,
        bootstrap_fallback: bool,
        num_bootstrap_samples: int,
        base_residuals_list: Optional[List[np.ndarray]],  # Can be None
        model_x_range_list: Optional[List[Optional[Tuple[float, float]]]],
        options: OptionsType  # Pass full options dict
    ) -> Tuple[Optional[List[Dict[str, np.ndarray]]], Optional[List[Dict[str, np.ndarray]]]]:
    """
    Calculates confidence intervals for the fitted model curves.
    Uses Delta method (via Jacobian & Covariance) or Bootstrap fallback.
    """
    ci_lower_list = None
    ci_upper_list = None
    num_datasets = len(data['x'])
    degrees_of_freedom = total_points - num_active_params

    if degrees_of_freedom <= 0:
        if log_level_threshold >= LOG_LEVELS['warn']:
            log_func("CI skipped: Degrees of freedom <= 0.", 'warn')
        return None, None

    # --- Get Smooth X points ---
    x_smooth_list_all: List[Optional[np.ndarray]] = [] # Ensure correct type hint
    valid_ranges_found = True
    num_points_curve = 200 # Default
    if isinstance(calculate_fitted_model_option, dict) and 'numPoints' in calculate_fitted_model_option:
        num_points_curve = calculate_fitted_model_option['numPoints']

    for ds_idx in range(num_datasets):
        x_data = data['x'][ds_idx]
        current_series_range = model_x_range_list[ds_idx] if model_x_range_list and len(model_x_range_list) > ds_idx else None
        x_range = _determine_x_range(x_data, current_series_range, ds_idx, log_func, log_level_threshold)
        if x_range is None:
            x_smooth_list_all.append(None) # Use None placeholder
            valid_ranges_found = False
        else:
            x_min, x_max = x_range

            current_x_smooth = np.linspace(x_min, x_max, num_points_curve)
            # --- ADD DEBUG ---
            if log_level_threshold >= LOG_LEVELS['debug']:
                log_func(f"==> _calculate_confidence_intervals DS {ds_idx}: x_smooth range [{current_x_smooth[0]:.3g}, {current_x_smooth[-1]:.3g}], num_points={len(current_x_smooth)}", "debug")
            # --- END DEBUG ---
            print(f"!!! _calculate_confidence_intervals DS {ds_idx}: Generated x_smooth from {current_x_smooth[0]} to {current_x_smooth[-1]}")

            x_smooth_list_all.append(np.linspace(x_min, x_max, num_points_curve))

    if not valid_ranges_found and not bootstrap_fallback: # If any range failed, standard CI cannot proceed
         warnings.warn("Cannot calculate standard CI because x-range determination failed for at least one dataset.", RuntimeWarning)
         return None, None
    # If bootstrap is enabled, it might still work for datasets with valid ranges

    # --- Decide Calculation Method ---
    use_bootstrap = False
    reason_for_bootstrap = ""
    # Check covariance matrix validity *only if* standard method is possible
    if covariance_matrix is None:
        reason_for_bootstrap = "Covariance matrix is None"
        use_bootstrap = True
    elif np.any(~np.isfinite(covariance_matrix)):
         reason_for_bootstrap = "Covariance matrix contains non-finite values"
         use_bootstrap = True
    elif num_active_params > 0: # Avoid diag if no active params
        variances = np.diag(covariance_matrix)
        if np.any(variances < 0):
            reason_for_bootstrap = "Negative variances found in covariance matrix"
            use_bootstrap = True

    if use_bootstrap:
        if log_level_threshold >= LOG_LEVELS['warn']:
            log_func(f"Standard CI calculation not possible ({reason_for_bootstrap}).", 'warn')
        if bootstrap_fallback:
            if log_level_threshold >= LOG_LEVELS['info']:
                log_func("Attempting bootstrap fallback for CIs.", 'info')
            try:
                 if base_residuals_list is None:
                      raise RuntimeError("Cannot run bootstrap: final residuals not available.")

                 # --- FIX: Pass x_smooth_list_all to bootstrap ---
                 ci_lower_list, ci_upper_list = _run_bootstrap_ci(
                     param_manager, final_params_active, data, model_functions,
                     options, # Pass the full options dict
                     log_func, log_level_threshold,
                     x_smooth_list_all # <-- PASS THE CALCULATED LIST HERE
                 )
                 # --- END FIX ---

            except Exception as bs_e:
                 if log_level_threshold >= LOG_LEVELS['error']:
                      log_func(f"Bootstrap CI calculation failed: {bs_e}", 'error')
                      log_func(traceback.format_exc(), 'debug')
                 ci_lower_list, ci_upper_list = None, None
        else: # Bootstrap fallback disabled
            if log_level_threshold >= LOG_LEVELS['warn']:
                log_func("Bootstrap fallback disabled, confidence intervals cannot be calculated.", 'warn')
            return None, None
    else: # Standard CI calculation
        # ... (Standard CI logic as implemented previously, using x_smooth_list_all[ds_idx]) ...
        if log_level_threshold >= LOG_LEVELS['debug']:
            log_func("Calculating confidence intervals using Delta method.", 'debug')
        ci_lower_list = []
        ci_upper_list = []
        alpha = 1.0 - confidence_level
        t_crit = abs(student_t.ppf(alpha / 2.0, df=degrees_of_freedom))
        final_params_struct = param_manager.reconstruct(final_params_active)

        for ds_idx in range(num_datasets):
            x_smooth = x_smooth_list_all[ds_idx] # Get pre-calculated grid
            if x_smooth is None: # Skip if range determination failed for this dataset
                nan_curve = np.array([])
                ci_lower_list.append({'x': nan_curve, 'y': nan_curve})
                ci_upper_list.append({'x': nan_curve, 'y': nan_curve})
                continue

            # Initialize arrays for this dataset
            y_model_smooth = np.zeros_like(x_smooth, dtype=float)
            jacobian_model_smooth = np.zeros((len(x_smooth), num_active_params), dtype=float)
            calculation_ok_for_ds = True

            param_groups = final_params_struct[ds_idx]
            model_groups = model_functions[ds_idx]

            # Calculate model and Jacobian at smooth x points
            for pg_idx, model_func in enumerate(model_groups):
                if not calculation_ok_for_ds: break
                pg_params = param_groups[pg_idx]
                epsilon = options['epsilon']

                for i, x_point_val in enumerate(x_smooth):
                    x_point_arr = np.array([x_point_val])
                    try:
                        # Evaluate base model value for this component
                        model_result = model_func(pg_params, x_point_arr)
                        current_y_comp = model_result[0]
                        if not np.isfinite(current_y_comp): raise ValueError("Model returned non-finite value")
                        y_model_smooth[i] += current_y_comp

                        # Calculate Jacobian of this component w.r.t *its own* params
                        jac_comp_local = np.zeros(len(pg_params))
                        params_perturbed_local = pg_params.copy()
                        for p_local_idx in range(len(pg_params)):
                            p_orig = params_perturbed_local[p_local_idx]
                            h = epsilon * abs(p_orig) if p_orig != 0 else epsilon
                            if h == 0: h = epsilon # Ensure non-zero step
                            params_perturbed_local[p_local_idx] += h
                            model_perturbed = model_func(params_perturbed_local, x_point_arr)
                            if not np.isfinite(model_perturbed[0]): raise ValueError("Perturbed model returned non-finite value")
                            jac_comp_local[p_local_idx] = (model_perturbed[0] - current_y_comp) / h
                            params_perturbed_local[p_local_idx] = p_orig # Restore

                        # Map local Jacobian contributions to the global active parameter Jacobian
                        flat_start_idx = sum(sum(np.asarray(pg).size for pg in final_params_struct[ds]) for ds in range(ds_idx)) + \
                                         sum(np.asarray(pg).size for pg in final_params_struct[ds_idx][:pg_idx])

                        for p_local_idx in range(len(pg_params)):
                            flat_param_idx = flat_start_idx + p_local_idx
                            meta = param_manager._param_meta[flat_param_idx]
                            current_active_idx = meta['active_idx']
                            master_active_idx = None
                            if meta['master_idx'] is not None:
                                 master_meta = param_manager._param_meta[meta['master_idx']]
                                 master_active_idx = master_meta['active_idx']

                            # Check if Jacobian value is finite before adding
                            jac_val_local = jac_comp_local[p_local_idx]
                            if not np.isfinite(jac_val_local):
                                 warnings.warn(f"Non-finite local Jacobian value encountered for CI at ds={ds_idx}, pg={pg_idx}, pi={p_local_idx}, x={x_point_val}. Setting contribution to 0.", RuntimeWarning)
                                 jac_val_local = 0.0 # Avoid propagating NaN/Inf

                            if current_active_idx is not None:
                                jacobian_model_smooth[i, current_active_idx] += jac_val_local
                            elif master_active_idx is not None:
                                jacobian_model_smooth[i, master_active_idx] += jac_val_local

                    except Exception as e:
                        warnings.warn(f"Could not evaluate model or Jacobian for CI at ds={ds_idx}, pg={pg_idx}, x={x_point_val}: {e}. Skipping CI for this dataset.", RuntimeWarning)
                        calculation_ok_for_ds = False; break # Stop processing points/components for this dataset

            # Calculate variance and CI bounds only if calculation succeeded
            if calculation_ok_for_ds:
                # Check covariance_matrix again (should be defined if not use_bootstrap)
                if covariance_matrix is not None and np.all(np.isfinite(jacobian_model_smooth)):
                     try:
                         if not np.all(np.isfinite(covariance_matrix)): raise ValueError("Covariance matrix contains non-finite values.")
                         jc = jacobian_model_smooth @ covariance_matrix
                         model_variance = np.sum(jc * jacobian_model_smooth, axis=1)
                         negative_var_mask = model_variance < 0
                         if np.any(negative_var_mask):
                             warnings.warn(f"Negative variances encountered during CI calculation for dataset {ds_idx}. Setting to zero.", RuntimeWarning)
                             model_variance[negative_var_mask] = 0
                         model_std_err = np.sqrt(model_variance)
                         delta_y = t_crit * model_std_err
                         ci_lower_list.append({'x': x_smooth, 'y': y_model_smooth - delta_y})
                         ci_upper_list.append({'x': x_smooth, 'y': y_model_smooth + delta_y})
                     except Exception as e:
                         warnings.warn(f"Could not calculate CI variance for dataset {ds_idx}: {e}", RuntimeWarning)
                         nan_curve = np.full_like(x_smooth, np.nan)
                         ci_lower_list.append({'x': x_smooth, 'y': nan_curve})
                         ci_upper_list.append({'x': x_smooth, 'y': nan_curve})
                else: # Covariance matrix None or Jacobian has NaNs
                     reason = "covariance matrix is None" if covariance_matrix is None else "Jacobian has non-finite values"
                     warnings.warn(f"Could not calculate CI variance for dataset {ds_idx} because {reason}.", RuntimeWarning)
                     nan_curve = np.full_like(x_smooth, np.nan)
                     ci_lower_list.append({'x': x_smooth, 'y': nan_curve})
                     ci_upper_list.append({'x': x_smooth, 'y': nan_curve})
            else: # Calculation failed for this dataset
                 nan_curve = np.full_like(x_smooth, np.nan) if x_smooth is not None else np.array([])
                 ci_lower_list.append({'x': x_smooth if x_smooth is not None else np.array([]), 'y': nan_curve})
                 ci_upper_list.append({'x': x_smooth if x_smooth is not None else np.array([]), 'y': nan_curve})

    return ci_lower_list, ci_upper_list

def _bootstrap_worker(
    sample_index: int,
    original_data_np: DataType,
    model_function_all: ModelFunctionsType,
    original_fit_params_struct: ParametersNpType,
    options_all: OptionsType,
    param_manager_orig: ParameterManager,
    x_smooth_list_all: List[np.ndarray],
    log_queue: Optional[queue.Queue] = None
) -> Tuple[int, Optional[List[np.ndarray]]]:
    """
    Worker function for a single bootstrap sample fit.
    Fits resampled data and returns the fitted Y values on the smooth grid.
    """
    import copy
    worker_options = copy.deepcopy(options_all)
    num_datasets = len(original_data_np['x'])
    num_active_params = param_manager_orig.get_num_active()

    # Custom logging for this worker process
    log_level = worker_options.get('logLevel', 'info')
    log_level_threshold = LOG_LEVELS.get(log_level.lower(), LOG_LEVELS['info'])

    def worker_log(message, level):
        if LOG_LEVELS.get(level.lower(), LOG_LEVELS['none']) <= log_level_threshold:
            log_entry = (sample_index, message, level)
            if log_queue:
                try:
                    log_queue.put(log_entry)
                except Exception:
                    pass  # Ignore queue errors in worker

    worker_options['onLog'] = worker_log
    worker_options['onProgress'] = _default_progress_func  # Disable progress for bootstrap fits
    worker_options['confidenceInterval'] = None  # Disable nested CI calc
    worker_options['calculateFittedModel'] = False  # Don't need full curve dict from sub-fit
    worker_options['calculateComponentModels'] = False
    worker_options['logLevel'] = 'error'  # Only log errors from sub-fits by default

    try:
        # 1. Create Resampled Data
        resampled_data_np: DataType = {'x': [], 'y': [], 'ye': []}
        for ds_idx in range(num_datasets):
            n_points = len(original_data_np['x'][ds_idx])
            if n_points == 0:
                resampled_data_np['x'].append(np.array([]))
                resampled_data_np['y'].append(np.array([]))
                resampled_data_np['ye'].append(np.array([]))
                continue
            indices = np.random.choice(n_points, n_points, replace=True)
            resampled_data_np['x'].append(original_data_np['x'][ds_idx][indices])
            resampled_data_np['y'].append(original_data_np['y'][ds_idx][indices])
            resampled_ye = np.maximum(1e-12, original_data_np['ye'][ds_idx][indices])  # Ensure positive errors
            resampled_data_np['ye'].append(resampled_ye)

        # 2. Prepare Initial Guess
        temp_flat_all_orig = np.concatenate([pg.ravel() for ds in original_fit_params_struct for pg in ds])
        initial_params_active_bs = temp_flat_all_orig[param_manager_orig.get_active_indices()] if num_active_params > 0 else np.array([])

        # 3. Run LM Optimization on Resampled Data
        bs_params_active, bs_cost, bs_converged, bs_iter, _, _, bs_error = _run_lm_optimization_loop(
            param_manager=param_manager_orig,
            initial_params_active=initial_params_active_bs,
            data_np=resampled_data_np,
            model_function=model_function_all,
            options=worker_options,
            log_func=worker_log,
            progress_func=_default_progress_func,
            log_level_threshold=log_level_threshold
        )

        if bs_error or not bs_converged:
            worker_log(f"Bootstrap sample {sample_index} fit failed or did not converge (Error: {bs_error}, Converged: {bs_converged}).", 'debug')
            return sample_index, None

        # 4. Calculate Fitted Y values on the smooth grid
        bs_final_params_struct = param_manager_orig.reconstruct(bs_params_active)
        bs_fitted_y_smooth_list = []
        for ds_idx in range(num_datasets):
            x_smooth = x_smooth_list_all[ds_idx]
            if x_smooth is None:
                bs_fitted_y_smooth_list.append(None)
                continue

            y_smooth = np.zeros_like(x_smooth, dtype=float)
            param_groups = bs_final_params_struct[ds_idx]
            model_groups = model_function_all[ds_idx]
            calc_ok = True
            for pg_idx, model_func in enumerate(model_groups):
                if not calc_ok:
                    break
                pg_params = param_groups[pg_idx]
                for i, x_point_val in enumerate(x_smooth):
                    try:
                        model_result = model_func(pg_params, np.array([x_point_val]))
                        if not np.isfinite(model_result[0]):
                            raise ValueError("Non-finite model value")
                        y_smooth[i] += model_result[0]
                    except Exception as eval_e:
                        worker_log(f"Bootstrap sample {sample_index}, ds={ds_idx}: Error evaluating fitted model on smooth grid at x={x_point_val}: {eval_e}", 'warn')
                        y_smooth[:] = np.nan
                        calc_ok = False
                        break
            bs_fitted_y_smooth_list.append(y_smooth if calc_ok else None)

        return sample_index, bs_fitted_y_smooth_list

    except Exception as worker_e:
        worker_log(f"Unexpected error in bootstrap worker {sample_index}: {worker_e}", 'error')
        import traceback
        worker_log(traceback.format_exc(), 'debug')
        return sample_index, None

def _run_bootstrap_ci(
    param_manager: ParameterManager,
    final_params_active: np.ndarray,
    original_data_np: DataType,
    model_function_all: ModelFunctionsType,
    options: OptionsType,
    log_func: Callable[[str, str], None],
    log_level_threshold: int,
    x_smooth_list_all: List[np.ndarray]
) -> Tuple[Optional[List[Dict[str, np.ndarray]]], Optional[List[Dict[str, np.ndarray]]]]:
    """Orchestrates the bootstrap CI calculation using multiprocessing."""
    num_bootstrap_samples = options['numBootstrapSamples']
    confidence_level = options['confidenceInterval']
    num_datasets = len(original_data_np['x'])
    num_workers = options.get('num_workers', None)

    if log_level_threshold >= LOG_LEVELS['info']:
        log_func(f"Starting bootstrap CI calculation with {num_bootstrap_samples} samples...", 'info')

    # --- Setup ---
    collected_y_values: List[Optional[List[List[float]]]] = []
    for ds_idx in range(num_datasets):
        x_smooth = x_smooth_list_all[ds_idx]
        if x_smooth is not None:
            collected_y_values.append([[] for _ in range(len(x_smooth))])
        else:
            collected_y_values.append(None)

    original_fit_params_struct = param_manager.reconstruct(final_params_active)
    successful_samples = 0

    # Determine number of workers
    max_workers = cpu_count()
    workers = max_workers if num_workers is None else max(1, min(num_workers, max_workers))
    use_parallel = workers > 1

    # --- Run Workers ---
    if use_parallel:
        with Manager() as manager:
            log_queue = manager.Queue()
            pool = Pool(processes=workers)
            worker_partial = partial(
                _bootstrap_worker,
                original_data_np=original_data_np,
                model_function_all=model_function_all,
                original_fit_params_struct=original_fit_params_struct,
                options_all=options,
                param_manager_orig=param_manager,
                x_smooth_list_all=x_smooth_list_all,
                log_queue=log_queue
            )
            async_result = pool.map_async(worker_partial, range(num_bootstrap_samples))

            # Process logs while waiting
            while not async_result.ready():
                try:
                    log_entry = log_queue.get(timeout=0.1)
                    s_idx, msg, level = log_entry
                    log_func(f"[BS Sample {s_idx}] {msg}", level)
                except queue.Empty:
                    pass
                except Exception as q_err:
                    print(f"Error processing BS log queue: {q_err}", file=sys.stderr)

            pool_results = async_result.get()

            # Drain remaining logs
            while not log_queue.empty():
                try:
                    log_entry = log_queue.get_nowait()
                    s_idx, msg, level = log_entry
                    log_func(f"[BS Sample {s_idx}] {msg}", level)
                except queue.Empty:
                    break
                except Exception as q_err:
                    print(f"Error draining BS log queue: {q_err}", file=sys.stderr)

            pool.close()
            pool.join()

            # Collect results
            for _, result_data in pool_results:
                if result_data is not None:
                    successful_samples += 1
                    for ds_idx in range(num_datasets):
                        if collected_y_values[ds_idx] is not None and result_data[ds_idx] is not None:
                            y_smooth_sample = result_data[ds_idx]
                            for x_idx, y_val in enumerate(y_smooth_sample):
                                if np.isfinite(y_val):
                                    collected_y_values[ds_idx][x_idx].append(y_val)

    else:  # Sequential execution
        worker_partial = partial(
            _bootstrap_worker,
            original_data_np=original_data_np,
            model_function_all=model_function_all,
            original_fit_params_struct=original_fit_params_struct,
            options_all=options,
            param_manager_orig=param_manager,
            x_smooth_list_all=x_smooth_list_all,
            log_queue=None
        )
        for i in range(num_bootstrap_samples):
            if log_level_threshold >= LOG_LEVELS['debug'] and i % 50 == 0 and i > 0:
                log_func(f"Bootstrap progress: {i}/{num_bootstrap_samples}", 'debug')
            _, result_data = worker_partial(i)
            if result_data is not None:
                successful_samples += 1
                for ds_idx in range(num_datasets):
                    if collected_y_values[ds_idx] is not None and result_data[ds_idx] is not None:
                        y_smooth_sample = result_data[ds_idx]
                        for x_idx, y_val in enumerate(y_smooth_sample):
                            if np.isfinite(y_val):
                                collected_y_values[ds_idx][x_idx].append(y_val)

    # --- Post-processing and Percentile Calculation ---
    if log_level_threshold >= LOG_LEVELS['info']:
        log_func(f"Bootstrap finished. Successful samples: {successful_samples}/{num_bootstrap_samples}", 'info')

    if successful_samples < min(10, 0.1 * num_bootstrap_samples):
        if log_level_threshold >= LOG_LEVELS['error']:
            log_func(f"Bootstrap Error: Very few successful samples ({successful_samples}). Cannot reliably calculate CI bands.", 'error')
        return None, None

    if successful_samples < 0.5 * num_bootstrap_samples:
        if log_level_threshold >= LOG_LEVELS['warn']:
            log_func("Bootstrap Warning: Less than 50% of bootstrap samples were successful. Results may be less reliable.", 'warn')

    ci_lower_final = []
    ci_upper_final = []
    lower_quantile = (1.0 - confidence_level) / 2.0
    upper_quantile = 1.0 - lower_quantile

    for ds_idx in range(num_datasets):
        x_smooth = x_smooth_list_all[ds_idx]
        if x_smooth is None or collected_y_values[ds_idx] is None:
            nan_curve = np.array([])
            ci_lower_final.append({'x': nan_curve, 'y': nan_curve})
            ci_upper_final.append({'x': nan_curve, 'y': nan_curve})
            continue

        num_x_points = len(x_smooth)
        y_lower = np.full(num_x_points, np.nan)
        y_upper = np.full(num_x_points, np.nan)

        for x_idx in range(num_x_points):
            y_dist = collected_y_values[ds_idx][x_idx]
            if len(y_dist) >= 2:
                y_lower[x_idx] = np.percentile(y_dist, lower_quantile * 100.0)
                y_upper[x_idx] = np.percentile(y_dist, upper_quantile * 100.0)

        ci_lower_final.append({'x': x_smooth, 'y': y_lower})
        ci_upper_final.append({'x': x_smooth, 'y': y_upper})

    return ci_lower_final, ci_upper_final

# --- Main API Functions ---

def _run_lm_optimization_loop(
    param_manager: ParameterManager,
    initial_params_active: np.ndarray,
    data_np: DataType,
    model_function: ModelFunctionsType,
    options: OptionsType,  # Pass the full options dict
    log_func: Callable[[str, str], None],
    progress_func: Callable[[Dict[str, Any]], None],
    log_level_threshold: int
) -> Tuple[np.ndarray, float, bool, int, Optional[np.ndarray], Optional[List[np.ndarray]], Optional[str]]:
    """
    Performs the core Levenberg-Marquardt iterations.

    Returns:
        Tuple: (final_active_params, final_cost, converged_status, iterations_done,
                final_jacobian, final_residuals_list, error_message)
    """
    # ...existing code...
    params_active = initial_params_active.copy()
    num_active_params = param_manager.get_num_active()

    # Extract options needed for the loop
    lm_lambda = options['lambdaInitial']
    max_iterations = options['maxIterations']
    error_tolerance = options['errorTolerance']
    gradient_tolerance = options['gradientTolerance']
    epsilon = options['epsilon']
    lambda_increase = options['lambdaIncreaseFactor']
    lambda_decrease = options['lambdaDecreaseFactor']
    robust_cost = options['robustCostFunction']
    constraints = options['constraints']
    constraint_func = options['constraintFunction']

    converged = False
    iteration = 0
    error_message = None
    cost = np.inf
    cost_prev = np.inf
    final_jacobian = None
    final_residuals_list = None

    if log_level_threshold >= LOG_LEVELS['debug']:
        log_func(f"Starting LM loop. Initial active params: {params_active}", 'debug')

    while iteration < max_iterations:
        iteration += 1
        cost_at_iter_start = np.nan  # Cost corresponding to params_active at start of iter

        try:
            # 1. Calculate current residuals and cost
            params_struct = param_manager.reconstruct(params_active)
            residuals_list, _, _, _ = _calculate_residuals_and_model(params_struct, data_np, model_function, debug_iteration=iteration)
            current_cost = _calculate_cost(residuals_list, robust_cost)
            cost_at_iter_start = current_cost  # Store cost for current params

            if not np.isfinite(cost_at_iter_start):
                error_message = "Cost function resulted in non-finite value (NaN or Inf)."
                if log_level_threshold >= LOG_LEVELS['error']:
                    log_func(error_message, 'error')
                break

            # Update accepted cost if this is the first iteration or if the previous step was accepted
            if iteration == 1 or cost_at_iter_start < cost:
                cost = cost_at_iter_start
                final_residuals_list = residuals_list

            if log_level_threshold >= LOG_LEVELS['debug']:
                log_func(f"Iteration {iteration}, Current Cost: {cost:.6e}, Lambda: {lm_lambda:.3e}", 'debug')

            # Call progress callback
            try:
                progress_func({
                    'iteration': iteration, 'chiSquared': cost, 'lambda': lm_lambda,
                    'activeParameters': params_active.copy()
                })
            except Exception as prog_e:
                warnings.warn(f"Error in onProgress callback: {prog_e}", RuntimeWarning)

            # 2. Check convergence
            if iteration > 1 and np.isfinite(cost_prev):
                cost_change = abs(cost - cost_prev)
                if cost_change < error_tolerance * (1 + abs(cost_prev)):
                    if log_level_threshold >= LOG_LEVELS['info']:
                        log_func(f"Converged: Chi-squared change ({cost_change:.2e}) below tolerance.", 'info')
                    converged = True
                    break

            # 3. Calculate Jacobian and Gradient
            if num_active_params > 0:
                jacobian, residuals_flat = _calculate_jacobian(param_manager, params_active, data_np, model_function, epsilon, residuals_list)
                final_jacobian = jacobian

                gradient = jacobian.T @ residuals_flat
                max_grad_comp = np.max(np.abs(gradient)) if len(gradient) > 0 else 0.0
                if max_grad_comp < gradient_tolerance:
                    if log_level_threshold >= LOG_LEVELS['info']:
                        log_func(f"Converged: Maximum gradient component ({max_grad_comp:.2e}) below tolerance.", 'info')
                    converged = True
                    break
            else:
                if log_level_threshold >= LOG_LEVELS['info']:
                    log_func("No active parameters to optimize. 'Converged'.", 'info')
                converged = True
                break

            # 4. Calculate Hessian Approximation and Parameter Step
            hessian_approx = jacobian.T @ jacobian
            diag_elements = np.diag(hessian_approx).copy()
            diag_elements[diag_elements <= 1e-12] = 1e-12  # Floor for damping
            damping_matrix = lm_lambda * np.diag(diag_elements)
            augmented_hessian = hessian_approx + damping_matrix

            try:
                delta_p = np.linalg.solve(augmented_hessian, gradient)
            except np.linalg.LinAlgError:
                if log_level_threshold >= LOG_LEVELS['warn']:
                    log_func(f"Iteration {iteration}: Linear solve failed. Trying pseudoinverse.", 'warn')
                try:
                    delta_p = np.linalg.pinv(augmented_hessian) @ gradient
                except np.linalg.LinAlgError:
                    error_message = f"Iteration {iteration}: Failed to solve for parameter step even with pseudoinverse."
                    break

            # 5. Evaluate prospective parameters and cost
            params_active_new = params_active + delta_p
            params_struct_new = param_manager.reconstruct(params_active_new)
            params_struct_new = _apply_constraints(params_struct_new, constraints, constraint_func)
            temp_flat_all_new = np.concatenate([pg.ravel() for ds in params_struct_new for pg in ds])
            params_active_new = temp_flat_all_new[param_manager.get_active_indices()] if num_active_params > 0 else np.array([])

            residuals_new_list, _, _, _ = _calculate_residuals_and_model(params_struct_new, data_np, model_function, debug_iteration=iteration + 0.5)
            cost_new = _calculate_cost(residuals_new_list, robust_cost)

            # 6. Accept or reject step, update lambda
            if cost_new < cost and np.isfinite(cost_new):
                if log_level_threshold >= LOG_LEVELS['debug']:
                    log_func(f"Iteration {iteration}: Step accepted. Cost decreased from {cost:.6e} to {cost_new:.6e}", 'debug')
                cost_prev = cost
                params_active = params_active_new
                lm_lambda /= lambda_decrease
            else:
                if log_level_threshold >= LOG_LEVELS['debug']:
                    reason = "non-finite cost" if not np.isfinite(cost_new) else f"cost increased/stalled ({cost_new:.6e} >= {cost:.6e})"
                    log_func(f"Iteration {iteration}: Step rejected ({reason}). Increasing lambda.", 'debug')
                lm_lambda *= lambda_increase
                cost_prev = cost
                if lm_lambda > 1e10:
                    if log_level_threshold >= LOG_LEVELS['warn']:
                        log_func("Lambda reached maximum limit. Fit may be stalled.", 'warn')

        except Exception as e:
            error_message = f"Error during LM iteration {iteration}: {e}"
            import traceback
            if log_level_threshold >= LOG_LEVELS['error']:
                log_func(error_message, 'error')
                log_func(traceback.format_exc(), 'debug')
            break

    # --- End of Loop ---
    if error_message is None and iteration >= max_iterations and not converged:
        if log_level_threshold >= LOG_LEVELS['warn']:
            log_func(f"Fit did not converge within {max_iterations} iterations.", 'warn')

    # Return final state
    return params_active, cost, converged, iteration, final_jacobian, final_residuals_list, error_message


def lm_fit_global(
    data: Dict[str, List[Sequence[float]]],
    model_function: ModelFunctionsType,
    initial_parameters: ParametersType,
    options: Optional[OptionsType] = None
) -> ResultType:
    """
    Performs Levenberg-Marquardt global fitting on multiple datasets.

    Args:
        data: Dictionary containing 'x', 'y', 'ye'.
              'x': List of sequences (e.g., lists or np.arrays) of independent variable values.
              'y': List of sequences of dependent variable values.
              'ye': List of sequences of error/uncertainty values (std devs). Must not contain zeros.
        model_function: List of lists of model functions. model_function[dsIdx][paramGroupIdx]
                        is a callable `func(params_array, x_point_array)` returning `np.array([y_value])`.
        initial_parameters: Nested list of initial parameter guesses, matching model_function structure.
        options: Optional dictionary of fitting options (see DEFAULT_OPTIONS).

    Returns:
        Dictionary containing fitting results.
    """
    start_time = time.time()
    current_options = DEFAULT_OPTIONS.copy()
    if options:
        current_options.update(options)

    # --- Input Validation and Preparation ---
    result_on_error = {'error': None} # Placeholder for errors
    if not isinstance(data, dict) or not all(k in data for k in ['x', 'y', 'ye']):
        result_on_error['error'] = "Data must be a dictionary containing 'x', 'y', and 'ye' keys."
        return result_on_error
    if not (isinstance(data['x'], list) and isinstance(data['y'], list) and isinstance(data['ye'], list) and
            len(data['x']) == len(data['y']) == len(data['ye'])):
        result_on_error['error'] = "Data 'x', 'y', 'ye' must be lists of the same length (number of datasets)."
        return result_on_error
    if not (isinstance(model_function, list) and isinstance(initial_parameters, list) and
            len(data['x']) == len(model_function) == len(initial_parameters)):
         result_on_error['error'] = "Number of datasets in data, model_function, and initial_parameters must match."
         return result_on_error

    # Convert data to numpy arrays internally
    data_np: DataType = {'x': [], 'y': [], 'ye': []}
    try:
        for i in range(len(data['x'])):
             x_arr = np.asarray(data['x'][i], dtype=float)
             y_arr = np.asarray(data['y'][i], dtype=float)
             ye_arr = np.asarray(data['ye'][i], dtype=float)

             if not (x_arr.shape == y_arr.shape == ye_arr.shape):
                  result_on_error['error'] = f"Dataset {i}: x, y, and ye must have the same shape. Found {x_arr.shape}, {y_arr.shape}, {ye_arr.shape}."
                  return result_on_error
             if np.any(ye_arr <= 0):
                  result_on_error['error'] = f"Dataset {i}: 'ye' must contain only positive values."
                  return result_on_error
             if x_arr.size == 0: # Use .size for numpy arrays
                  result_on_error['error'] = f"Dataset {i}: Contains no data points."
                  return result_on_error
             data_np['x'].append(x_arr)
             data_np['y'].append(y_arr)
             data_np['ye'].append(ye_arr)

    except Exception as e:
         result_on_error['error'] = f"Failed to convert data to NumPy arrays: {e}"
         return result_on_error
    
    # ****** ENSURE THIS LINE IS PRESENT HERE ******
    total_points = sum(x.size for x in data_np['x']) # Calculate total points using .size for numpy arrays
    # ****** END OF REQUIRED LINE ******

    # Prepare options
    log_level = current_options['logLevel']
    log_level_threshold = LOG_LEVELS.get(log_level.lower(), LOG_LEVELS['info'])
    log_func = current_options['onLog']
    progress_func = current_options['onProgress']

    if not callable(log_func): log_func = _default_log_func
    if not callable(progress_func): progress_func = _default_progress_func
    current_options['onLog'] = log_func # Ensure actual callable is stored
    current_options['onProgress'] = progress_func


    if log_level_threshold >= LOG_LEVELS['info']: log_func(f"Starting global fit...", 'info')
    if log_level_threshold >= LOG_LEVELS['debug']: log_func(f"Options: { {k:v for k,v in current_options.items() if not callable(v)} }", 'debug') # Avoid printing functions


    # Parameter Manager Setup
    try:
        # Ensure initial_parameters structure is valid before passing
        if not all(isinstance(ds, list) for ds in initial_parameters):
            raise TypeError("initial_parameters must be a list of lists.")
        if not all(isinstance(pg, list) for ds in initial_parameters for pg in ds):
             raise TypeError("initial_parameters must be a list of lists of lists.")

        param_manager = ParameterManager(
            initial_parameters,
            current_options['fixMap'],
            current_options['linkMap']
        )
        params_active = param_manager.get_initial_active_params()
        num_active_params = param_manager.get_num_active()

        if log_level_threshold >= LOG_LEVELS['debug']:
             log_func(f"Total parameters: {param_manager.total_params_flat}", 'debug')
             log_func(f"Active parameters: {num_active_params}", 'debug')
             log_func(f"Active param labels: {param_manager.get_active_param_labels()}", 'debug')

    except Exception as e:
         err_msg = f"Error initializing ParameterManager: {e}"
         if log_level_threshold >= LOG_LEVELS['error']: log_func(err_msg, 'error')
         result_on_error['error'] = err_msg
         return result_on_error

    # Apply initial constraints
    try:
        # 1. Get initial full structure using initial active params
        params_struct_initial = param_manager.reconstruct(params_active)

        # 2. Apply constraints to the full structure
        params_struct_constrained = _apply_constraints(
            params_struct_initial,
            current_options['constraints'],
            current_options['constraintFunction']
        )

        # 3. Extract the *active* parameter values from the *constrained* structure
        #    Iterate through the parameter metadata. If a parameter is active,
        #    get its value from the constrained structure.
        constrained_active_params_list = []
        if param_manager.get_num_active() > 0:
            for meta in param_manager._param_meta:
                if meta['active_idx'] is not None: # This parameter is active
                    ds, pg, pi = meta['ds'], meta['pg'], meta['pi']
                    try:
                        # Access the constrained value using original coordinates
                        constrained_value = params_struct_constrained[ds][pg][pi]
                        constrained_active_params_list.append(constrained_value)
                    except IndexError:
                        raise RuntimeError(f"Internal Error: Structure mismatch accessing constrained param at [{ds}][{pg}][{pi}]")

            # Ensure the list length matches num_active_params
            if len(constrained_active_params_list) != param_manager.get_num_active():
                 raise RuntimeError(f"Internal Error: Mismatch extracting constrained active params. Expected {param_manager.get_num_active()}, got {len(constrained_active_params_list)}.")

            # Update params_active with the constrained values
            params_active = np.array(constrained_active_params_list, dtype=float)
        else:
             params_active = np.array([], dtype=float) # No active params

        if log_level_threshold >= LOG_LEVELS['debug']:
             log_func(f"Active params after initial constraints: {params_active}", 'debug')

    except Exception as e:
         err_msg = f"Error applying initial constraints: {e}"
         if log_level_threshold >= LOG_LEVELS['error']: log_func(err_msg, 'error')
         result_on_error['error'] = err_msg
         return result_on_error

    # Calculate total_points
    total_points = sum(x.size for x in data_np['x']) if data_np else 0
    num_active_params = param_manager.get_num_active() # Get K

    # --- Call the Core LM Loop ---
    try:
        params_active, cost, converged, iteration, final_jacobian, final_residuals_list_from_loop, error_message = \
            _run_lm_optimization_loop(
                param_manager,
                params_active, # Initial active params after constraints
                data_np,
                model_function,
                current_options, # Pass the full options dict
                log_func,
                progress_func,
                log_level_threshold
            )
        # Use the cost returned from the loop as the initial chi_sq value
        chi_sq = cost
        # Use residuals from loop if available and no error occurred

        # --- ADD DEBUG ---
        print(f"!!! lm_fit_global: Loop finished. Final active params: {params_active}")
        print(f"!!! lm_fit_global: Loop finished. Final cost: {chi_sq}")
        # --- END DEBUG ---

        final_residuals_list = final_residuals_list_from_loop if error_message is None else None

    except Exception as loop_e:
        # ... (handle loop errors) ...
        # Initialize variables needed later
        params_active = param_manager.get_initial_active_params()
        chi_sq = np.nan
        converged = False
        iteration = 0
        final_jacobian = None
        final_residuals_list = None
        error_message = f"Error during LM optimization loop: {loop_e}"

     # --- Post-Fitting Calculations ---
    final_params_struct = None # Initialize
    try:
        final_params_struct = param_manager.reconstruct(params_active)

        # --- ADD DEBUG ---
        print(f"!!! lm_fit_global: Reconstructed final params structure: {final_params_struct}")
        # --- END DEBUG ---

        # Recalculate final state ONLY if needed (e.g., loop error or residuals missing)
        # Otherwise, use values returned from the loop
        if error_message is None and final_residuals_list is None:
             if log_level_threshold >= LOG_LEVELS['debug']:
                  log_func("Recalculating final residuals/cost post-loop.", 'debug')
             final_residuals_list, final_model_list, final_component_models_list, _ = \
                 _calculate_residuals_and_model(final_params_struct, data_np, model_function, debug_iteration=-1)
             # --- FIX: Access robust_cost from current_options ---
             chi_sq = _calculate_cost(final_residuals_list, current_options['robustCostFunction'])
             # --- END FIX ---
        elif error_message is None:
             # If loop finished okay and returned residuals, recalculate cost just to be sure
             # (cost from loop is based on params *before* last update if rejected)
             # --- FIX: Access robust_cost from current_options ---
             chi_sq = _calculate_cost(final_residuals_list, current_options['robustCostFunction'])
             # --- END FIX ---
             # Also recalculate final models based on final accepted params
             _, final_model_list, final_component_models_list, _ = \
                 _calculate_residuals_and_model(final_params_struct, data_np, model_function, debug_iteration=-1)
    except Exception as post_e:
         error_message = error_message or f"Error during final state calculation: {post_e}"
         if log_level_threshold >= LOG_LEVELS['error']: log_func(error_message, 'error')
         chi_sq = np.nan
         final_residuals_list = None
         final_model_list = None
         final_component_models_list = None

    # Calculate final statistics and errors
    covariance_matrix = None
    parameter_errors = np.full(num_active_params, np.nan, dtype=float)
    scale_factor = np.nan
    dof = total_points - num_active_params if total_points > 0 else 0

    if error_message is None and num_active_params > 0 and final_jacobian is not None and dof > 0 and final_residuals_list is not None:
        covariance_matrix, parameter_errors, scale_factor = _calculate_covariance_and_errors(
            final_jacobian, np.concatenate(final_residuals_list), num_active_params, total_points,
            chi_sq, current_options['covarianceLambda'], current_options['robustCostFunction'],
            log_func, log_level_threshold
        )
    elif error_message is None and dof <= 0:
         if log_level_threshold >= LOG_LEVELS['warn']:
              log_func(f"Cannot calculate errors or statistics: Degrees of freedom ({dof}) <= 0.", 'warn')

    # Reconstruct full error structure
    final_param_errors_struct = _reconstruct_full_errors(param_manager, parameter_errors)


    # Calculate Goodness-of-fit statistics
    red_chi_sq = chi_sq / dof if dof > 0 and np.isfinite(chi_sq) else np.inf
    aic = np.nan
    aicc = np.nan
    bic = np.nan
    if np.isfinite(chi_sq) and total_points > 0:
        k = num_active_params
        n = total_points
        aic = chi_sq + 2 * k
        if n > k + 1:
             aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        else:
             aicc = np.inf # Correction term explodes
        bic = chi_sq + k * np.log(n)

    # Generate fitted model curves
    fitted_model_curves = None
    if current_options['calculateFittedModel'] and error_message is None and final_params_struct is not None:
        try:
            fitted_model_curves = _generate_fitted_curves(
                final_params_struct, data_np, model_function,
                current_options['calculateFittedModel'], current_options['model_x_range'],
                log_func, log_level_threshold
            )
        except Exception as e: warnings.warn(f"Failed to generate fitted curves: {e}", RuntimeWarning)

    # Generate component model curves
    fitted_component_curves = None
    if current_options['calculateComponentModels'] and error_message is None and final_params_struct is not None:
        try:
            # --- ADD DEBUG ---
            print(f"!!! lm_fit_global: Calling _generate_component_curves with final params...")
            # --- END DEBUG ---
            fitted_component_curves = _generate_component_curves(
                 final_params_struct, data_np, model_function,
                 current_options['calculateFittedModel'] or True, current_options['model_x_range'],
                 log_func, log_level_threshold
            )
        except Exception as e: warnings.warn(f"Failed to generate component curves: {e}", RuntimeWarning)

    # Calculate Confidence Intervals
    ci_lower = None
    ci_upper = None
    if current_options['confidenceInterval'] is not None and error_message is None:
         conf_level = current_options['confidenceInterval']
         if 0 < conf_level < 1:
              try:
                   if final_residuals_list is None: # Check if residuals are available
                        warnings.warn("Final residuals not available, cannot calculate CIs.", RuntimeWarning)
                   else:
                       ci_lower, ci_upper = _calculate_confidence_intervals(
                           param_manager, params_active, final_jacobian, covariance_matrix,
                           data_np, model_function, conf_level, total_points, num_active_params,
                           current_options['calculateFittedModel'] or True,
                           log_func, log_level_threshold,
                           current_options['bootstrapFallback'], current_options['numBootstrapSamples'],
                           final_residuals_list, current_options['model_x_range'],
                           current_options # Pass full options dict
                       )
              except Exception as e:
                   warnings.warn(f"Failed to calculate confidence intervals: {e}", RuntimeWarning)
                   log_func(traceback.format_exc(), 'debug')
         else:
              warnings.warn(f"Invalid confidenceInterval option ({conf_level}).", RuntimeWarning)


    # Convert final nested numpy arrays back to lists for output
    final_params_struct_list = [[list(pg) for pg in ds] for ds in final_params_struct] if final_params_struct is not None else initial_parameters

    end_time = time.time()
    if log_level_threshold >= LOG_LEVELS['info']: log_func(f"Fit finished in {end_time - start_time:.3f} seconds.", 'info')

    # Construct result dictionary (keep curves/CIs/residuals as numpy)
    result: ResultType = {
        'p_active': list(params_active),
        'p_reconstructed': final_params_struct_list,
        'finalParamErrors': final_param_errors_struct,
        'chiSquared': float(chi_sq) if np.isfinite(chi_sq) else None,
        'covarianceMatrix': covariance_matrix.tolist() if covariance_matrix is not None else None,
        'parameterErrors': [float(e) if np.isfinite(e) else None for e in parameter_errors],
        'iterations': iteration,
        'converged': converged,
        'activeParamLabels': param_manager.get_active_param_labels(),
        'error': error_message,
        'totalPoints': total_points,
        'degreesOfFreedom': dof if dof >= 0 else 0,
        'reducedChiSquared': float(scale_factor) if np.isfinite(scale_factor) else None, # Use scale_factor as rcs
        'aic': float(aic) if np.isfinite(aic) else None,
        'aicc': float(aicc) if np.isfinite(aicc) else None,
        'bic': float(bic) if np.isfinite(bic) else None,
        'residualsPerSeries': final_residuals_list, # List[np.ndarray]
        'fittedModelCurves': fitted_model_curves, # List[Dict[str, np.ndarray]]
        'ci_lower': ci_lower, # List[Dict[str, np.ndarray]] or None
        'ci_upper': ci_upper, # List[Dict[str, np.ndarray]] or None
        'fittedModelComponentCurves': fitted_component_curves # List[List[Dict[str, np.ndarray]]] or None
    }
    return result


def lm_fit(
    data: Dict[str, Sequence[float]], # Single dataset {x:[], y:[], ye:[]}
    model_function: Union[ModelFunctionType, List[ModelFunctionType]],
    initial_parameters: Union[Sequence[float], List[Sequence[float]]],
    options: Optional[OptionsType] = None
) -> ResultType:
    """
    Convenience wrapper for fitting a single dataset using lm_fit_global.

    Args:
        data: Dictionary with 'x', 'y', 'ye' for a single dataset.
        model_function: A single model function or a list of functions (for composite model).
        initial_parameters: A single list of parameters or a list of lists (matching model_function).
        options: Fitting options, similar to lm_fit_global, but maps/constraints
                 should be in the single-dataset format (e.g., fixMap=[[False, True]]).

    Returns:
        Dictionary containing fitting results.
    """
    result_on_error = {'error': None}
    # --- Wrap inputs into the multi-dataset structure ---
    if not isinstance(data, dict) or not all(k in data for k in ['x', 'y', 'ye']):
        result_on_error['error'] = "Data dictionary must contain 'x', 'y', and 'ye' keys."
        return result_on_error

    data_global = {
        'x': [data['x']],
        'y': [data['y']],
        'ye': [data['ye']]
    }

    # Handle model_function structure
    model_function_list = []
    if isinstance(model_function, list):
        if not all(callable(f) for f in model_function):
             result_on_error['error'] = "If model_function is a list, all elements must be callable functions."
             return result_on_error
        model_function_list = model_function # Already a list of functions for one dataset
    elif callable(model_function):
        model_function_list = [model_function] # Wrap single function
    else:
        result_on_error['error'] = "model_function must be a callable function or a list of callable functions."
        return result_on_error
    model_function_global = [model_function_list]


    # Handle initial_parameters structure
    initial_parameters_list = []
    if not initial_parameters: # Check if empty list/sequence
         result_on_error['error'] = "initial_parameters cannot be empty."
         return result_on_error

    # Check if it looks like a list of lists already
    try:
        is_nested_list = isinstance(initial_parameters[0], (list, tuple, np.ndarray))
    except IndexError:
         result_on_error['error'] = "initial_parameters cannot be an empty sequence."
         return result_on_error
    except TypeError:
         result_on_error['error'] = "initial_parameters must be a sequence (list/tuple) or list of sequences."
         return result_on_error


    if is_nested_list:
        # Assume it's already in the format [[p1, p2], [p3]] for composite model
         if len(initial_parameters) != len(model_function_list):
              result_on_error['error'] = f"Structure mismatch: {len(initial_parameters)} parameter groups provided, but {len(model_function_list)} model functions found."
              return result_on_error
         initial_parameters_list = [list(p) for p in initial_parameters] # Ensure inner are lists
    else:
         # Assume it's a flat list [p1, p2, p3] for a single model function
         if len(model_function_list) != 1:
             result_on_error['error'] = f"Flat initial_parameters list provided, but {len(model_function_list)} model functions found (expected 1)."
             return result_on_error
         initial_parameters_list = [list(initial_parameters)] # Wrap into list of lists [[p1, p2]]

    initial_parameters_global = [initial_parameters_list]


    # Handle options like fixMap, linkMap, constraints - wrap them
    options_global = options.copy() if options else {}
    for map_key in ['fixMap', 'linkMap', 'constraints']:
        if map_key in options_global and options_global[map_key] is not None:
            # Expects single dataset format, e.g., [[False, True], [True]]
            # Wrap it in another list: [[[False, True], [True]]]
            options_global[map_key] = [options_global[map_key]]

    return lm_fit_global(data_global, model_function_global, initial_parameters_global, options_global)


# --- Independent Fitting (Parallel) ---

# Need a top-level helper function for multiprocessing.Pool.map
# It needs access to shared data structures, models, and options.
# Using functools.partial is one way to pass these without relying on globals.

def _fit_single_dataset_worker(
    dataset_index: int,
    # Arguments passed via functools.partial:
    data_all: DataType,
    model_function_all: ModelFunctionsType,
    initial_parameters_all: ParametersType,
    options_all: OptionsType, # This contains the full model_x_range list
    log_queue: Optional[Manager] = None
) -> Tuple[int, ResultType]:
    """
    Worker function to fit a single dataset. Designed for use with multiprocessing.Pool.
    """
    # Deep copy options to avoid side effects between workers if options contain mutable objects
    import copy
    dataset_options = copy.deepcopy(options_all)

    try:
        dataset_data = {
            'x': [data_all['x'][dataset_index]],
            'y': [data_all['y'][dataset_index]],
            'ye': [data_all['ye'][dataset_index]]
        }
        dataset_model = [model_function_all[dataset_index]]
        dataset_params = [initial_parameters_all[dataset_index]]
    except IndexError:
         return dataset_index, {'error': f"Worker {dataset_index}: Data/Model/Params index out of bounds."}


    # Remove linkMap as it's irrelevant for independent fits
    dataset_options.pop('linkMap', None)

    # Extract single-dataset fixMap and constraints if provided in full structure
    for map_key in ['fixMap', 'constraints']:
        original_map = dataset_options.get(map_key, None)
        if isinstance(original_map, list) and len(original_map) > dataset_index:
             # Assume it's the multi-dataset structure, extract the relevant part
             dataset_options[map_key] = [original_map[dataset_index]]
        elif original_map is not None:
             # If it's not a list or index is wrong, it's likely malformed for this context
             # We could warn or ignore. Ignoring seems safer if structure is ambiguous.
             # print(f"[Worker {dataset_index}] Warning: Structure of '{map_key}' not suitable for independent fit extraction. Ignoring.", file=sys.stderr)
             dataset_options[map_key] = None

    # --- Extract model_x_range for this specific dataset ---
    model_x_range_full = dataset_options.get('model_x_range', None)
    current_series_range = None
    if model_x_range_full and isinstance(model_x_range_full, list) and len(model_x_range_full) > dataset_index:
        current_series_range = model_x_range_full[dataset_index]
    # Wrap it back into a list containing only this series' range for lm_fit_global
    dataset_options['model_x_range'] = [current_series_range]
    # --- End extraction ---

    # Custom logging for this worker process
    original_log_func = dataset_options.get('onLog', _default_log_func) # Get potentially updated default
    log_level = dataset_options.get('logLevel', 'info')
    log_level_threshold = LOG_LEVELS.get(log_level.lower(), LOG_LEVELS['info'])

    def worker_log(message, level):
         # Only queue if message level is relevant
         if LOG_LEVELS.get(level.lower(), LOG_LEVELS['none']) <= log_level_threshold:
             log_entry = (dataset_index, message, level)
             if log_queue:
                  try:
                      log_queue.put(log_entry)
                  except Exception as e:
                      # Cannot easily log this error without potential recursion/deadlock
                      print(f"[Worker {dataset_index}] CRITICAL: Failed to put log in queue: {e}", file=sys.stderr)
             else: # Fallback if queue not provided (e.g., sequential run)
                  # Check again if original is callable, as deepcopy might affect it? Unlikely for top-level funcs.
                   if callable(original_log_func):
                       original_log_func(f"[Dataset {dataset_index}] {message}", level)

    dataset_options['onLog'] = worker_log

    # Custom progress (less critical to pipe back in real-time, could aggregate later)
    original_progress_func = dataset_options.get('onProgress', _default_progress_func)
    def worker_progress(progress_data):
         # Add dataset index to progress data if needed by main process
         progress_data['datasetIndex'] = dataset_index
         if callable(original_progress_func):
             try:
                 original_progress_func(progress_data) # Call original
             except Exception as prog_e:
                 # Cannot easily log this. Print to stderr?
                 print(f"[Worker {dataset_index}] Error in onProgress callback: {prog_e}", file=sys.stderr)

    dataset_options['onProgress'] = worker_progress


    # Perform the fit using lm_fit_global on the single dataset
    # lm_fit_global expects list-of-lists for models/params, which we have constructed.
    # It also expects dict[str, list[array]] for data, which we have.
    result = lm_fit_global(dataset_data, dataset_model, dataset_params, dataset_options)

    return dataset_index, result


def lm_fit_independent(
    data: Dict[str, List[Sequence[float]]],
    model_function: ModelFunctionsType,
    initial_parameters: ParametersType,
    options: Optional[OptionsType] = None,
    num_workers: Optional[int] = None
) -> List[ResultType]:
    """
    Fits multiple datasets independently, potentially in parallel.

    Args:
        data: Dictionary containing 'x', 'y', 'ye' lists for multiple datasets.
        model_function: List of lists of model functions for each dataset.
        initial_parameters: Nested list of initial parameters for each dataset.
        options: Fitting options, applied to each fit (linkMap is ignored).
        num_workers: Number of worker processes to use. Defaults to cpu_count().
                     Set to 1 to disable parallelism.

    Returns:
        A list of result dictionaries, one for each dataset fit.
    """
    start_time = time.time()
    current_options = DEFAULT_OPTIONS.copy()
    if options:
        current_options.update(options)

    # Basic input validation
    if not isinstance(data, dict) or not all(k in data for k in ['x','y','ye']):
         raise ValueError("Data must be a dict with 'x', 'y', 'ye' keys.")
    num_datasets = len(data.get('x', []))
    if num_datasets == 0:
        return []

    if not (len(data.get('y',[])) == num_datasets and len(data.get('ye',[])) == num_datasets and \
            isinstance(model_function, list) and len(model_function) == num_datasets and \
            isinstance(initial_parameters, list) and len(initial_parameters) == num_datasets):
         raise ValueError("Mismatch in the number or type of datasets between data, model_function, and initial_parameters.")

    # Prepare shared arguments (convert data to NumPy first)
    try:
        data_np: DataType = {
            'x': [np.asarray(x, dtype=float) for x in data['x']],
            'y': [np.asarray(y, dtype=float) for y in data['y']],
            'ye': [np.asarray(ye, dtype=float) for ye in data['ye']]
        }
        # Validate numpy arrays further if needed (shapes, ye>0 etc.) - lm_fit_global does this too
    except Exception as e:
         raise ValueError(f"Failed to convert input data to NumPy arrays: {e}") from e

    # Ensure model functions are callable
    if not all(isinstance(mfl, list) and all(callable(mf) for mf in mfl) for mfl in model_function):
         raise TypeError("model_function must be a list of lists of callable functions.")


    # Determine number of workers
    max_workers = cpu_count()
    if num_workers is None:
        workers = max_workers
    elif num_workers <= 0:
         workers = 1 # Sensible default for invalid input
    else:
        workers = min(num_workers, max_workers)

    # Disable parallelism if workers = 1 or only one dataset
    use_parallel = workers > 1 and num_datasets > 1

    results_list = [cast(ResultType, None)] * num_datasets # Initialize list to store results in order

    # Get logging function safely
    log_func = current_options.get('onLog', _default_log_func)
    if not callable(log_func): log_func = _default_log_func
    log_level = current_options.get('logLevel', 'info')
    log_level_threshold = LOG_LEVELS.get(log_level.lower(), LOG_LEVELS['info'])

    if log_level_threshold >= LOG_LEVELS['info']:
         parallel_msg = f"in parallel using {workers} workers" if use_parallel else "sequentially"
         log_func(f"Starting independent fits for {num_datasets} datasets {parallel_msg}...", 'info')


    if use_parallel:
         # Use Manager Queue for collecting logs from workers reliably
        with Manager() as manager:
            log_queue: queue.Queue = manager.Queue() # Type hint if needed
            pool = Pool(processes=workers) # Create pool inside manager context

            # Prepare partial function with fixed arguments for the worker
            # Pass the numpy data, original models/params lists, and options dict
            worker_partial = partial(
                _fit_single_dataset_worker,
                data_all=data_np,
                model_function_all=model_function,
                initial_parameters_all=initial_parameters,
                options_all=current_options, # Worker will deepcopy this
                log_queue=log_queue
            )

            # Run tasks
            async_result = None
            pool_results = None
            try:
                 # Use map_async to potentially retrieve logs while running
                 async_result = pool.map_async(worker_partial, range(num_datasets))

                 # Process logs from the queue while waiting for results
                 processed_logs = 0
                 finished_count = 0
                 while not async_result.ready():
                     try:
                         log_entry = log_queue.get(timeout=0.1) # Non-blocking with timeout
                         ds_idx, msg, level = log_entry
                         log_func(f"[Dataset {ds_idx}] {msg}", level) # Call user's log function
                         processed_logs += 1
                     except queue.Empty:
                         pass # Continue waiting/polling
                     except Exception as q_err:
                          print(f"Error processing log queue: {q_err}", file=sys.stderr)
                          # Decide whether to break or continue trying
                          time.sleep(0.1) # Avoid busy-waiting on error

                 # Get results once ready
                 pool_results = async_result.get()
                 finished_count = len(pool_results)

                 # Drain any remaining logs
                 while not log_queue.empty():
                      try:
                          log_entry = log_queue.get_nowait()
                          ds_idx, msg, level = log_entry
                          log_func(f"[Dataset {ds_idx}] {msg}", level)
                          processed_logs += 1
                      except queue.Empty:
                          break
                      except Exception as q_err:
                          print(f"Error draining log queue: {q_err}", file=sys.stderr)


                 # Store results in correct order
                 for idx, result_dict in pool_results:
                      if 0 <= idx < num_datasets:
                           results_list[idx] = result_dict
                      else:
                           print(f"Warning: Received result for unexpected dataset index {idx}", file=sys.stderr)

                 # Check if any results are missing (e.g., worker crash)
                 if finished_count < num_datasets:
                      err_msg = f"Parallel execution finished, but only received {finished_count}/{num_datasets} results. Some fits may have failed."
                      if log_level_threshold >= LOG_LEVELS['error']: log_func(err_msg, 'error')
                      # Fill missing results with error state
                      for i in range(num_datasets):
                           if results_list[i] is None:
                                results_list[i] = {'error': 'Parallel worker failed to return result.'}


            except Exception as e:
                 err_msg = f"Error during parallel execution setup or result retrieval: {e}"
                 import traceback
                 print(traceback.format_exc(), file=sys.stderr) # Print full traceback
                 if log_level_threshold >= LOG_LEVELS['error']: log_func(err_msg, 'error')
                 # Return list of errors
                 results_list = [{'error': err_msg} for _ in range(num_datasets)]
            finally:
                 pool.close() # Ensure pool resources are released
                 pool.join()


    else: # Sequential execution
        # No need for queue, just call worker directly
        worker_partial = partial(
            _fit_single_dataset_worker,
            data_all=data_np,
            model_function_all=model_function,
            initial_parameters_all=initial_parameters,
            options_all=current_options, # Worker will deepcopy this
            log_queue=None # No queue needed
        )
        for i in range(num_datasets):
             if log_level_threshold >= LOG_LEVELS['debug']: log_func(f"Starting fit for dataset {i}...", 'debug')
             try:
                  idx, result_dict = worker_partial(i)
                  results_list[idx] = result_dict
             except Exception as e:
                  err_msg = f"Error fitting dataset {i}: {e}"
                  import traceback
                  print(traceback.format_exc(), file=sys.stderr) # Print full traceback
                  if log_level_threshold >= LOG_LEVELS['error']: log_func(err_msg, 'error')
                  results_list[i] = {'error': err_msg}


    end_time = time.time()
    if log_level_threshold >= LOG_LEVELS['info']:
        log_func(f"Independent fits finished in {end_time - start_time:.3f} seconds.", 'info')

    # --- Collect and Format Results ---
    final_results: List[ResultType] = []
    for s_idx, res_or_exc in enumerate(results_list):
        if isinstance(res_or_exc, dict) and 'error' in res_or_exc:
            # Append error result directly
            final_results.append(res_or_exc)
        elif isinstance(res_or_exc, ResultType):
            # Append the result directly without unwrapping, as it already contains NumPy arrays
            final_results.append(res_or_exc)
        else:
            # Handle unexpected types
            final_results.append({'error': f"Unexpected result type for dataset {s_idx}: {type(res_or_exc)}"})

    return final_results


# --- Simulation Function ---

def simulate_from_params(
    data_x: List[Sequence[float]],
    model_functions: ModelFunctionsType,
    parameters: ParametersType,
    options: Optional[OptionsType] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Simulates dependent variable (y) data based on x values, models, and parameters.

    Args:
        data_x: List of sequences of independent variable values for each dataset.
        model_functions: List of lists of model functions.
        parameters: Nested list of parameter values for each model function.
        options: Optional dictionary:
                 - 'noiseStdDev' (float | List[float] | None): Std dev for Gaussian noise.
                 - 'noiseType' (str | List[str]): 'gaussian', 'poisson', or 'none'. Default 'gaussian'.
                 - 'logFn' (Callable): Function for logging. Default print.

    Returns:
        Dictionary containing {'x': [...], 'y': [...]} where 'x' and 'y' are lists of NumPy arrays.
    """
    opts = {
        'noiseStdDev': None,
        'noiseType': 'gaussian',
        'logFn': lambda msg, level: print(f"[{level.upper()}] {msg}") # Default print logger for sim
    }
    if options:
        opts.update(options)

    log_func = opts['logFn']
    if not callable(log_func): log_func = lambda msg, level: print(f"[{level.upper()}] {msg}")

    num_datasets = len(data_x)

    if not (len(model_functions) == num_datasets and len(parameters) == num_datasets):
         raise ValueError("Mismatch in number of datasets between data_x, model_functions, and parameters.")

    # Ensure parameters are NumPy arrays for model functions
    params_np: ParametersNpType = [
        [np.asarray(pg, dtype=float) for pg in ds]
        for ds in parameters
    ]
    data_x_np = [np.asarray(x, dtype=float) for x in data_x]

    simulated_y_list = []

    # Normalize noise options (make them lists if single values provided)
    noise_type = opts['noiseType']
    if isinstance(noise_type, str):
        noise_type_list = [noise_type] * num_datasets
    elif isinstance(noise_type, list) and len(noise_type) == num_datasets:
         noise_type_list = noise_type
    else:
         raise ValueError("noiseType must be a string or a list matching the number of datasets.")

    noise_std_dev = opts['noiseStdDev']
    if isinstance(noise_std_dev, (int, float)) or noise_std_dev is None:
        noise_std_dev_list = [noise_std_dev] * num_datasets
    elif isinstance(noise_std_dev, list) and len(noise_std_dev) == num_datasets:
        noise_std_dev_list = noise_std_dev
    else:
        raise ValueError("noiseStdDev must be a number, None, or a list matching the number of datasets.")


    for ds_idx in range(num_datasets):
        x_ds = data_x_np[ds_idx]
        params_ds = params_np[ds_idx]
        models_ds = model_functions[ds_idx]
        y_model_ds = np.zeros_like(x_ds, dtype=float)

        if len(params_ds) != len(models_ds):
             raise ValueError(f"Dataset {ds_idx}: Mismatch between parameter groups and model functions.")

        # Calculate noiseless model
        for pg_idx, model_func in enumerate(models_ds):
            pg_params = params_ds[pg_idx]
            try:
                 # Vectorized calculation (or loop if necessary - depends on model_func)
                 for i, x_point in enumerate(x_ds):
                     y_model_ds[i] += model_func(pg_params, np.array([x_point]))[0]
            except Exception as e:
                 raise RuntimeError(f"Error evaluating model function {model_func.__name__} during simulation for dataset {ds_idx}: {e}") from e

        # Add noise
        y_noisy_ds = y_model_ds.copy()
        noise_t = noise_type_list[ds_idx].lower()
        noise_sd = noise_std_dev_list[ds_idx]

        if noise_t == 'gaussian':
            if noise_sd is None or not isinstance(noise_sd, (int, float)) or noise_sd < 0:
                 log_func(f"Dataset {ds_idx}: noiseStdDev must be non-negative number for Gaussian noise. Skipping noise.", 'warn')
            elif noise_sd > 0:
                 y_noisy_ds += np.random.normal(0, noise_sd, size=y_model_ds.shape)
        elif noise_t == 'poisson':
             # Poisson noise variance = mean. Noise std dev = sqrt(mean).
             # Generate Poisson samples based on the *mean* (the model value).
             # Ensure model values are non-negative.
             clamped_y_model = np.maximum(y_model_ds, 0)
             if np.any(y_model_ds < 0):
                 log_func(f"Dataset {ds_idx}: Model values must be non-negative for Poisson noise. Clamping negative values to 0.", 'warn')
             y_noisy_ds = np.random.poisson(clamped_y_model)
        elif noise_t != 'none':
            log_func(f"Dataset {ds_idx}: Unknown noiseType '{noise_type_list[ds_idx]}'. Skipping noise.", 'warn')

        simulated_y_list.append(y_noisy_ds)


    return {
        'x': data_x_np, # Return list of numpy arrays
        'y': simulated_y_list # List of numpy arrays
    }


# --- 1. Define Model Functions ---
# Must accept params=np.array([...]) and x=np.array([xValue]), return np.array([yValue])
def gaussian_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp, center, stddev = params
    val = x[0]
    if stddev == 0: return np.array([np.nan])
    exponent = -0.5 * ((val - center) / stddev)**2
    # Prevent overflow/underflow in exp
    if abs(exponent) > 700: return np.array([0.0])
    return np.array([amp * np.exp(exponent)])

def linear_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    slope, intercept = params
    return np.array([slope * x[0] + intercept])

def my_logger(message, level):
    """Example custom logging function."""
    print(f"[{level.upper()}] {message}")


# --- Example Usage (mimicking JS README) ---
if __name__ == "__main__":
    # Required for multiprocessing pool finalization on some platforms if run directly
    import math
    import matplotlib.pyplot as plt  # Import for plotting at the end

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
        # --- INCREASE NOISE SIGNIFICANTLY ---
        'ye': [
            [2.5, 2.5, 2.5, 2.5, 2.5, 2.5],  # Increased error for DS0
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Increased error for DS1
        ]
        # ------------------------------------
    }

    # --- 3. Define Model Structure ---
    model_function_in = [
        [gaussian_model],  # Dataset 0: Gaussian only
        [linear_model]     # Dataset 1: Linear only
    ]

    # --- 4. Initial Parameter Guesses ---
    initial_parameters_in = [
        [[9.0, 3.5, 1.0]],  # Dataset 0: Gaussian params [amp, center, stddev]
        [[1.8, 2.5]]        # Dataset 1: Linear params [slope, intercept]
    ]

    # --- 5. Define Options ---
    fix_map_in = [
        [[False, True, False]],  # Fix center for Gaussian in DS 0
        [[False, False]]         # Both linear params free in DS 1
    ]
    link_map_in = [
        [[None, None, "shared_param"]],
        [[None, "shared_param"]]
    ]
    extrap_range_ds0 = (0.0, 7.0)
    model_x_range_opt = [extrap_range_ds0, None]

    fit_options = {
        'maxIterations': 200,
        'logLevel': 'info',  # Use 'info' - bootstrap can be verbose on 'debug'
        'onLog': my_logger,
        'fixMap': fix_map_in,
        'linkMap': link_map_in,
        'covarianceLambda': 1e-9,
        'confidenceInterval': 0.95,
        'calculateFittedModel': {'numPoints': 50},  # Fewer points for faster demo
        'bootstrapFallback': True,  # Explicitly enable fallback
        'numBootstrapSamples': 50,  # Fewer samples for faster demo
        'calculateComponentModels': True,
        'model_x_range': model_x_range_opt,
        'num_workers': None  # Use default workers for bootstrap/independent
    }

    # --- 6. Run the Global Fit ---
    print("\n--- Running Global Fit (with increased noise to potentially trigger bootstrap) ---")
    result = {}
    try:
        result = lm_fit_global(data_in, model_function_in, initial_parameters_in, fit_options)

        # --- 7. Process Results ---
        if result.get('error'):
            print(f"\nFit failed: {result['error']}")
        else:
            print("\n--- Fit Results ---")
            print(f"Converged: {result['converged']} in {result['iterations']} iterations.")
            print(f"Final Chi^2: {result['chiSquared']:.5e}")
            print(f"Reduced Chi^2: {result['reducedChiSquared']:.5f}")
            print(f"Active Parameter Labels: {result['activeParamLabels']}")
            print(f"Final Active Parameters: {[f'{p:.4f}' for p in result['p_active']]}")
            print(f"Active Parameter Errors: {[f'{e:.4f}' if e is not None else 'N/A' for e in result['parameterErrors']]}")

            if result.get('ci_lower') and result.get('ci_upper'):
                print(f"\nConfidence Intervals calculated.")
                if result['ci_lower'][0] and np.any(np.isfinite(result['ci_lower'][0]['y'])):
                    print("   (CI data appears valid for DS0)")
                else:
                    print("   (CI data might be NaN for DS0)")
            else:
                print("\nConfidence Intervals not calculated or failed.")

    except Exception as e:
        print(f"\nError during fitting process: {e}")
        import traceback
        traceback.print_exc()

    # --- Plotting ---
    try:
        fig, axs = plt.subplots(len(data_in['x']), 2, figsize=(12, 5 * len(data_in['x'])), squeeze=False)
        fig.suptitle("LMFitGlobal Results (Increased Noise / Bootstrap Test)")

        for i in range(len(data_in['x'])):
            ax_fit = axs[i, 0]
            ax_res = axs[i, 1]
            ax_fit.set_title(f"Global Fit - Dataset {i}")
            ax_res.set_title(f"Global Fit Residuals - Dataset {i}")

            x_plot = np.asarray(data_in['x'][i])
            y_plot = np.asarray(data_in['y'][i])
            ye_plot = np.asarray(data_in['ye'][i])
            ax_fit.errorbar(x_plot, y_plot, yerr=ye_plot, fmt='o', label='Data', markersize=5, capsize=3)

            if result.get('fittedModelCurves') and len(result['fittedModelCurves']) > i:
                curve = result['fittedModelCurves'][i]
                if curve and curve['x'].size > 0:
                    ax_fit.plot(curve['x'], curve['y'], 'r-', label='Total Fit')

                    if result.get('ci_lower') and result.get('ci_upper') and \
                       len(result['ci_lower']) > i and len(result['ci_upper']) > i:
                        ci_l = result['ci_lower'][i]
                        ci_u = result['ci_upper'][i]
                        if ci_l and ci_u and ci_l['x'].size > 0:
                            ax_fit.fill_between(ci_l['x'], ci_l['y'], ci_u['y'], color='red', alpha=0.2, label='95% CI')

            if result.get('residualsPerSeries') and len(result['residualsPerSeries']) > i:
                residuals = result['residualsPerSeries'][i]
                if residuals is not None and residuals.size > 0:
                    ax_res.plot(x_plot, residuals, '.-')
                    ax_res.axhline(0, color='grey', linestyle='--', linewidth=1)

            ax_fit.legend()
            ax_fit.grid(True, alpha=0.5)
            ax_res.grid(True, alpha=0.5)
            ax_res.set_xlabel("X")
            ax_fit.set_ylabel("Y")
            ax_res.set_ylabel("Residual")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
    except Exception as plot_err:
        print(f"\nError during plotting: {plot_err}")
        import traceback
        traceback.print_exc()