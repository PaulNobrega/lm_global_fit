import sys
import os
import asyncio
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, Response

# --- Add parent directory to path if needed ---
# This allows importing globalfit_py_v2 if app.py is run directly
# Adjust if your structure is different
script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, script_dir) # Add current dir to path for the import below

# --- Import your fitting library ---
# Assuming the file is named globalfit_py_v2.py in the same directory
try:
    from lm_global_fit.global_fit import (
        simulate_from_params,
        lm_fit_global,
        ModelFunctionsType, # Import type for mapping
    )
        # Import model functions defined in the script for mapping
except ImportError as e:
    print(f"Error importing fitting library: {e}")
    print("Make sure 'global_fit.py' is in the same directory as 'app.py'.")
    sys.exit(1)
try:
    from examples.flask.fun import (
        gaussian_model,
        linear_model,
        constant_model,
        exponential_model
    )
except ImportError as e:
    print(f"Error importing function library: {e}")
    print("Make sure 'fun.py' is in the same directory as 'app.py'.")
    sys.exit(1)

# --- Flask App Setup ---
app = Flask(__name__)

# --- Model Mapping (Map keys from JS to Python functions) ---
# Ensure the keys match the 'value' in the HTML select options
MODEL_MAP = {
    "gaussian": gaussian_model,
    "linear": linear_model,
    "constant": constant_model,
    "exponential": exponential_model
    # Add other models here if defined in globalfit_py_v2.py
}

# --- JSON Serialization Helper ---
def safe_jsonify(data):
    """Converts numpy arrays and handles NaN/Inf for JSON."""
    def convert(o):
        if isinstance(o, np.ndarray):
            # Convert NaN/Inf within arrays before converting to list
            clean_array = np.where(np.isfinite(o), o, None) # Replace NaN/Inf with None
            return clean_array.tolist()
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32,
                            np.float64)):
            # Convert NaN/Inf float values
            return float(o) if np.isfinite(o) else None
        elif isinstance(o, (np.complex_, np.complex64, np.complex128)):
            return {'real': o.real, 'imag': o.imag}
        elif isinstance(o, (np.bool_)):
            return bool(o)
        elif isinstance(o, (np.void)):
            return None
        # Handle potential NaN/Inf at the top level if not caught by array conversion
        if isinstance(o, float) and not np.isfinite(o):
            return None
        # Let default error handling catch other types
        # raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        return o # Return as is if not a special numpy type or NaN/Inf float

    # Use json.dumps with the custom converter, then load back for jsonify
    # This ensures consistent handling via json module's checks
    try:
        json_string = json.dumps(data, default=convert, allow_nan=False) # Ensure allow_nan=False
        return Response(json_string, mimetype='application/json')
    except Exception as e:
        print(f"Error during JSON serialization: {e}")
        # Return a simple error JSON if serialization fails
        error_payload = json.dumps({"error": f"JSON Serialization Error: {e}"})
        return Response(error_payload, status=500, mimetype='application/json')


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    """Endpoint to run the simulation."""
    print("Received /simulate request") # Server log
    try:
        payload = request.get_json()
        if not payload:
            return safe_jsonify({"error": "Invalid JSON payload received."}), 400

        data_x = payload.get('data_x')
        model_keys_struct = payload.get('model_keys')
        parameters = payload.get('parameters')
        sim_options = payload.get('options', {})

        if not all([data_x, model_keys_struct, parameters]):
            return safe_jsonify({"error": "Missing required fields: data_x, model_keys, parameters."}), 400

        # Map model keys to actual Python functions
        model_functions: ModelFunctionsType = []
        for ds_keys in model_keys_struct:
            ds_funcs = []
            for key in ds_keys:
                func = MODEL_MAP.get(key)
                if not func:
                    return safe_jsonify({"error": f"Unknown model key '{key}' received."}), 400
                ds_funcs.append(func)
            model_functions.append(ds_funcs)

        # Call the Python simulation function
        sim_result_dict = simulate_from_params(
            data_x,
            model_functions,
            parameters,
            sim_options
        )

        if sim_result_dict is None:
             return safe_jsonify({"error": "Simulation failed on backend (returned None)."}), 500

        # Return the result (safe_jsonify handles numpy arrays)
        return safe_jsonify(sim_result_dict)

    except Exception as e:
        print(f"Error in /simulate: {e}")
        traceback.print_exc()
        return safe_jsonify({"error": f"Simulation failed on backend: {e}"}), 500


@app.route('/fit', methods=['POST'])
def fit():
    """Endpoint to run the fitting."""
    print("Received /fit request") # Server log
    try:
        payload = request.get_json()
        if not payload:
            return safe_jsonify({"error": "Invalid JSON payload received."}), 400

        sim_data = payload.get('data') # Expects {x:[], y:[], ye:[]}
        model_keys_struct = payload.get('model_keys')
        initial_parameters = payload.get('initial_parameters')
        fit_options = payload.get('options', {})
        fit_options['verbose'] = True # Enable verbose output for debugging
        fit_options['debug'] = True # Enable debug mode for the fitting process
        fit_options['log_level'] = 4

        if not all([sim_data, model_keys_struct, initial_parameters]):
            return safe_jsonify({"error": "Missing required fields: data, model_keys, initial_parameters."}), 400
        if not all(k in sim_data for k in ['x', 'y', 'ye']):
             return safe_jsonify({"error": "Fit data must include 'x', 'y', and 'ye'."}), 400

        # Map model keys to actual Python functions
        model_functions: ModelFunctionsType = []
        for ds_keys in model_keys_struct:
            ds_funcs = []
            for key in ds_keys:
                func = MODEL_MAP.get(key)
                if not func:
                    return safe_jsonify({"error": f"Unknown model key '{key}' received."}), 400
                ds_funcs.append(func)
            model_functions.append(ds_funcs)

        # --- Run the async fit function ---
        # Use asyncio.run() to execute the async function in the sync Flask route
        # Note: This blocks the Flask worker until the fit is done.
        # For production, consider Celery or other async task queues.
        try:
            # lm_fit_global is NOT async in the provided script, call directly
            fit_result_dict = lm_fit_global(
                sim_data,
                model_functions,
                initial_parameters,
                fit_options
            )
        except Exception as fit_exc:
             print(f"Exception during lm_fit_global call: {fit_exc}")
             traceback.print_exc()
             return safe_jsonify({"error": f"Fit execution failed: {fit_exc}"}), 500
        # --- End async fit ---

        # Return the result (safe_jsonify handles numpy arrays and NaN/Inf)
        return safe_jsonify(fit_result_dict)

    except Exception as e:
        print(f"Error in /fit: {e}")
        traceback.print_exc()
        return safe_jsonify({"error": f"Fit request failed on backend: {e}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # Use host='0.0.0.0' to make it accessible on your network if needed
    # Use debug=True for development (auto-reloads, provides debugger)
    # WARNING: Do NOT use debug=True in production!
    app.run(debug=True, host='127.0.0.1', port=5000)