import lm_global_fit as lm
import numpy as np

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
        result = lm.lm_fit_global(data_in, model_function_in, initial_parameters_in, fit_options)

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