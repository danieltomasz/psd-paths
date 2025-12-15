# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.display import display
from hamilton import telemetry

telemetry.disable_telemetry()

from hamilton import driver
from spectral import mne_pipeline


def style_cfg(*, node, node_class): 
    """
    Returns: (style_dict, node_class, legend_label)
    """
    tag_value = node.tags.get("kind")

    # Style 1: Visualization (Gold)
    if tag_value == "visualization":
        return ({"style": "filled", "fillcolor": "#FFD700"}, node_class, "Plotting")
    
    # Style 2: Report (Pale Green) <--- NEW SECTION
    elif tag_value == "report":
        return ({"style": "filled", "fillcolor": "#98FB98"}, node_class, "Report")
    
    # Default: Data Processing (Blue/Grey default)
    return ({}, node_class, None)

filter_params = {
    "l_freq": 1.0,
    "h_freq": 45.0,
    "h_trans_bandwidth": "auto",
    "fir_window": "hamming",
    "fir_design": "firwin",
    "phase": "zero",
    "picks": ["ecg", "eeg"],
}

inputs = {"subject_id": "170", "fline": [50, 100], "filter_params": filter_params}

# Initialize Driver
dr = driver.Driver({}, mne_pipeline)



# 2. Display
dr.display_all_functions(orient="TB", custom_style_function=style_cfg)

# %%
results = dr.execute(
    final_vars=[ "raw_potatoed", "plot_raw_psd", "report_with_psd", "raw_filtered", "raw_annotated_pyprep", "plot_filtered_psd", "report_saved_path"],
    inputs=inputs,
)


# %%
fig = results["raw_potatoed"].iloc[0].plot(scalings='auto', show=True,  butterfly=True)

# %%
standard_scalings = {
    "eeg": 60e-6,  # 40 µV (Good for clean brainwaves)
    "ecg": 500e-6,  # 500 µV (ECG is naturally much larger)
    "eog": 150e-6,
}
fig = results["raw_filtered"].iloc[0].plot(scalings=standard_scalings, show=True,  butterfly=True)

# %%

display(results["plot_raw_psd"].iloc[0] )

# %%
display(results["plot_filtered_psd"].iloc[0] )

# %%
sensor_plot = results["raw_annotated_pyprep"].iloc[0].plot_sensors(show_names=True)

