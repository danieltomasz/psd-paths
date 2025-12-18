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
from IPython.display import display
from hamilton import telemetry

telemetry.disable_telemetry()

from hamilton import driver
from spectral.flows import epoching, ica_flow 
from spectral.helpers import style_cfg

# Initialize Driver
dr = driver.Driver({}, epoching)

# 2. Display
dr.display_all_functions(orient="TB", custom_style_function=style_cfg)

# %%
results = dr.execute(
    final_vars=["plot_epoch_rejection", "report_epochs_saved_path","saved_epochs_path"],
    inputs={
        "subject_id": "170",
        "append_to_existing": False
    },
)

# %%
display(results["plot_epoch_rejection"].iloc[0])

# %%
from hamilton import driver
from spectral.flows import ica_flow


dr_ica = driver.Driver(
    {  # config as first positional argument (no "config=")
        "subject_id": "170",
        "append_to_existing": False,
        "ica_params": {
            "n_components": 0.99,
            "method": "picard",
            "max_components": 50
        },
        "threshold": 0.8,
        "artifact_types": ["eye blink", "heart beat", "muscle artifact"]
    },
    ica_flow,  # modules as positional arguments
)
dr_ica.display_all_functions(orient="TB", custom_style_function=style_cfg)




# %%
results_ica = dr_ica.execute(
    final_vars=[
        "saved_ica_path",
        "labeled_components",
        "artifact_components",
        "report_with_ica",
    ]
)

print("Artifact components:", results_ica["artifact_components"])

# %%
results_ica = dr_ica.execute(
    final_vars=[
        "saved_ica_path",
        "labeled_components",
        "artifact_components",
        "report_with_ica",
    ],
    inputs={
        "subject_id": "170",
        "append_to_existing": False,
        "ica_params": {"n_components": 0.99, "method": "picard", "max_components": 50},
        "threshold": 0.8,  # ICLabel probability threshold
        "artifact_types": ["eye blink", "heart beat", "muscle artifact"],
    },
)
print("Artifact components:", results["artifact_components"])

