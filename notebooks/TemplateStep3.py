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

from hamilton import driver
from spectral.flows import ica_flow
from spectral.helpers import style_cfg


# Use Builder instead of Driver constructor
dr_ica = (
    driver.Builder()
    .with_config({
        "subject_id": "170",
        "append_to_existing": False,
        "ica_params": {
            "n_components": 0.99,
            "method": "picard",
            "max_components": 50
        },
        "threshold": 0.8,
        "artifact_types": ["eye blink", "heart beat", "muscle artifact"]
    })
    .with_modules(ica_flow)
    .build()
)
dr_ica.display_all_functions(orient="TB", custom_style_function=style_cfg)

# %%
results = dr_ica.execute(  final_vars=["ica_labels", "labeled_components", "plot_ica_components_all"])

# %%
display(results["plot_ica_components_all"])

# %%
import hamilton
print(hamilton.__version__)

# %%
# Use the driver object directly
dot = dr_ica.visualize_execution(
    final_vars=["plot_ica_components_all"],  # Specify the target node(s) here
    orient="TB",
    custom_style_function=style_cfg
)

# If you are in a Jupyter notebook, this will render it:
dot

