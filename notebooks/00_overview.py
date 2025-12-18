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
from spectral import pipeline_raw
from spectral.helpers import style_cfg


# %%
# Initialize Driver
dr = driver.Driver({}, pipeline_raw)



# 2. Display
dr.display_all_functions(orient="TB", custom_style_function=style_cfg)

# %%
