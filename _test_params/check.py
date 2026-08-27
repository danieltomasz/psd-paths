from hamilton import driver
from spectral.flows import preprocessing
filter_params = {"l_freq":1.0,"h_freq":40.0,"h_trans_bandwidth":"auto",
                 "fir_window":"hamming","fir_design":"firwin","phase":"zero",
                 "picks":["ecg","eeg"]}
# deliberately pass NOTHING extra -> must reproduce the old hardcoded behaviour via defaults
dr = driver.Driver({}, preprocessing)
dr.execute(["raw_annotated_pyprep"],
           inputs={"subject_id":"105","fline":[50,100],"filter_params":filter_params})
print("DONE")
