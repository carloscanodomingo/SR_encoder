# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:29:09 2022

@author: Carlos Cano Domingo
"""

import pickle
from helper_data_loader import read_sr_data
import numpy as np
import pandas as pd
import feather

root = ""
data_path = 'data'
fit_sr = '\\SR_table_DL.mat'
datapath = filepath = root + data_path + fit_sr;

 # Read data from MATLAB file
data = read_sr_data(filepath)
with open("models_enc.pkl","rb") as handle_file:
    load_dict = pickle.load(handle_file)
    
value = next(iter(load_dict.values()))["data"]

d = {"datetime": data["datetime"],
     "rmse": data["rmse"][0], "noise": data["noise"][0],
     "selected": data["selected"][0], "raw": data["raw"].tolist(),
     "loretnz_freq":  (data["freq_values"] * 50).tolist(),
     "input_raw":    value["raw"][:,0,:].tolist(),
        "input_lorentz": value["lorentz"][:,0,:].tolist()}


for (key, value) in load_dict.items():
    current_value = value["data"]
    #[key + "_fitfreq"] = current_value["freq_fit_first"]
    d[key + "_encoded"] = current_value["encoded"][:,:].tolist()
    d[key + "_decoded"] = current_value["decoded"][:,0,:].tolist()
    d[key + "_freq"] = current_value["freq_est"].tolist()


df = pd.DataFrame(data = d)


path = 'python_result_encoded.feather'
#feather.write_dataframe(df, path)
#df.to_feather(path)