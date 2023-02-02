# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 00:24:09 2022

@author: Carlos Cano Domingo
"""

root = ""
data_path = 'data'
fit_sr = '\\SR_table_DL.mat'
datapath = filepath = root + data_path + fit_sr;

from helper_data_loader import read_sr_data
import numpy as np

 # Read data from MATLAB file
data = read_sr_data(filepath)

rmse_quantile_high = np.quantile(data["rmse"], 1.0)
rmse_quantile_low = np.quantile(data["rmse"], 0.8)
label_selected_bool = (data["rmse"]< rmse_quantile_high )[0] & (data["rmse"] > rmse_quantile_low )[0] & (data["selected"] == 0)[0]


print(np.sum((data["selected"] == 0)))
print(np.sum(label_selected_bool))