# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 23:25:21 2022

@author: Carlos Cano Domingo
"""
from helper_models import VQVAE, VAE, AutoEncoder, Reshape, Trim
from helper_train import get_data_complete
import torch
def save_data_model():
    root = ""
    data_path = 'data'
    fit_sr = '\\SR_table_DL.mat'
    datapath = filepath = root + data_path + fit_sr;
    start_model_path = "models/"
    end_model_path = ".model"
    models = ["AE_RAW", "VAE_RAW", "VQVAE_RAW", "AE_LORENTZ", "VAE_LORENTZ", "VQVAE_LORENTZ"]
    models_dict = {}
    for index_model in range(6):
        model_name = models[index_model] 
        model_path = start_model_path + model_name + end_model_path
        print(model_path)
        model = torch.load(model_path)
        models_dict[model_name] = model
        
    #model = torch.load(model_path + "AE_RAW.model")
    #data = get_data_complete(datapath, model, "cpu")
   # model_dict = {"model": model, "data": data}
    #models_dict = {"AE_RAW":model_dict}
    

    return models_dict