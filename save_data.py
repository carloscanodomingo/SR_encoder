
from helper_models import VQVAE, VAE, AutoEncoder, Reshape, Trim, VectorQuantizer
from helper_train import get_data_complete, get_est
from helper_data_loader import get_dataloaders_mnist
import torch
# load pickle module
import pickle
import scipy.io

    
root = ""
data_path = 'data'
fit_sr = '\\SR_table_DL.mat'
datapath = filepath = root + data_path + fit_sr;
start_model_path = "models/100/"
end_model_path = ".model"
BATCH_SIZE = 32 * 20
_, _, _,all_loader  = get_dataloaders_mnist(filepath, batch_size = BATCH_SIZE,  case = "ALL")
models = ["AE_RAW", "VAE_RAW", "VQVAE_RAW", "AE_LORENTZ", "VAE_LORENTZ", "VQVAE_LORENTZ"]
models_struct = ["ae_raw", "vae_raw", "vqvae_raw", "ae_lorentz", "vae_lorentz", "vqvae_lorentz"]
models_dict = {}
for index_model in range(6):
    model_name = models[index_model] 
    model_path = start_model_path + model_name + end_model_path
    print(model_path)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    data = get_data_complete(filepath, all_loader, model, "cpu")
    current_dict = {"model": model, "data": data}
    models_dict[model_name] = current_dict
        
# create a binary pickle file 
f = open("models_enc.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(models_dict,f)
mdict = {}
for index_model in range(6):
    model_name = models[index_model] 
    models_dict_struct = "encoded_" + models_struct[index_model]
    mdict[models_dict_struct] =  models_dict[model_name]["data"]['encoded']
    
    
    
scipy.io.savemat('encoded.mat', mdict=mdict)
# close file
f.close()
