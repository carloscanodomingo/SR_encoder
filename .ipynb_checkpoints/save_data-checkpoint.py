
from helper_models import VQVAE, VAE, AutoEncoder, Reshape, Trim, VectorQuantizer
from helper_train import get_data_complete
import torch
# load pickle module
import pickle

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
    data = get_data_complete(filepath, model, "cpu")
    current_dict = {"model": model, "data": data}
    models_dict[model_name] = current_dict
        
# create a binary pickle file 
f = open("models.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(models_dict,f)

# close file
f.close()
