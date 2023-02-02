import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
import h5py
from scipy.signal import decimate
from itertools import compress
from sklearn import preprocessing    
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
class CustomImageDataset(Dataset):
    def __init__(self,data_raw,data_lorentz, freq, label, transform=None, target_transform=None):
        self.label = label
        self.data_raw = data_raw
        self.data_lorentz = data_lorentz
        self.freq = freq
        self.target_transform = target_transform
        self.transform = transform
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_raw = self.data_raw[idx, :]
        data_lorentz = self.data_lorentz[idx, :]
        freq = self.freq[idx,:]
        label = self.label[idx]
        if self.transform:
            data_raw = self.transform(data_raw)
            data_lorentz = self.transform(data_lorentz)
        if self.target_transform:
            label = self.target_transform(label)
        return data_raw, data_lorentz ,freq, label

    
def read_sr_data(filepath):

    file = h5py.File(filepath, 'r')
    
    data_raw = file.get('signal_raw')[()]
    data_raw = data_raw.transpose();
    
    data_lorentz = file.get('signal_lorentz')[()]
    data_lorentz = data_lorentz.transpose();
    
    label_min_f64 = file.get('label_min')[()]
    label_min = label_min_f64.astype(np.int32)
    label_min = label_min.transpose();

    label_hour_f64 = file.get('label_hour')[()]
    label_hour = label_hour_f64.astype(np.int32)
    label_hour = label_hour.transpose();

    label_day_f64 = file.get('label_day')[()]
    label_day = label_day_f64.astype(np.int32)
    label_day = label_day.transpose();

    label_month_f64 = file.get('label_month')[()]
    label_month = label_month_f64.astype(np.int32)
    label_month = label_month.transpose();

    label_year_f64 = file.get('label_year')[()]
    label_year = label_year_f64.astype(np.int32)
    label_year = label_year.transpose();

    n_len = len(label_year[0])
    seconds = np.zeros(n_len).astype(np.int32)
    

    data = np.array([label_year[0], label_month[0], label_day[0], label_hour[0], label_min[0],seconds])
    data = data.T
    date = [datetime(*x) for x in data]

    label_selected_f64 = file.get('label_selected')[()]
    label_selected = label_selected_f64.astype(np.int32)
    label_selected = label_selected.transpose();
    
    freq_values = file.get('label_freq')[()]
    freq_values = freq_values.astype(np.float32) / 50.0
    freq_values = freq_values.transpose()

    label_rmse_f64 = file.get('label_rmse_array')[()]
    label_rmse_f64 = label_rmse_f64.transpose();
    
    label_noise_f64 = file.get('label_noise')[()]
    label_noise_f64 = label_noise_f64.transpose();
    
    all_data = {"raw": data_raw,"lorentz": data_lorentz, "freq_values": freq_values, "datetime": date, "rmse" : label_rmse_f64, "selected": label_selected, "noise": label_noise_f64 };
    return all_data

def compress_data(data, label_selected_bool):
    return (np.array(list(compress(data, label_selected_bool))))

def transform_data(raw_data, lorentz_data):
    
     # Convert to float32
    x_raw = raw_data.astype('float32');
    x_lorentz = lorentz_data.astype('float32');
    data_raw = x_raw
    data_lorentz = x_lorentz

    # Normalize data 
    toguether = np.hstack((data_raw, data_lorentz))
    print(toguether.shape)
    # Normalize data 
    min_max_scaler = preprocessing.MinMaxScaler()

    X = min_max_scaler.fit_transform(toguether.transpose()).transpose()
    X = np.hsplit(X,2)
    x_raw = X[0]
    x_lorentz = X[1]
    x_raw = np.reshape(x_raw, (len(x_raw), 1, x_raw.shape[1])) 

    x_lorentz = np.reshape(x_lorentz, (len(x_lorentz), 1, x_lorentz.shape[1])) 
    

    return(x_raw, x_lorentz)
    
    
def get_dataloaders_mnist(filepath, batch_size = 64, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None, test_transforms=None, case = "NORMAL", test_size = 0.33, lorentz = True):
    
    # Read data from MATLAB file
    data = read_sr_data(filepath)
    
    # change to different cases
    if case == "NORMAL":
        rmse_quantile = np.quantile(data["rmse"], 1)
        label_selected_bool = (data["rmse"] < rmse_quantile)[0] & (data["selected"] == 1)[0]
    elif case == "WORST":
        rmse_quantile_high = np.quantile(data["rmse"], 1.0)
        rmse_quantile_low = np.quantile(data["rmse"], 0.8)
        label_selected_bool = (data["rmse"]< rmse_quantile_high )[0] & (data["rmse"] > rmse_quantile_low )[0] & (data["selected"] == 0)[0]
    else:
        label_selected_bool = np.full(data["rmse"].shape[1], True)
    # Remove extra dimension    
    print(label_selected_bool.shape)
    print(label_selected_bool.dtype)
    label_selected_bool = label_selected_bool

    # Extract hours and select label (ToDo -- Too Fixed)
    label_hour = map(lambda dates: dates.hour, data["datetime"])
    label_month = map(lambda dates: dates.month, data["datetime"])

    label_hour_pre = np.array(list(map(lambda hour: 0 if (hour <= 24 and hour >= 0) else 1, label_hour)))
    label_month_pre = np.array(list(map(lambda month: 0 if month < 3 else 1 if month < 6 else 2 if month < 9 else 3, label_month)))

    result = np.array(list(map(lambda hour,month: (hour + (month )), label_hour_pre, label_month_pre)))
    label = np.array(list(compress(result, label_selected_bool)))

    # Select frequencie valid following case
    freq_values = compress_data(data["freq_values"], label_selected_bool)
    
    # Select raw, lorentz and datetime valid following case
    data_raw_wide = compress_data(data["raw"], label_selected_bool)
    data_lorentz_wide = compress_data(data["lorentz"], label_selected_bool)

    
    data_raw, data_lorentz = transform_data(data_raw_wide, data_lorentz_wide);
    
        
    # Separate the train dataset from the test dataset
    x_raw_train, x_raw_test, x_lorentz_train, x_lorentz_test,freq_values_train, freq_values_test, label_train, label_test = train_test_split(
        data_raw, data_lorentz,freq_values,  label, test_size=test_size, random_state=42)
    
    # Create Custom Data set
    training_data = CustomImageDataset(
    x_raw_train,
    x_lorentz_train,
    freq_values_train,
    label_train
    )

    validation_data = CustomImageDataset(
        x_raw_test,
        x_lorentz_test,
        freq_values_test,
        label_test,
    )
    
    # Create DataLoader
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True )
    indices = torch.randperm(len(training_data))[:20]
    fixed_loader = Subset(training_data, indices)
    
    all_dataset = CustomImageDataset(
    data_raw,
    data_lorentz,
    freq_values,
    label
    )
    
    # Create DataLoader
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, drop_last=False )
    
    test_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader, fixed_loader, all_loader