from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss_classifier
from helper_evaluate import compute_epoch_loss_autoencoder
from helper_data_loader import read_sr_data, transform_data
import time
import torch
import torch.nn.functional as F

from collections import OrderedDict
import json
import subprocess
import sys
import xml.etree.ElementTree
import wandb
import random
import numpy as np
import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit


def get_data_complete(filepath, all_loader, model, device):
    data = read_sr_data(filepath)
    data["ax"] = np.linspace(3.5, 43.5, num=256)
    dict_result = get_est(all_loader, model, "cpu")
    data["lorentz"] = dict_result["lorentz"]
    data["raw"] = dict_result["raw"]
    data["freq_est"] = dict_result["freq"]
    data["decoded"] = dict_result["decoded"]
    data["freq_fit_first"] = get_fit_first_lorentz(data)
    return data

def lorentzian(x, A1, B1, C1, k):
    return  A1/(1 + ((x - B1)/(C1 / 2)) ** 2) + k

def get_fit_first_lorentz(data):
    # Obtain xdata and ydata
    xdata = np.reshape(data["ax"], (256))
    selected = np.logical_and(xdata > 6, xdata < 10 )
    xdata = xdata[selected]
    ydata = data["decoded"]
    freq_first = [];

    count = 0;
    data_decoded = np.reshape(data["decoded"], (-1, 256))

    for (index, ydata) in enumerate(data_decoded):
        ydata = np.reshape(ydata, (256))
        ydata = (signal.detrend(ydata))
        ydata = ydata[selected]

        # Initial guess of the parameters (you must find them some way!)
        pguess = [0.399961864586141, 7.78 , 0.0244699833220734,0];
        # Fit the data
        param_bounds=([0, 7.0, 0, 0],[np.inf,9,np.inf,0.1])
        try:
            popt, pcov = curve_fit(lorentzian, xdata,ydata , p0 = pguess, maxfev=5000, bounds = param_bounds)
        except RuntimeError:
            count = count + 1
        freq_first.append(popt[1])
            
    # Results
    return(np.array(freq_first))

def get_est(all_data_loader, model, device):
    batch_size = 1000;
    freq_result = [];
    decoded_result = [];
    raw = [];
    lorentz = [];
    model_cpu = model.to(device);
    #raw_chuncked = [raw_data[i:i+batch_size] for i in range(0,len(raw_data),batch_size)]
    for index_batch, (batch_data_raw, batch_data_lorentz,freq_lorentz, label_data) in enumerate(all_data_loader):
        model.to(device)
        with torch.no_grad():
            (encoder,_,_, frequencies, decoded) = model( batch_data_raw.to(device))
        freq_result.append(frequencies.detach().cpu().numpy() * 50)
        decoded_result.append(decoded.detach().cpu().numpy())
        raw.append(batch_data_raw.detach().cpu().numpy())
        lorentz.append(batch_data_lorentz.detach().cpu().numpy())
    freq_result = np.concatenate(freq_result)
    decoded_result = np.concatenate(decoded_result)
    raw = np.concatenate(raw)
    lorentz = np.concatenate(lorentz)
    return{"lorentz": lorentz, "raw": raw, "freq": freq_result, "decoded": decoded_result}
    
def train_autoencoder_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None , encoded_type = "LORENTZ"):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_ae_per_batch': [],
               'train_loss_freq_per_batch': [],
                'train_loss_per_epoch': [],
                'worst_train_loss_per_epoch': [],
                'worst_train_loss_ae_per_epoch': [],
                'worst_train_loss_freq_per_epoch': []
               }
    SR_mean = torch.FloatTensor([0.1541774, 0.2833938, 0.40509298, 0.5348934, 0.65381306, 0.78991985]).to(device)
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(train_loader):

            data_raw = data_raw.to(device)
            data_lorentz = data_lorentz.to(device)
            values_freq = values_freq.to(device)
            
            # FORWARD AND BACK PROP
            (encoded, _, _, est_freq, decoded) = model(data_raw)

            loss_freq = F.mse_loss(est_freq / SR_mean, values_freq / SR_mean) * 20
            if encoded_type == "LORENTZ":
                loss_ae = loss_fn(decoded, data_lorentz)  
            if encoded_type == "RAW":
                loss_ae = loss_fn(decoded, data_raw) 
            
            loss = loss_freq + loss_ae;
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            log_dict['train_loss_ae_per_batch'].append(loss_ae.item())
            log_dict['train_loss_freq_per_batch'].append(loss_freq.item())
            wandb.log({"loss": loss.item()})
            wandb.log({"loss_ae": loss_ae.item()})
            wandb.log({"loss_freq": loss_freq.item()})
        
        index = random.sample(range(len(worst_case_loader)), 5)
        worst_loss_freq = [];
        worst_loss_ae = [];
        worst_loss = []
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(worst_case_loader):
            if batch_idx in index:
                data_raw = data_raw.to(device)
                data_lorentz = data_lorentz.to(device)
                values_freq = values_freq.to(device)

                # FORWARD AND BACK PROP
                (encoded, _, _, est_freq, decoded) = model(data_raw)
                ind_loss_freq = F.mse_loss(est_freq / SR_mean, values_freq / SR_mean) * 20
                if encoded_type == "LORENTZ":
                    ind_loss_ae = loss_fn(decoded, data_lorentz)  
                if encoded_type == "RAW":
                    ind_loss_ae = loss_fn(decoded, data_raw) 
                ind_loss = ind_loss_freq + ind_loss_ae;

                worst_loss_freq.append(ind_loss_freq.item())#.to('cpu').numpy())
                worst_loss_ae.append(ind_loss_ae.item())#.to('cpu').numpy())
                worst_loss.append(ind_loss.item())#.to('cpu').numpy())
           
        worst_loss_freq =  np.mean(worst_loss_freq);
        worst_loss_ae =  np.mean(worst_loss_ae);
        worst_loss =  np.mean(worst_loss)
        
        log_dict['worst_train_loss_per_epoch'].append(worst_loss)
        log_dict['worst_train_loss_ae_per_epoch'].append(worst_loss_ae)
        log_dict['worst_train_loss_freq_per_epoch'].append(worst_loss_freq)
        wandb.log({"worst_loss": worst_loss})
        wandb.log({"worst_loss_ae": worst_loss_ae})
        wandb.log({"worst_loss_freq": worst_loss_freq})
        if not epoch % logging_interval:
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            print('Epocah: %03d/%03d | Loss: %.4f | Loss_freq: %.4f | Loss_ae: %.4f | Worst_Loss: %.4f | Worst_Loss_freq: %.4f | Worst_Loss_ae: %.4f |'
                  % (epoch+1, num_epochs, loss, loss_freq, loss_ae, worst_loss_freq, worst_loss_ae, worst_loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict
def train_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None, model_type = "AE" , encoded_type = "LORENTZ"):
    
    if model_type == "AE":
        log_dict = train_autoencoder_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn,
                         logging_interval, 
                         skip_epoch_stats,
                         save_model, encoded_type = encoded_type )
    if model_type == "VAE":
        log_dict = train_vae_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn,
                         logging_interval, 
                         skip_epoch_stats,
                         save_model, encoded_type = encoded_type )
    if model_type == "VQVAE":
        log_dict = train_vqvae_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn,
                         logging_interval, 
                         skip_epoch_stats,
                         save_model, encoded_type = encoded_type )        
    return log_dict;

def train_vae_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None, encoded_type = "LORENTZ"):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_ae_per_batch': [],
               'train_loss_freq_per_batch': [],
                'train_loss_per_epoch': [],
                'worst_train_loss_per_epoch': [],
                'worst_train_loss_ae_per_epoch': [],
                'worst_train_loss_freq_per_epoch': [],
                'train_loss_kl_per_batch': []
               }
    SR_mean = torch.FloatTensor([0.1541774, 0.2833938, 0.40509298, 0.5348934, 0.65381306, 0.78991985]).to(device)
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(train_loader):

            data_raw = data_raw.to(device)
            data_lorentz = data_lorentz.to(device)
            values_freq = values_freq.to(device)
            

            
            (encoded,z_mean, z_log_var, est_freq, decoded) = model(data_raw)
            
            kl_div = torch.mean( -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim =1 ), dim = 0);
            #batchsize = kl_div.size(0)
            #lantent_dim = z_mean.size()[1]
            #print(kl_div.size())
            kl_div = kl_div   * 0.001 #average over batch dimension
            if encoded_type == "LORENTZ":
                loss_ae = loss_fn(decoded, data_lorentz)  
            if encoded_type == "RAW":
                loss_ae = loss_fn(decoded, data_raw)  
            loss_freq = loss_fn(est_freq / SR_mean, values_freq / SR_mean)  * 200



            
            loss = loss_freq + loss_ae +  kl_div ;
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            log_dict['train_loss_ae_per_batch'].append(loss_ae.item())
            log_dict['train_loss_freq_per_batch'].append(loss_freq.item())
            log_dict['train_loss_kl_per_batch'].append(kl_div.item())
            wandb.log({"loss": loss.item()})
            wandb.log({"loss_ae": loss_ae.item()})
            wandb.log({"loss_freq": loss_freq.item()})
            wandb.log({"loss_kl": loss_freq.item()})
        
        index = random.sample(range(len(worst_case_loader)), 5)
        worst_loss_freq = [];
        worst_loss_ae = [];
        worst_loss = []
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(worst_case_loader):
            if batch_idx in index:
                data_raw = data_raw.to(device)
                data_lorentz = data_lorentz.to(device)
                values_freq = values_freq.to(device)

                # FORWARD AND BACK PROP
                (encoded, _, _, est_freq, decoded) = model(data_raw)
                
                ind_loss_freq = F.mse_loss(est_freq / SR_mean, values_freq / SR_mean) * 20
                
                if encoded_type == "LORENTZ":
                    ind_loss_ae = loss_fn(decoded, data_lorentz)  
                if encoded_type == "RAW":
                    ind_loss_ae = loss_fn(decoded, data_raw)  

                ind_loss = ind_loss_freq + ind_loss_ae;

                worst_loss_freq.append(ind_loss_freq.item())#.to('cpu').numpy())
                worst_loss_ae.append(ind_loss_ae.item())#.to('cpu').numpy())
                worst_loss.append(ind_loss.item())#.to('cpu').numpy())
           
        worst_loss_freq =  np.mean(worst_loss_freq);
        worst_loss_ae =  np.mean(worst_loss_ae);
        worst_loss =  np.mean(worst_loss)
        
        log_dict['worst_train_loss_per_epoch'].append(worst_loss)
        log_dict['worst_train_loss_ae_per_epoch'].append(worst_loss_ae)
        log_dict['worst_train_loss_freq_per_epoch'].append(worst_loss_freq)
        wandb.log({"worst_loss": worst_loss})
        wandb.log({"worst_loss_ae": worst_loss_ae})
        wandb.log({"worst_loss_freq": worst_loss_freq})
        if not epoch % logging_interval:
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            print('Epocah: %03d/%03d | Loss: %.4f | Loss_freq: %.4f | Loss_ae: %.4f | kl_div %.4f | Worst_Loss: %.4f | Worst_Loss_freq: %.4f | Worst_Loss_ae: %.4f |'
                  % (epoch+1, num_epochs, loss, loss_freq, loss_ae, kl_div, worst_loss_freq, worst_loss_ae, worst_loss))


        

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


def train_vqvae_v1(num_epochs, model, optimizer, device, 
                         train_loader,worst_case_loader, wandb,freq_mult, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None, encoded_type = "LORENTZ"):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_ae_per_batch': [],
               'train_loss_freq_per_batch': [],
                'train_loss_per_epoch': [],
                'worst_train_loss_per_epoch': [],
                'worst_train_loss_ae_per_epoch': [],
                'worst_train_loss_freq_per_epoch': [],
                'train_loss_vq_per_batch': [],
                'train_loss_perplexity_per_batch': []
               }
    SR_mean = torch.FloatTensor([0.1541774, 0.2833938, 0.40509298, 0.5348934, 0.65381306, 0.78991985]).to(device)
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(train_loader):

            data_raw = data_raw.to(device)
            data_lorentz = data_lorentz.to(device)
            values_freq = values_freq.to(device)
            
            (encoded,perplexity, vq_loss, est_freq, decoded) = model(data_raw)
            

            if encoded_type == "LORENTZ":
                loss_ae = loss_fn(decoded, data_lorentz)  
            if encoded_type == "RAW":
                loss_ae = loss_fn(decoded, data_raw)  
                
            loss_freq = loss_fn(est_freq / SR_mean, values_freq / SR_mean)  * 200

            loss = loss_freq + loss_ae +  vq_loss ;
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            log_dict['train_loss_ae_per_batch'].append(loss_ae.item())
            log_dict['train_loss_freq_per_batch'].append(loss_freq.item())
            log_dict['train_loss_vq_per_batch'].append(vq_loss.item())
            log_dict['train_loss_perplexity_per_batch'].append(perplexity.item())
            wandb.log({"loss": loss.item()})
            wandb.log({"loss_ae": loss_ae.item()})
            wandb.log({"loss_freq": loss_freq.item()})
            wandb.log({"loss_vq": vq_loss.item()})
            wandb.log({"perplexity": perplexity.item()})
        
        index = random.sample(range(len(worst_case_loader)), 5)
        worst_loss_freq = [];
        worst_loss_ae = [];
        worst_loss = []
        for batch_idx, (data_raw, data_lorentz, values_freq, _) in enumerate(worst_case_loader):
            if batch_idx in index:
                data_raw = data_raw.to(device)
                data_lorentz = data_lorentz.to(device)
                values_freq = values_freq.to(device)

                # FORWARD AND BACK PROP
                (encoded,_, vq_loss, est_freq, decoded) = model(data_raw)
                
                ind_loss_freq = F.mse_loss(est_freq / SR_mean, values_freq / SR_mean) * 20
                
                if encoded_type == "LORENTZ":
                    ind_loss_ae = loss_fn(decoded, data_lorentz)  
                if encoded_type == "RAW":
                    ind_loss_ae = loss_fn(decoded, data_raw)  

                ind_loss = ind_loss_freq + ind_loss_ae;

                worst_loss_freq.append(ind_loss_freq.item())#.to('cpu').numpy())
                worst_loss_ae.append(ind_loss_ae.item())#.to('cpu').numpy())
                worst_loss.append(ind_loss.item())#.to('cpu').numpy())
           
        worst_loss_freq =  np.mean(worst_loss_freq);
        worst_loss_ae =  np.mean(worst_loss_ae);
        worst_loss =  np.mean(worst_loss)
        
        log_dict['worst_train_loss_per_epoch'].append(worst_loss)
        log_dict['worst_train_loss_ae_per_epoch'].append(worst_loss_ae)
        log_dict['worst_train_loss_freq_per_epoch'].append(worst_loss_freq)
        wandb.log({"worst_loss": worst_loss})
        wandb.log({"worst_loss_ae": worst_loss_ae})
        wandb.log({"worst_loss_freq": worst_loss_freq})
        if not epoch % logging_interval:
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            print('Epocah: %03d/%03d | Loss: %.4f | Loss_freq: %.4f | Loss_ae: %.4f | vq: %.4f  | Worst_Loss: %.4f | Worst_Loss_freq: %.4f | Worst_Loss_ae: %.4f |'
                  % (epoch+1, num_epochs, loss, loss_freq, loss_ae,vq_loss,  worst_loss_freq, worst_loss_ae, worst_loss))


        

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict