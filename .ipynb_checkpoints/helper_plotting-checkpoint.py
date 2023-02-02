import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from helper_data_loader import transform_data
from helper_train import get_est
import matplotlib.dates
from scipy.signal import savgol_filter
# Load in convex hull method
from scipy.spatial import ConvexHull
# Import pandas library
from sklearn.metrics import mean_squared_error
import pandas as pd
def plot_freq_loss(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 10), n_images=15,  show = False):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.show(False)
    fig.tight_layout()
    d = [];
    f = [];
    model.to(device)
    with torch.no_grad():
            for i, (features,_, values_freq, targets) in enumerate(data_loader):
                features = features.to(device)
                targets = targets.to(device)
                values_freq = values_freq.to(device)

                (_,_,_,freq_est,_) = model(features)

                d.append(freq_est.to('cpu').numpy())
                f.append(values_freq.to('cpu').numpy())
    d = np.concatenate(d)
    f = np.concatenate(f)
    for index in range(0,6):
        axes[index // 2, index  % 2].scatter(
            f[:, index] * 50, d[:, index] * 50,
            alpha=0.5)
        axes[index // 2, index  % 2].set_title("SR_MODE: " + str(index + 1))
    return fig 


def plot_residual_loss(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 10), n_images=15, show = False):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.show(False)
    fig.tight_layout(h_pad = 2.0)
    x = [];
    y = [];
    model.to(device)
    with torch.no_grad():
            for i, (features,_, values_freq, targets) in enumerate(data_loader):
                features = features.to(device)
                targets = targets.to(device)
                lorentz_freq = values_freq.to(device)
                
                (_,_,_,freq_est,_) = model(features)

                y.append(freq_est.to('cpu').numpy() - lorentz_freq.to('cpu').numpy())
                x.append(freq_est.to('cpu').numpy())
    x = np.concatenate(x)
    y = np.concatenate(y)


    for index_row in range(0,3):
        for index_col in range(0,2):
            index = index_col + index_row * 2
            points = np.c_[x[:, index] * 50,  y[:, index] * 50]
            # Calculate the position of the points in the convex hull
            hull = ConvexHull(points)
            axes[index_row, index_col].scatter(
            np.delete(points[:,0], hull.vertices),np.delete(points[:,1], hull.vertices),
            alpha=0.2, marker ="o", s = 25)
            axes[index_row, index_col].set_title("SR mode " + str(index + 1))
        if index_col == 0:
            axes[index_row, index_col].set_ylabel("Residual")
        else:
            axes[index_row, index_col].set_ylabel("")
        if index_row == 2:
            axes[index_row, index_col].set_xlabel("Estimated Freq (Hz)")
        else:
            axes[index_row, index_col].set_xlabel("")
        
    return fig 
def plot_cases(data_loader, model,device,  figsize=(25, 20)):
    x = np.linspace(3.5, 43.5, num=256)
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=figsize)
    for index_batch, (batch_data_raw, batch_data_lorentz,freq_lorentz, label_data) in enumerate(data_loader):
        model.to(device)
        with torch.no_grad():
            (encoder_plot,_,_, frequencies, decoder_plt) = model( batch_data_raw.to(device))
                
                
        ax1 = axes[index_batch, 0]
        ax2 = axes[index_batch, 1]
        ax3 = axes[index_batch, 2]
        ax1.set_title('Raw vs Lorentz')
        ax1.plot(x, batch_data_raw[1,].squeeze().numpy(), color = 'red')
        ax1.plot(x, batch_data_lorentz[1,].squeeze().numpy(), color = 'blue')
        ax1.vlines(freq_lorentz[1,] * 50, 0, 1, color = "blue");
        ax1.vlines(frequencies[1,].to("cpu").numpy() * 50, 0, 1,linestyles ="dotted", color = "green");
        
        ax2.set_title('Lorentz vs Decoded')
        ax2.plot(x, decoder_plt[1,0,:].to('cpu').numpy(), color = 'green')
        ax2.plot(x, batch_data_lorentz[1,].squeeze().numpy(), color = 'blue')
        #ax2.vlines(frequencies[1,].to("cpu").numpy() * 50, 0, 1,linestyles ="dotted", color = "green");
        #ax2.vlines(freq_lorentz[1,] * 50, 0, 1 ,linestyles ="dashed",color = "blue");
        
        ax3.set_title('Raw vs Decoded')
        ax3.plot(x, batch_data_raw[1,].squeeze().numpy(), color = 'red')
        ax3.plot(x, decoder_plt[1,0,:].to('cpu').numpy(), color = 'green')
        ax3.vlines(frequencies[1,].to("cpu").numpy() * 50, 0, 1, color = "green");
        ax3.vlines(freq_lorentz[1,] * 50, 0, 1 ,linestyles ="dotted",color = "blue");
        
        if index_batch + 1 == 5:
            break;
    return fig
        
def plot_loss(log_dict, NUM_EPOCHS, model_type = "AE"):  
    if model_type == "AE" or model_type == "VQVAE":
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,10))
        plot_training_loss(axes[0,0], log_dict['train_loss_per_batch'], NUM_EPOCHS)
        axes[0,0].set_title('Loss Total Per batch')
        plot_training_loss(axes[0,1],log_dict['train_loss_ae_per_batch'], NUM_EPOCHS)
        axes[0,1].set_title('Loss AE Per Batch')
        plot_training_loss(axes[0,2],log_dict['train_loss_freq_per_batch'], NUM_EPOCHS)
        axes[0,2].set_title('Loss Freq Per Batch')
        plot_training_loss(axes[1,0],log_dict['worst_train_loss_per_epoch'], NUM_EPOCHS)
        axes[1,0].set_title('Worst Loss Total Per Epoch')
        plot_training_loss(axes[1,1],log_dict['worst_train_loss_ae_per_epoch'], NUM_EPOCHS)
        axes[1,1].set_title('Worst Loss AE Per Epoch')
        plot_training_loss(axes[1,2],log_dict['worst_train_loss_freq_per_epoch'], NUM_EPOCHS)
        axes[1,2].set_title('Worst Loss Freq Per Epoch')
    if model_type == "VAE":
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        plot_training_loss(axes[0,0], log_dict['train_loss_per_batch'], NUM_EPOCHS)
        axes[0,0].set_title('Loss Total Per batch')
        plot_training_loss(axes[0,1],log_dict['train_loss_ae_per_batch'], NUM_EPOCHS)
        axes[0,1].set_title('Loss AE Per Batch')
        plot_training_loss(axes[0,2],log_dict['train_loss_freq_per_batch'], NUM_EPOCHS)
        axes[0,2].set_title('Loss Freq Per Batch')
        plot_training_loss(axes[0,3],log_dict['train_loss_kl_per_batch'], NUM_EPOCHS)
        axes[0,2].set_title('Loss Freq Per Batch')
        plot_training_loss(axes[1,0],log_dict['worst_train_loss_per_epoch'], NUM_EPOCHS)
        axes[1,0].set_title('Worst Loss Total Per Epoch')
        plot_training_loss(axes[1,1],log_dict['worst_train_loss_ae_per_epoch'], NUM_EPOCHS)
        axes[1,1].set_title('Worst Loss AE Per Epoch')
        plot_training_loss(axes[1,2],log_dict['worst_train_loss_freq_per_epoch'], NUM_EPOCHS)
        axes[1,2].set_title('Worst Loss Freq Per Epoch')
    return fig 
        
def plot_frequencies(data, model, figsize = (20,30), modeltype ='autoencoder'):
    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=figsize)
    dates_num = matplotlib.dates.date2num(data["datetime"]) 
    with torch.no_grad():
        for index_mode in range(0,6):
            axes[index_mode * 2].plot_date(dates_num, savgol_filter(data["freq_values"][:,index_mode] * 50,( 24 * 2 * 15) + 1, 5) )
            axes[index_mode * 2].set_title('Freq Lorentz')
            axes[index_mode * 2 + 1].plot_date(dates_num, savgol_filter(data["freq_est"][:,index_mode],( 24 * 2 * 15) + 1, 5))
            axes[index_mode * 2 + 1].set_title('Freq Regression')
        plt.legend()
        plt.show()
    return fig 

def plot_noise(data, figsize = (20,10), modeltype ='autoencoder'):
    fig, axes = plt.subplots(nrows = 2, ncols=1, figsize=figsize)
    dates_num = matplotlib.dates.date2num(data["datetime"]) 
    fig.tight_layout(h_pad  = 2)
    sr_correlation = []
    x = data["noise"]
    
    freq_est = data["freq_est"][:,0];
    freq_lorentz = data["freq_values"][:,0] * 50
    error_freq = np.absolute(freq_lorentz - freq_est)
    
    print(x.shape)
    #obtain m (slope) and b(intercept) of linear regression line
    #m, b = np.polyfit(np.reshape(x, -1), error_freq, 1)
    #use red as color for regression line
    #axes[0].plot(x, m*x+b, color='red')
    axes[0].scatter(
        x, error_freq,
        alpha=0.1, marker ="o", s = 15)
    axes[0].set_xlabel("Variability of Raw Signal SR Mode 1", fontsize=16)
    axes[0].set_ylabel("Freq Error", fontsize=16)
    axes[0].set_xlim((0, 2))
    axes[0].set_ylim((0, 1.5))
    
    raw = np.reshape(data["raw"],(-1, 256))
    lorentz = np.reshape(data["decoded"],(-1, 256))
    rmse_value = np.array( mean_squared_error(raw.transpose(), lorentz.transpose(), multioutput='raw_values'))  
    
    #obtain m (slope) and b(intercept) of linear regression line
    #m, b = np.polyfit(np.reshape(x, -1), rmse_value, 1)
    #use red as color for regression line
    #axes[1].plot(x, m*x+b, color='red')
    axes[1].scatter(
        data["noise"], rmse_value,
        alpha=0.1, marker ="o", s = 15)
    axes[1].set_xlabel("Variability of Raw Signal", fontsize=16)
    axes[1].set_xlim((0, 2))
    axes[1].set_ylim((0, 0.02))
    axes[1].set_ylabel("RMSE", fontsize=16)
    plt.legend()
    plt.show()
    return fig 

def plot_hist_freq(data, show = True):
    est = data["freq_est"][:,0];
    lorentz = data["freq_values"][:,0] * 50
    fit = data["freq_fit_first"];
    fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(20,8))
    fig.tight_layout()
    xlim = [6.8, 9.2]
    n_bins = "auto"
    axes[0].hist(lorentz, bins=n_bins, density= True)
    axes[0].set_title("Lorentz")
    axes[0].set_ylabel("Freq", fontsize=16)
    axes[0].set_xlim(xlim)  
    
    axes[1].hist(fit, bins=n_bins, density= True)
    axes[1].set_title("Adjusted from Encoded")
    axes[1].set_ylabel("Freq", fontsize=16)
    axes[1].set_xlim(xlim)  
    
    axes[2].hist(est, bins=n_bins, density= True)
    axes[2].set_title("DL Regresion")
    axes[2].set_ylabel("Freq", fontsize=16)
    axes[2].set_xlim(xlim)  
    return fig


def plot_season(data, sr_mode=1, show = True):
    SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"];
    labels = ["Lorentz", "Estimated"]
    LORENTZ_NAME = [ 'LORENTZ_{}'.format(x) for x in range(1,7)]
    EST_NAME = [ 'EST_{}'.format(x) for x in range(1,7)]
    df1 = pd.DataFrame(data["freq_values"] * 50, columns = LORENTZ_NAME)
    df2 = pd.DataFrame(data["freq_est"], columns = EST_NAME)
    df3 = pd.DataFrame(data["datetime"], columns = ["datetime"])
    frames = [df3, df1, df2]
    df = pd.DataFrame(pd.concat(frames, axis = 1))
    df.set_index('datetime')
    df_mean = df
    df_mean['hour'] = df_mean.datetime.dt.hour
    df_mean['month'] = df_mean.datetime.dt.month
    df_mean['year'] = df_mean.datetime.dt.year
    df_group = df_mean.groupby( [df_mean.year, (df_mean.month // 4), df_mean.hour]).mean()

    LORENTZ_NAME = 'LORENTZ_' + str(sr_mode)
    EST_NAME = 'EST_' + str(sr_mode)
    df_group = pd.DataFrame(df_group.filter(items=[LORENTZ_NAME, EST_NAME]))

    fig, axes = plt.subplots(nrows = 5, ncols=4, figsize=(20,20),sharey=True)
    fig.show(show)
    #fig.suptitle("Diurnal Frequency Variation of SR mode " + str(sr_mode), fontsize=16)
    fig.tight_layout()
    for index_year in range(0, 5):
        for index_month in range(0,4):
            current_ax = axes[index_year, index_month]
            df_group.loc[(2016 + index_year, index_month)].plot(ax = current_ax)
            current_ax.legend(labels = labels)
            if index_year == 0:
                current_ax.set_title(SEASONS[index_month])
            current_ax.set_ylabel(str(index_year + 2016) + " - Freq (Hz)", fontsize=16)
            if index_year == 4:
                current_ax.set_xlabel("Hour of the day", fontsize=16)
            else: 
                current_ax.set_xlabel("")
    return fig 
            
def plot_training_loss(ax1, minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    num_losses = len(minibatch_losses) // 2
    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    
    
def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    
def plot_generated_images(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):

     
    
    for batch_idx, (features, _) in enumerate(data_loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
                
def plot_latent_space_with_labels(data_loader, model, device):
        d = [];
        model.eval()
        with torch.no_grad():
            for i, (features,_, _, targets) in enumerate(data_loader):

                features = features.to(device)
                targets = targets.to(device)


                embedding = model.encoder(features)


                d.append(embedding.to('cpu').numpy())
            d = np.concatenate(d)
            print(d[1])
            plt.scatter(
                d[:, 0], d[:, 1],
                alpha=0.5)

        plt.legend()
        plt.show()

    
    
def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decoder(rand_features)
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary') 