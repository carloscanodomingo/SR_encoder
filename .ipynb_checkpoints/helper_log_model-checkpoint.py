from helper_plotting import plot_noise, plot_latent_space_with_labels, plot_frequencies, plot_loss, plot_freq_loss, plot_season, plot_training_loss, plot_cases, plot_residual_loss,plot_hist_freq
from  helper_metrics import selected_freq, corr_noise
import os

path = '/Users/krunal/Desktop/code/pyt/database'


def log_result(wandb,NUM_EPOCHS,model, device, data, log_dict,train_loader, test_loader, train_loader_worst,  model_type, encoded_type):
    initial_path = "images/"
    format_path = ".png"
    start_name =  model_type + "_" + encoded_type + "_";
    os.makedirs(initial_path, exist_ok=True)
    
    # Loss Images
    loss_path = initial_path + "loss/"
    os.makedirs(loss_path, exist_ok=True)
    path_loss_batch = loss_path + model_type + "_" + encoded_type + "_" + format_path
    plot_loss(log_dict, NUM_EPOCHS, model_type).savefig(path_loss_batch)
    wandb.log({"loss_img": wandb.Image(path_loss_batch)})

    # Residual Images
    start_path = initial_path + "Residual/"
    name = start_name + "residual_";
    os.makedirs(start_path, exist_ok=True)
    
    worst_path = start_path + "Worst/"
    os.makedirs(worst_path, exist_ok=True)
    
    path_worst_residual_freq_loss = worst_path + name + "worst_" + format_path;
    plot_residual_loss(train_loader_worst, model, device, show = False).savefig(path_worst_residual_freq_loss)
    wandb.log({"residual_worst": wandb.Image(path_worst_residual_freq_loss)})

    normal_path = start_path + "Normal/"
    os.makedirs(normal_path, exist_ok=True)
    
    path_test_residual_freq_loss = normal_path  + name + "normal_" + format_path;
    plot_residual_loss(test_loader, model, device,  show = False).savefig(path_test_residual_freq_loss)
    wandb.log({"residual_normal": wandb.Image(path_test_residual_freq_loss)})
    
    
    # QQplot Images
    start_path = initial_path + "Qqplot/"
    name = start_name + "qqplot_"
    os.makedirs(start_path, exist_ok=True)
    
    worst_path = start_path + "Worst/"
    os.makedirs(worst_path, exist_ok=True)
    
    path_worst_freq_qqplot = worst_path + name + "worst_" + format_path;
    plot_freq_loss(train_loader_worst, model, device,  show = False).savefig(path_worst_freq_qqplot)
    wandb.log({"qqplot_worst": wandb.Image(path_worst_freq_qqplot)})
    
    normal_path = start_path + "Normal/"
    os.makedirs(normal_path, exist_ok=True)
    
    path_train_freq_qqplot = normal_path  + name + "normal_" + format_path;
    plot_freq_loss(train_loader, model, device, show = False).savefig(path_train_freq_qqplot)
    wandb.log({"qqplot_normal": wandb.Image(path_train_freq_qqplot)})



    
    # Cases Images
    start_path = initial_path + "Cases/"
    name = start_name + "cases_"
    os.makedirs(start_path, exist_ok=True)
    
    worst_path = start_path + "Worst/"
    os.makedirs(worst_path, exist_ok=True)
    
    path_worst_cases = worst_path + name + "worst_" + format_path;
    plot_cases(train_loader_worst, model, device).savefig(path_worst_cases)
    wandb.log({"cases_worst": wandb.Image(path_worst_cases)})
    
    normal_path = start_path + "Normal/"
    os.makedirs(normal_path, exist_ok=True)

    path_test_cases = normal_path  + name + "normal_" + format_path;
    plot_cases(test_loader, model, device ).savefig(path_test_cases)
    wandb.log({"cases_normal": wandb.Image(path_test_cases)})

     # TestCase Freq 
    dict_freq = selected_freq(data)
    wandb.log(dict_freq)
    
    # TestCase Correlation 
    dict_corr = corr_noise(data)
    wandb.log(dict_corr)
    
    # Variance Images
    start_path = initial_path + "Variance/"
    name = start_name + "variance_"
    os.makedirs(start_path, exist_ok=True)

    path_variance = start_path + name + format_path;
    plot_noise(data).savefig(path_variance)
    wandb.log({"variance": wandb.Image(path_variance)})
    
    # Hist Images
    start_path = initial_path + "Hist/"
    name = start_name + "hist_"
    os.makedirs(start_path, exist_ok=True)

    path_hist_freq = start_path + name + format_path;
    plot_hist_freq(data).savefig(path_hist_freq)
    wandb.log({"hist": wandb.Image(path_hist_freq)})
    
    # Diurnal Images
    start_path = initial_path + "Diurnal/"
    name = start_name + "diurnal_"
    os.makedirs(start_path, exist_ok=True)
    
    for index_sr_mode in range(1,7):
        mode_path = initial_path + str(index_sr_mode) +  "/"
        name = name + str(index_sr_mode) +  "_"
        os.makedirs(mode_path, exist_ok=True)
        
        freq_qqplot = mode_path + name + format_path
        plot_season(data, sr_mode=index_sr_mode, show = False).savefig(freq_qqplot)
        wandb.log({ "diurnal_" + str(index_sr_mode) : wandb.Image(freq_qqplot)})