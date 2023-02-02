
library(ggplot2)
library(dplyr)
library(reshape2)
library(extrafont)
loadfonts(device = "win")
coulours_cases_old = c(Raw = "firebrick2", Lorentz = "deepskyblue2", 
                   AE = "seagreen3", VAE = "goldenrod1", VQVAE = "plum")


coulours_cases_other = c(Raw = "firebrick2", Lorentz = "deepskyblue2", 
                   'AE RAW' = "#3f5b0f", 'AE LORENTZ' = "#74D055FF",
                   'VAE RAW' = "#2D718EFF", 'VAE LORENTZ' = "#00dcf8",
                   'VQVAE RAW' = "#d38c02", 'VQVAE LORENTZ' = "#FDE725FF")

coulours_cases_viridis = c(Raw = "firebrick2", Lorentz = "deepskyblue2", 
                   'AE RAW' = "#2D718EFF", 'AE LORENTZ' = "#74D055FF",
                   'VAE RAW' = "#238A8DFF", 'VAE LORENTZ' = "#B8DE29FF",
                   'VQVAE RAW' = "#20A386FF", 'VQVAE LORENTZ' = "#FDE725FF")


coulours_cases = c(Raw = "#F8766D", Lorentz = "#00B9E3", 
                           'AE RAW' = "#D39200", 'AE LORENTZ' = "#93AA00",
                           'VAE RAW' = "#00BA38", 'VAE LORENTZ' = "#00C19F",
                           'VQVAE RAW' = "#DB72FB", 'VQVAE LORENTZ' = "#FF61C3")


lines_cases = c(Raw = "solid", Lorentz = "solid", 
                   'AE RAW' = "solid", 'AE LORENTZ' = "twodash",
                   'VAE RAW' = "solid", 'VAE LORENTZ' = "twodash",
                   'VQVAE RAW' = "solid", 'VQVAE LORENTZ' = "twodash")

labels_season = c("1" = "Winter","2" = "Spring","3" = "Summer","4" = "Autumn")
label_selected = c("0" = "Not Selected", "1" = "Selected")
labels_mode =c("1" = "SR 1", "2" = "SR 2", "3" = "SR 3", "4" = "SR 4", "5" = "SR 5", "6" = "SR 6")
AE_types = c("AE", "VAE","VQVAE");
signal_types = c("RAW", "LORENTZ")


freq_names = c( "AE RAW" , "VAE RAW","VQVAE RAW", "AE LORENTZ" , "VAE LORENTZ","VQVAE LORENTZ")
freq_names_data = c( "AE_RAW" , "VAE_RAW","VQVAE_RAW", "AE_LORENTZ" , "VAE_LORENTZ","VQVAE_LORENTZ")
freq_names_order = c("AE RAW", "AE LORENTZ" , "VAE RAW", "VAE LORENTZ","VQVAE RAW" ,"VQVAE LORENTZ")

SR_names = c("SR1","SR2","SR3","SR4","SR5","SR6")
all_print = c(freq_names, "Lorentz")

SR_freq_low = c(6.5, 13.2, 18.73, 24.70, 30.00, 38)
SR_freq = c(7.8, 14.2, 20.23, 26.70, 32.74, 39.63)
SR_freq_up = SR_freq + 0.75 *  (SR_freq - SR_freq_low)

plot_rmse <- function(data_ts, is_normalized)
{
  data_freq = list();
  SR_names = c("SR1","SR2","SR3","SR4","SR5","SR6")
  for (index_sr in c(1:6))
  {
    data_frame_freq = get_freq_data(data_ts, "decoded", index_sr)
    data_frame_freq$selected= data_ts$selected;
    data_frame_freq = data_frame_freq %>%
      group_by(selected)  %>%
      summarize(across(freq_names, ~rmse(.,Lorentz))) %>% 
      gather( type, freq_cor,all_of(freq_names) , factor_key=TRUE)
    if (is_normalized == TRUE)
      data_frame_freq = data_frame_freq %>%
      group_by(selected) %>%
      summarize(freq_cor = freq_cor/ max(freq_cor), type = type)

    data_freq$selected = data_frame_freq$selected
    data_freq$type = data_frame_freq$type
    data_freq[[paste("SR",index_sr,sep = "")]] =data_frame_freq$freq_cor
  }
  
  data_freq = as.data.frame(data_freq)
  
  data_freq %>%
    gather( mode, freq_cor,all_of(SR_names) , factor_key=TRUE) %>%
    ggplot(aes(x = type, y = freq_cor, fill=type)) +
    geom_bar(stat="identity", width = 0.7, position=position_dodge(width = 4)) +
    scale_fill_manual(values = coulours_cases[freq_names]) +
    facet_grid(selected ~ mode  ,labeller = labeller( selected = label_selected), scales = "free") +
    theme_bw() + 
    ylab("Normalized RMSE") +
    theme(
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.x = element_blank(),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 8),
    legend.margin=margin(1,1,1,1),
    legend.box.margin=margin(-10,-10,-1,-10)) + 
    guides(fill = guide_legend(override.aes = list(width = 0.2), nrow = 1))
}
plot_rmse_corr <- function(data_ts)
{
  data_freq = list();
  SR_names = c("SR1","SR2","SR3","SR4","SR5","SR6")
  for (index_sr in c(1:6))
  {
    data_frame_freq = get_freq_data(data_ts, "decoded", index_sr)
    data_frame_freq$selected= data_ts$selected;
    data_frame_freq$rmse= data_ts$rmse;
    data_frame_freq$noise= data_ts$noise;
    
    data_frame_freq = data_frame_freq %>%
      mutate(across(freq_names, ~abs(.-Lorentz)))%>%
      dplyr::select(-c(Lorentz, time)) %>%
      gather( type, freqs,all_of(freq_names) , factor_key=TRUE)
    data_freq$type = data_frame_freq$type
    data_freq$rmse = data_frame_freq$rmse
    data_freq$noise = data_frame_freq$noise
    data_freq$selected = data_frame_freq$selected
    data_freq[[paste("SR",index_sr,sep = "")]] =data_frame_freq$freqs
  }
  
  data_freq = as.data.frame(data_freq)
  
  test <- data_freq %>%
    gather( mode, diff_freq,all_of(SR_names) , factor_key=TRUE) %>%
    group_by(type, selected, mode) %>%
    #summarise(cor = mean(diff_freq), .groups = 'drop') %>%
    summarise(cor = cor(diff_freq, noise), .groups = 'drop') %>%
    ggplot(aes(x = type, y = cor, fill=type)) +
    geom_bar(stat="identity", color = "gray") +
    scale_fill_manual(values = coulours_cases) + 
    facet_grid(selected ~ mode  ,labeller = labeller( selected = label_selected), scales = "free") +
    theme(
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.x = element_blank(),
      legend.position = "bottom") +
    guides(fill = guide_legend(nrow = 1))
}

plot_case <- function(data_ts, type, index)
{
  freq = seq(3.5, 43.5, length.out = 256)
  select_row = data_ts[index,]

  
  current_data =  data.frame(unlist(select_row["AE_LORENTZ_raw"]));
  names(current_data) = "Raw";
  
  current_data$Lorentz = unlist(select_row["AE_LORENTZ_lorentz"]);
  
  

  if (type == "RAW")
  {
    current_data["AE RAW"]  = unlist(select_row[paste(AE_types[1], type, "decoded", sep = "_")]);
    current_data["VAE RAW"] = unlist(select_row[paste(AE_types[2], type, "decoded", sep = "_")]);
    current_data["VQVAE RAW"]= unlist(select_row[paste(AE_types[3], type, "decoded", sep = "_")]);
    current_data = current_data %>% relocate(Raw, .after = last_col())

    line_type = c(Raw = "solid", Lorentz = "dashed", 
                       'AE RAW' = "solid", 'AE LORENTZ' = "solid",
                       'VAE RAW' = "solid", 'VAE LORENTZ' = "solid",
                       'VQVAE RAW' = "solid", 'VQVAE LORENTZ' = "solid")
    
  }else if (type == "LORENTZ")
  {
    current_data["AE LORENTZ"]  = unlist(select_row[paste(AE_types[1], type, "decoded", sep = "_")]);
    current_data["VAE LORENTZ"] = unlist(select_row[paste(AE_types[2], type, "decoded", sep = "_")]);
    current_data["VQVAE LORENTZ"]= unlist(select_row[paste(AE_types[3], type, "decoded", sep = "_")]);
    
    current_data = current_data %>% relocate(Lorentz, .after = last_col())
    
    line_type = c(Raw = "dashed", Lorentz = "solid", 
                  'AE RAW' = "solid", 'AE LORENTZ' = "solid",
                  'VAE RAW' = "solid", 'VAE LORENTZ' = "solid",
                  'VQVAE RAW' = "solid", 'VQVAE LORENTZ' = "solid")
  }
  else 
  {
    warnings("Type not valid")
  }


  current_data$x = freq
  
  #melt data frame into long format
  df <- melt(current_data ,  id.vars = 'x', variable.name = 'Signal')
  
  
  ggplot(data = df, aes(x, value)) + 
    geom_line(aes(colour = Signal,linetype = Signal ),alpha = 0.75, size = 1.1, ) +
    scale_color_manual(values = coulours_cases) + 
    scale_linetype_manual(values = line_type) + 
    guides(color = guide_legend(override.aes = list(alpha = 1), nrow = 2))
}
get_signals = function(data_ts, index)
{
  freq = seq(3.5, 43.5, length.out = 256)
  select_row = data_ts[index,]
  
  current_data = list();
  current_data[["Raw"]] =  unlist(select_row["input_raw"]);

  current_data[["Lorentz"]] = unlist(select_row["input_lorentz"]);
  
  for (index_ae in c(1:length(freq_names_data)))
  {
    current_data[[freq_names[index_ae]]]  = unlist(select_row[paste(freq_names_data[index_ae], "decoded", sep = "_")]);
    
  }
  current_data$x = freq
  data_signals = as_tibble(current_data)
  #melt data frame into long format
  df <- melt(data_signals ,  id.vars = 'x', variable.name = 'Signal')
}
plot_case_img <- function(data_ts, index)
{
  plt_list = list()
  
  plt_list[[1]] = plot_case(data_ts,"RAW",  index)+ 
  labs(title  = type, subtitle =  "", caption = "") + 
    xlab("Frequency (Hz)") + 
    ylab(TeX("Normalized Intensity")) +
    theme(
      plot.title = element_text(hjust = 0.5, size=12),
      plot.subtitle = element_blank(),
      legend.title = element_text(hjust = 0.5, size=12),
      plot.margin=grid::unit(c(0,0,0,0), "mm")
    ) + 
    guides(color = guide_legend(override.aes = list(alpha = 1), nrow = 2))
  
  plt_list[[2]] = plot_case(data_ts,"LORENTZ",  index) + 
  labs(title  = type, subtitle =  "", caption = "") + 
    xlab("Frequency (Hz)") + 
    ylab(TeX("Normalized Intensity")) +
    theme(
      plot.title = element_text(hjust = 0.5, size=12),
      plot.subtitle = element_blank(),
      legend.title = element_text(hjust = 0.5, size=12),
      plot.margin=grid::unit(c(0,0,0,0), "mm"),
      axis.ticks.y=element_blank(),
      axis.text.y = element_blank(),
      axis.title.y=element_blank(),
      legend.position = "bottom"
    ) + 
    guides(color = guide_legend(override.aes = list(alpha = 1), nrow = 2))
  
  output_plot <- ggarrange(plotlist = plt_list, common.legend = TRUE, legend = "bottom") + 
    theme(legend.title = element_blank())
  output_plot
}
plot_three_img <- function(data_ts, index_good, index_normal, index_worst)
{
  plt_list = list()
  
  freq_names_raw = c("Raw", "AE RAW", "VAE RAW", "VQVAE RAW")
  freq_names_lorentz = c("Lorentz", "AE LORENTZ", "VAE LORENTZ", "VQVAE LORENTZ")
  signals_good = as_tibble(get_signals(data_ts, current_index_good))
  signals_good$type = "Low Noisy"
  signals_good %>% mutate(method = case_when(any(freq_names_raw ==(Signal))~ "Raw", any(Signal == freq_names_lorentz) ~ "Lorenz") )
  
  signals_normal = as_tibble(get_signals(data_ts, current_index_normal))
  signals_normal$type = "Averaged Noisy"
  signals_worst = as_tibble(get_signals(data_ts, current_index_worst))
  signals_worst$type = "Extremely Noisy"
  to_save_levelts =  c("Low Noisy", "Averaged Noisy", "Extremely Noisy")
  
  
  signals = signals_good %>%
    add_row(signals_normal) %>%
    add_row(signals_worst) %>%
    mutate(across(type))  
  signals$type <- factor(signals$type,      # Reordering group factor levels
                          levels = to_save_levelts)
  signals = signals %>%
    mutate(method = ifelse(is.element(Signal, freq_names_raw), "Raw","Lorentz") )

  ggplot(data = signals, aes(x, value)) +
    geom_line(aes(colour = Signal ),alpha = 0.75, size = 1.1, ) +
    scale_color_manual(values = coulours_cases) + 
    guides(color = guide_legend(override.aes = list(alpha = 1), nrow = 2)) +
    facet_grid(method ~ type ) +
    xlab("Frequency (Hz)") +
    ylab("Normalized Intensity") +
    theme(

      legend.position =  "bottom"
  )
    
}

print_table_freq <- function(data_ts, index)
{
  data_freq <- get_freqs(data_ts, index) %>%  relocate(Lorentz, .before = everything())
  rownames(data_freq) = c("SR 1","SR 2","SR 3","SR 4","SR 5","SR 6")
  name_table = c("Lorentz","AE" , "VAE","VQVAE", "AE" , "VAE","VQVAE")
  data_freq %>%
    mutate_if(is.numeric, format, digits=3,nsmall = 0)  %>% 
    kbl(caption=paste("Frequency summary of the six DL models applied to  - ", index, sep=""),
        format="latex",
        col.names = name_table,
        align="r") %>%
    kable_minimal(full_width = F,  html_font = "Source Sans Pro")
}
plot_case_freq <- function(data_ts, index)
{
  signal_raw = get_signals(data_ts, index) %>%
    dplyr::filter(Signal == "Raw") %>%
    dplyr::select(x, value)
  
  signal_lorentz = get_signals(data_ts, index) %>%
    dplyr::filter(Signal == "Lorentz") %>%
    dplyr::select(x, value)
  
  signals = get_signals(data_ts, index) %>%
    dplyr::filter(Signal != "Raw" & Signal != "Lorentz")
  data_freq <- get_freqs(data_ts, index)
  
  data_freq_raw = list()
  data_freq_raw$Lorentz = data_freq$Lorentz
  data_freq_raw$freq = data_freq$Lorentz
  data_freq_raw = as_tibble(data_freq_raw)
  

  
  data_freq_gathered <- data_freq %>%
    dplyr::select(-c(Lorentz)) %>%
    gather( Signal, freqs,, factor_key=TRUE)
    
    
    
  data_freq_gathered$Signal  = factor(data_freq_gathered$Signal, levels = freq_names_order)
  signals$Signal  = factor(signals$Signal, levels = freq_names_order)
  ggplot() + 
    geom_line(data = signal_raw,aes(x = x, y = value), color = coulours_cases["Raw"], size = 0.8, alpha = 0.8) + 
    geom_line(data = signal_lorentz,aes(x = x, y = value), color = coulours_cases["Lorentz"], size = 0.8, alpha = 0.5) + 
    geom_line(data = signals,aes(x = x, y = value, color = Signal), size = 1.2) + 
    geom_vline(data = data_freq_raw, aes(xintercept = Lorentz), color = "black", size = 0.8, lty = "longdash", alpha = 0.7) +
    geom_vline(data = data_freq_gathered, aes(xintercept = freqs, color = Signal), size = 1) +
    scale_color_manual(values = coulours_cases) + 
    facet_wrap(~Signal, ncol = 2) + 
    theme(
      legend.position = "none",

    ) +
    ylab(TeX("Normalized Intensity")) + 
    xlab("Frequency (Hz)") + 
    guides(color = guide_legend(nrow = 1), fill =
             guide_legend(nrow = 1), override.aes = list(linetype=c(1,1,1,1,1,1,1,1), shape=c(NA,NA,NA,NA,NA,NA,NA,NA)))
    
}

get_freqs <- function(data_ts, index)
{
  modes = data.frame()
  for (index_mode in c(1:6))
  {
    data_freq <- get_freq_data(data_ts, "decoded", index_mode) %>% 
      dplyr::select(-c(time))
    modes <- rbind(modes, as.data.frame(data_freq[index,]))
  }
  
  return(modes)
}


get_freq_data <- function(data_ts, type, index_freq)
{
  if (type == "decoded")
  {
    freq_names_find = apply(expand.grid(AE_types, signal_types, "freq"), 1, paste, collapse="_")
    find_freq = index_freq; 
  }
  else if(type == "fit")
  {
    freq_names_find = apply(expand.grid(AE_types, signal_types, "fitfreq"), 1, paste, collapse="_")
    find_freq = 1;
  }
  data=list()
  

  for (index in 1:length(freq_names_find)) {
    name = freq_names_find[index]
    data_frame_name = freq_names[index]
    data[[data_frame_name]] = (sapply(data_ts[[name]],"[[",find_freq))
  }
  
  data = data.frame(data)
  names(data) = freq_names
  data$Lorentz = (sapply(data_ts[["loretnz_freq"]],"[[",find_freq))
  data$time = data_ts$datetime
  return(data)
}

print_table_hist <- function(data_ts)
{
  fit = list()
  for (index_freq in c(1:7))
  {
    freq_names = apply(expand.grid(AE_types, signal_types), 1, paste, collapse=" ")
    if (index_freq <= 6)
    {
      data <- get_freq_data(data_ts, "decoded", index_freq) %>%
        dplyr::select(freq_names)
      
      lorentz_ref = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",index_freq)) %>% 
        dplyr::filter(type == 1)
      lorentz_ref_no = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",index_freq)) 
      ref = ks.test(lorentz_ref$freqs, lorentz_ref_no$freqs, exact = FALSE)$statistic[[1]]
      
    }else if(index_freq == 7)
    {
      data <- get_freq_data(data_ts, "fit", 1) %>%
        dplyr::select(freq_names)
      
      lorentz_ref = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",1)) %>% 
        dplyr::filter(type == 1)
      ref = 0
    }

  
    ks_vector = data %>% 
      mutate(type =data_ts$selected) %>% 
      dplyr::filter(type == 1) %>%
      gather( type, freqs,all_of(freq_names) , factor_key=TRUE) %>% 
      group_by(type) %>% 
      do(ks = unlist(ks.test(lorentz_ref$freqs, .$freqs, exact = FALSE)$statistic[[1]])) %>%
      dplyr::pull(ks)
    fit[[index_freq]] = c(ref, unlist(ks_vector))
  }
  
  name_table = c("Ref", "AE" , "VAE","VQVAE", "AE" , "VAE","VQVAE")
  
  names(fit) = c(SR_names, "Fit");
  hist_table = t(as_tibble(fit))
  colnames(hist_table) = c("Ref", freq_names)
  hist_table = as.data.frame(hist_table)
  rownames(hist_table) = c(SR_names, "Fit");
  
  hist_table %>%
    mutate_if(is.numeric, format, digits=2,nsmall = 0)  %>% 
    kbl(caption=paste("Frequency summary of the six DL models applied to  - ", sep=""),
        format="latex",
        col.names = name_table,
        align="r") %>%
    kable_minimal(full_width = F,  html_font = "Source Sans Pro")
}

plot_freq <- function(data_ts, type, freq)
{
  freq_names = apply(expand.grid(AE_types, signal_types), 1, paste, collapse=" ")
  data <- get_freq_data(data_ts, type, freq) %>%
    dplyr::select(freq_names)
  
  lorentz_ref = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",freq)) %>% 
    dplyr::filter(type == 1)
  
  par = snormFit(lorentz_ref$freqs)$par
  ref = list()
  ref$mean = par[1]
  ref$sd =  par[2]
  ref$xi = par[3]
  fit = data %>% 
    mutate(type =data_ts$selected) %>% 
    dplyr::filter(type == 1) %>%
    gather( type, freqs,all_of(freq_names) , factor_key=TRUE) %>% 
    group_by(type) %>% 
    do(par = snormFit(.$freqs)$par, ks = ks.test(lorentz_ref$freqs, .$freqs, exact = FALSE)$statistic[[1]]) %>%
    mutate(mean = par[1], sd = par[2], xi = par[3]) %>%
    dplyr::select(type, mean, sd, xi,ks)
  
  
  bw = 0.0032 * SR_freq[freq]
  data_long <- gather(data, type, freqs,freq_names , factor_key=TRUE)
  ks = unlist(fit$ks)
  print(ks)
  face_wrap_names =freq_names# c(paste(freq_names, round(ks,2), sep = " - "))
  names(face_wrap_names) = freq_names
  data_long %>% 
    group_by(type) %>% 
    nest(data = c(freqs)) %>% 
    mutate(y = map(data, ~ dsnorm(
      .$freqs, mean = ref$mean,
      sd = ref$sd,
      xi = ref$xi
    ) * bw * sum(!is.na(.$freqs)))) %>% 
    unnest(c(data,y)) %>%
    ggplot(aes(x = freqs, color = type, fill = type)) +
    geom_histogram(data = data_long, binwidth = bw, alpha=0.4) +
    geom_line(aes(y = y)) + 
    scale_fill_manual(values = coulours_cases[freq_names]) +
    scale_color_manual(values = coulours_cases[freq_names]) +
    theme_bw() +
    ylab("Observations") + 
    xlab("Frequency (Hz)") + 
    theme(
      plot.title = element_text(size=8, hjust = 0.5),
      legend.position = "none",
      legend.title = element_blank()
    ) + facet_wrap(~type, labeller = as_labeller(face_wrap_names)) +
    xlim(SR_freq[freq] - 1 - (1 * (freq - 1)/2) , SR_freq[freq] + 1 + (1 * (freq - 1)/2)) +
   guides(color = guide_legend(nrow = 1), fill = guide_legend(nrow = 1))
  }


plot_season <- function(data_ts, array_freq, models, years)
{
  tstible_data = tibble();
  for (freq in array_freq)
  {
    tstible_data_temp <- as_tsibble(get_freq_data(data_ts, type = "decoded", freq) , index = time) %>%
      dplyr::select(Lorentz, time, all_of(models))
    tstible_data_temp$selected = data_ts$selected;
    tstible_data_temp$mode = freq;
    if(length(tstible_data) == 0)
    {
      tstible_data <- tstible_data_temp
    }else
    {
      tstible_data <- add_row(tstible_data, tstible_data_temp)
    }
  }
  
  seleced_legend = c("Lorentz",models)
  ts_day_hours = tstible_data %>%
    filter_index((~ "2020")) %>%
    filter_index(as.character(years)[1] ~ as.character(years)[2]) %>%
    index_by(time_h = ~ lubridate::floor_date(., unit = "hour")) %>%
    group_by(time_h, selected, mode) %>%
    summarise_all(mean) %>%
    gather(type, freqs,any_of(all_print), factor_key=TRUE)  %>%
    mutate(season = ceiling(month(time_h) / 3 ) ) %>%
    dplyr::select(-(time))  %>%
    mutate(hours = hour(time_h)) %>%
    as_tibble(key = type) %>%
    group_by(season, hours, type, selected, mode) %>%
    summarise_all(mean) %>%
    dplyr::select(-(time_h)) %>%
    mutate(type = as.factor(type)) %>%
    ungroup()
  
  
  ts_day_hours_all = tstible_data %>%
    filter_index((~ "2020")) %>%
    filter_index(as.character(years)[1] ~ as.character(years)[2]) %>%
    index_by(time_h = ~ lubridate::floor_date(., unit = "hour")) %>%
    group_by(time_h) %>%
    summarise_all(mean) %>%
    gather(type, freqs,any_of(all_print), factor_key=TRUE)  %>%
    mutate(season = ceiling(month(time_h) / 3 ) ) %>%
    dplyr::select(-(time))  %>%
    mutate(hours = hour(time_h)) %>%
    as_tibble(key = type) %>%
    group_by(season, hours, type) %>%
    summarise_all(mean) %>%
    dplyr::select(-(time_h)) %>%
    mutate(type = as.factor(type)) %>%
    mutate(selected = 2) %>%
    ungroup()
  
  
  #ts_day_hours = ts_day_hours %>% add_row(ts_day_hours_all)
  
  ts_day_hours  %>%
    ggplot(aes(x = hours, color = type, linetype = type)) +
    geom_line(aes(y = freqs), alpha=0.6, size = 1.2) + 
    scale_color_manual(values = coulours_cases[seleced_legend]) + 
    scale_linetype_manual(values = lines_cases[seleced_legend]) + 
    #theme_ipsum() +
    theme_bw() + 
    xlab("Hour of the day")  +
    ylab("Frequency (Hz)") + 
    theme(
      plot.title = element_blank(),
      legend.position = "bottom",
    ) + facet_nested( mode + selected ~ season,
                    labeller = labeller(season = labels_season, selected = label_selected, mode = labels_mode),
                    scales="free_y") +
    theme(
      strip.background = element_rect(colour="black", fill = "white"),
      panel.spacing = unit(0.1, "lines"),
    ) +
    guides(color = guide_legend(override.aes = list(alpha = 1), 
                                linetype = guide_legend(override.aes = list(linetype = "solid")),
                                 nrow = 1)) + 
    facetted_pos_scales(
      y = list(
        mode == 1 ~ scale_y_continuous(limits = c(7.1, 8)),
        mode == 2 ~ scale_y_continuous(limits = c(13.8, 14.5)),
        mode == 3 ~ scale_y_continuous(limits = c(19.9, 20.75)),
        mode == 4 ~ scale_y_continuous(limits = c(26.3, 27.1)),
        mode == 5 ~ scale_y_continuous(limits = c(32, 33.3)),
        mode == 6 ~ scale_y_continuous(limits = c(38.5, 40))
      )
    )

}


plot_hist_raw <- function(data_ts, select)
{
  data_freq = list();
  SR_names = c("SR1","SR2","SR3","SR4","SR5","SR6")
  for (index_sr in c(1:6))
  {
    data_frame_freq = get_freq_data(data_ts, "decoded", index_sr)
    data_freq$selected = data_ts$selected
    data_freq[[paste("SR",index_sr,sep = "")]] =data_frame_freq$Lorentz
  }

  data_freq = as.data.frame(data_freq)
  bw = 0.05
  data_long <- gather(data_freq, mode, freqs, all_of(SR_names) , factor_key=TRUE) %>%
    dplyr::filter(selected < 2)

  if (select == "Selected")
  {
    data_long <- data_long %>% 
      dplyr::filter(selected == 1)
  }
  else if(select == "Not Selected")
  {
    data_long <- data_long %>% 
      dplyr::filter(selected == 0)
  }else if(select == "all")
  {
    data_select <- data_long %>% 
      dplyr::filter(selected == 1)
  }

        
  #face_wrap_names = c(paste(freq_names, round(ks,2), sep = " - "))
  #names(face_wrap_names) = freq_names
  data_long %>% 
    ggplot(aes(x = freqs)) +
    geom_histogram(data = data_long,aes(color = mode, fill = mode), binwidth = bw, alpha=0.4) +
    geom_histogram(data = data_select,fill = "white", binwidth = bw, alpha=0.6) +
    scale_fill_viridis(discrete=TRUE, option="inferno") +
    scale_color_viridis(discrete=TRUE, option="inferno") +
    theme_bw() +
    ylab("Observations") + 
    xlab("Frequency (Hz)") + 
    theme(
      plot.subtitle = element_blank(),
      plot.title = element_blank(),
      legend.position = "none",
      legend.title = element_blank(),
    ) + facet_wrap(~mode,  scales ="free") + 
    guides(color = guide_legend(nrow = 1), fill = guide_legend(nrow = 1))
}



trying_fit_not_working <- function()
{
  
  decoded = data_ts$raw
  example = list()
  example$x = seq(3.5, 43.5, length.out = 256)
  example$y = decoded[[1]]
  example$y2 = detrend(example$y, tt = 'linear', bp = c())
  example = data.frame(example)
  
  
  example = melt(example, id =  "x")
  
  ggplot(example, aes(x = x, y = value, color = variable)) + 
    geom_line( size = 1.2) + 
    scale_fill_viridis(discrete=TRUE) +
    scale_color_viridis(discrete=TRUE) 
  
  
  # lower <- 0; upper <- 10
  lower= c(-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,
           7.0,12,18,25,30,36,
           0,0,0,0,0,0)#0.5,0.5,0.5,0.5,0.5,0.5)
  
  upper <- c(Inf,Inf,Inf,Inf,Inf,Inf,
             9,16,23,29,36,45,
             Inf,Inf,Inf,Inf,Inf,Inf)
  
  start_point <- c(0.399961864586141,0.0451092487106278,0.201045341699827,0.0428628361556463,0.639163201462111,0.2518,  
                   7.8, 14.2, 20.23, 26.70, 32.74, 39.63,
                   0.0244699833220734,0.0397795442038889,0.529818635947425,0.933367035373425,0.602214607998047,0.601176727619887)
  nms <- c(paste0("A", 1:6), paste0("B", 1:6), paste0("C", 1:6))
  names(start_point) = nms
  names(lower) <- names(upper) <- nms
  f <- function(x, ...) {
    with(list(...), if (...length() == 1) A1/(1 + ((x - B1)/(C1/2))^2) + A2/(1 + ((x - B2)/(C2/2))^2) + A3/(1 + ((x - B3)/(C3/2))^2) + A4/(1 + ((x - B4)/(C4/2))^2) + A5/(1 + ((x - B5)/(C5/2))^2)  + A6/(1 + ((x - B6)/(C6/2))^2))
  }   
  fo <- sprintf("y ~ f(freq, %s)", toString(paste(nms, "=", nms)))
  
  to_fit = list()
  to_fit$freq = seq(3.5, 43.5, length.out = 256)
  detrend_data = detrend(decoded[[1]], tt = 'linear', bp = c()) 
  to_fit$y = detrend_data - min(detrend_data);
  
  nlc <- nls.control(maxiter = 100000, warnOnly = TRUE)
  
  a = nls(fo, data = to_fit, control = nlc,  start = start_point)
  
  
  fit = data %>% 
    mutate(type =data_ts$selected) %>% 
    dplyr::filter(type == 1) %>%
    gather( type, freqs,all_of(freq_names) , factor_key=TRUE) %>% 
    group_by(type) %>% 
    do(par = snormFit(.$freqs)$par, ks = ks.test(lorentz_ref$freqs, .$freqs, exact = FALSE)$statistic[[1]]) %>%
    mutate(mean = par[1], sd = par[2], xi = par[3]) %>%
    dplyr::select(type, mean, sd, xi,ks)
  
  
  nest(data = c(freqs)) %>% 
    mutate(y = map(data, ~ dsnorm(
      .$freqs, mean = mean(.$freqs[(.$freqs > SR_freq_low[freq]) & (.$freqs < SR_freq_up[freq])]),
      sd = sd(.$freqs[(.$freqs > SR_freq_low[freq]) & (.$freqs < SR_freq_up[freq])]),
      xi = snormFit(.$freqs[(.$freqs > SR_freq_low[freq]) & (.$freqs < SR_freq_up[freq])])$par["xi"]
    ) * bw * sum(!is.na(.$freqs)))) %>% 
    unnest(c(data,y))
    
}
