xlibrary(arrow)
library(dplyr)
library(lubridate)
library(tsibble)
library(ggplot2)
library(ggpubr)
# Libraries
#install.packages("hrbrthemes") 
library(tidyverse)
library(hrbrthemes)
#install.packages("latex2exp", dependencies=TRUE)
library(latex2exp)
library(viridis)
library(forcats)
#install.packages("fGarch") 
library(fGarch)
#install.packages("moments")
library(moments)
#install.packages("dgof")
library("dgof")
#install.packages("pracma")
library(pracma)
install.packages('robustbase')
library('robustbase')
install.packages("Metrics")
library(Metrics)
#install.packages("gamlss.nl")
library("gamlss.nl")

library("ggh4x")

#install.packages("kableExtra")
library(dplyr)
library(kableExtra)
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

install.packages("remotes")
remotes::install_github("teunbrand/ggh4x")

font_import()
loadfonts(device = "win")

source("C:/GIT/MATLAB/Paper 5/utils_result.R")

name_perc = "100"
path = paste("python_result", name_perc, ".feather", sep = "")
data <- arrow::read_feather(path)
data_ts <- data %>%
  as_tibble(
    index = datetime, 
    regular = FALSE
  )
rm("data")






a = quantile(data_ts$rmse , 0.1) 
index_good = data_ts$rmse < a & data_ts$selected == 1
current_index_good = which(index_good == TRUE)[1000]

low = quantile(data_ts$rmse , 0.5) 
up = quantile(data_ts$rmse , 0.75) 
index_normal = data_ts$rmse < up & data_ts$rmse > low & data_ts$selected == 1
current_index_normal = which(index_normal == TRUE)[1000]
index_worst = data_ts$rmse > up & data_ts$selected == 0
current_index_worst = which(index_worst == TRUE)[320]
#current_index_worst = which(index_worst == TRUE)[300]

plot_three_img(data_ts,current_index_good, current_index_normal, current_index_worst )

path = paste("IMG/case/", "three_", current_index_good,
             "_", current_index_normal, "_", current_index_worst, ".png", sep = "")
ggsave(path, width = 20, height = 20, units = "cm")


plot_case_freq(data_ts, current_index_worst)
path = paste("IMG/case/", "case_", current_index_worst, ".png", sep = "")
ggsave(path, width = 20, height = 20, units = "cm")


plot_case_freq(data_ts, current_index_normal)
path = paste("IMG/case/", "case_", current_index_normal, ".png", sep = "")
ggsave(path, width = 20, height = 20, units = "cm")

plot_rmse(data_ts, TRUE)
path = paste("IMG/metric/", "rmse_norm", ".png", sep = "")
ggsave(path, width = 20, height = 8, units = "cm")

plot_rmse(data_ts, FALSE)
path = paste("IMG/metric/", "rmse", ".png", sep = "")
ggsave(path, width = 20, height = 8, units = "cm")

data_long <- gather(data_ts, type, freqs,freq_names , factor_key=TRUE)
fit = data %>% 
  mutate(type =data_ts$selected) %>% 
  dplyr::filter(type == 1) %>%
  gather( type, freqs,freq_names , factor_key=TRUE) %>% 
  group_by(type) %>% 
  do(par = snormFit(.$freqs)$par, ks = ks.test(Lorentz$freqs, .$freqs, exact = FALSE)$statistic[[1]]) %>%
  mutate(mean = par[1], sd = par[2], xi = par[3]) %>%
  select(type, mean, sd, xi,ks)


bw = 0.02
data_long %>% 
  group_by(type) %>% 
  nest(data = c(freqs)) %>% 
  mutate(y = map(data, ~ dnorm(
    .$freqs, mean = mean(.$freqs), sd = sd(.$freqs)
  ) * bw * sum(!is.na(.$freqs)))) %>% 
  unnest(c(data,y)) %>%
  ggplot(aes(x = freqs, color = type, fill = type)) +
  geom_histogram(data = data_long, binwidth = bw, alpha=0.6) +
  geom_line(aes(y = y)) + 
  scale_fill_viridis(discrete=TRUE) +
  scale_color_viridis(discrete=TRUE) +
  ggtitle("Bin size = 3") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=15)
  ) + facet_wrap(~type) + 
  xlim(6.8, 8.8)


p <- data_long %>%
  #filter( price<300 ) %>%
  ggplot( aes(x=freqs, color = type, fill = type)) +
  geom_histogram( bins = 50, alpha=0.6) +
  scale_fill_viridis(discrete=TRUE) +
  scale_color_viridis(discrete=TRUE) +
  ggtitle("Bin size = 3") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=15)
  ) + facet_wrap(~type) + 
  xlim(6.8, 8.8)
p

testdata <- data.frame(X=rsn(100, omega=3, alpha=0.5))
mod <- selm(X ~ 1, data=testdata)

data_long <- gather(data, type, freqs,freq_names , factor_key=TRUE)
bw = 0.02



bw = 0.01
Lorentz_complete = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",1))
Lorentz = data.frame(type =data_ts$selected,freqs =  sapply(data_ts$loretnz_freq,"[[",1)) %>% 
  dplyr::filter(type == 1)
Lorentz %>%
  group_by(type) %>% 
  nest(data = c(freqs)) %>% 
  mutate(y = map(data, ~ dsnorm(
    .$freqs, mean = mean(.$freqs), sd = sd(.$freqs), xi=snormFit(.$freqs)$par["xi"]
  ) * 0.05 * sum(!is.na(.$freqs)))) %>% 
  unnest(c(data,y)) %>%
  ggplot(aes(x = freqs)) +
  geom_histogram(data = Lorentz, binwidth = bw, alpha=0.6) +
  geom_line(aes(y = y)) + 
  ggtitle("Bin size = 3") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=15)
  )+ 
  xlim(6.8, 8.8)

Lorentz_par = (Lorentz$freqs)


tstible_data <- as_tsibble(get_freq_data(data_ts, type = "decoded", freq) , index = time)


for (index_mode in c(1:6))
{
  plot_freq(data_ts, "decoded", index_mode)
  path = paste("IMG/hist/", "decoded_", index_mode, ".pdf", sep = "")
  ggsave(path, width = 20, height = 12, units = "cm")
}



plot_freq(data_ts, "fit", 1)
path = paste("IMG/hist/", "fit_", 1, ".png", sep = "")
ggsave(path, width = 20, height = 12, units = "cm")
for (index_sr in c(1:6))
{
  plot_freq(data_ts, "decoded", index_sr)
  path = paste("IMG/hist/", "decoded_", index_sr, ".png", sep = "")
  ggsave(path, width = 20, height = 8, units = "cm")
}


index_sr = 1
freq_names_select = c("VAE RAW", "VQVAE LORENTZ")
plt = plot_season(data_ts, c(1:3),freq_names_select, c(2016,2020))
path = paste("IMG/season/", "seasons_1_2_3", ".png", sep = "")
ggsave(path, width = 30, height = 35, units = "cm")

plt =plot_season(data_ts, c(4:6),freq_names_select, c(2016,2020))
path = paste("IMG/season/", "seasons_4_5_6", ".png", sep = "")
ggsave(path, width = 30, height = 35, units = "cm")


plot_hist_raw(data_ts, "all")
path = paste("IMG/hist/", "lorentz_all", ".png", sep = "")
ggsave(path, width = 20, height = 12, units = "cm")
data_freq <- get_freq_data(data_ts, "decoded", 2)
plot_hist_raw(data_ts, "Selected")
path = paste("IMG/hist/", "lorentz_selected", ".png", sep = "")
ggsave(path, width = 20, height = 12, units = "cm")
data_freq <- get_freq_data(data_ts, "decoded", 2)
plot_hist_raw(data_ts, "Not Selected")
path = paste("IMG/hist/", "lorentz_selected", ".png", sep = "")
ggsave(path, width = 20, height = 12, units = "cm")
data_freq <- get_freq_data(data_ts, "decoded", 2)

data_freq %>%
  mutate(across(freq_names, ~abs(.-Raw))) %>%
  dplyr::select(-c(Raw, time)) %>%
  gather( type, freqs,all_of(freq_names) , factor_key=TRUE) %>%
  ggplot(aes(x = freqs, y = rmse, color = type)) + 
  geom_smooth(method='lm', formula= y~x) + 
  facet_wrap(~ selected)

plot_rmse(data_ts)

data_freq = list();
SR_names = c("SR1","SR2","SR3","SR4","SR5","SR6")
for (index_sr in c(1:6))
{
  data_frame_freq = get_freq_data(data_ts, "decoded", index_sr)
  data_frame_freq$selected= data_ts$selected;
  data_frame_freq$rmse= data_ts$rmse;
  data_frame_freq$noise= data_ts$noise;
  
  data_frame_freq = data_frame_freq %>%
    mutate(across(freq_names, ~abs(.-Raw)))%>%
    dplyr::select(-c(Raw)) %>%
    gather( type, freqs,all_of(freq_names) , factor_key=TRUE)
  data_freq$type = data_frame_freq$type
  data_freq$rmse = data_frame_freq$rmse
  data_freq$noise = data_frame_freq$noise
  data_freq$selected = data_frame_freq$selected
  data_freq$time = data_frame_freq$time
  data_freq[[paste("SR",index_sr,sep = "")]] =data_frame_freq$freqs
}

data_freq = as.data.frame(data_freq)
test <- data_freq %>%
  mutate( hour_day = floor(hour(time) / 2)) %>%
  dplyr::select(-c(time)) %>%
  dplyr::filter(selected == 1) %>%
  gather( mode, freq_cor,all_of(SR_names) , factor_key=TRUE) %>%
  group_by(type, mode, hour_day) %>%
  summarise_all(mean, .groups = 'drop') %>%
  ggplot(aes(x = type, y = freq_cor, fill=type)) +
  geom_bar(stat="identity", color = "black") +
  facet_wrap(~ hour_day, nrow = 2, ncol = 6) +
  theme(
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 1))
plt + 
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
