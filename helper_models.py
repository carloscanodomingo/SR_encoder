
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_utils import Reshape, Trim

    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        #print(input_shape)
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings
class VQVAE(nn.Module):
    def __init__(self, num_hiddens, kernel_size, data_len, latent_dim, regression_hidden, num_embeddings, dropout = 0, commitment_cost = 0.25):
        super().__init__()
        self.encoder = nn.Sequential( #784
            nn.Conv1d(in_channels=1,
                  out_channels=num_hiddens // 2,
                  kernel_size=kernel_size, stride=2, padding = kernel_size  // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=num_hiddens // 2,
                  out_channels=num_hiddens,
                  kernel_size=kernel_size // 2, stride=2, padding = kernel_size  // 2 // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels=num_hiddens,
                  out_channels=num_hiddens,
                  kernel_size=kernel_size // 4, stride=2, padding = kernel_size  // 4 // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
        )
        self.pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, 
                                  out_channels=latent_dim,
                                  kernel_size=1, 
                                  stride=1)
        
        self.vq_vae = VectorQuantizer(num_embeddings, latent_dim,
                                       commitment_cost)
        self.regresion = nn.Sequential(
                nn.Flatten(),
                nn.Linear(latent_dim * (data_len // (2 ** 3)), regression_hidden),
                nn.Linear(regression_hidden , 6),
                nn.Sigmoid()
                )
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim,
                         out_channels=num_hiddens,
                         kernel_size=3, 
                         stride=1, padding=1),
            #torch.nn.Linear(num_hiddens, num_hiddens * (data_len // (2 ** 3))),
            Reshape(-1, num_hiddens, (data_len // (2 ** 3))),
            nn.ConvTranspose1d(in_channels=num_hiddens,
                  out_channels=num_hiddens,
                  kernel_size=kernel_size // 4, stride=2, padding = kernel_size // 4 // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose1d(in_channels=num_hiddens,
                  out_channels=num_hiddens // 2,
                  kernel_size=kernel_size // 2, stride=2, padding = kernel_size // 2 // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                  out_channels=1,
                  kernel_size=kernel_size , stride=2, padding = kernel_size  // 2 - 1, bias=False),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            Trim(),  # 1x29x29 -> 1x28x28
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        z = self.pre_vq_conv(x)
        loss, encoded, perplexity, encoding = self.vq_vae(z)
        #print(encoded.shape)
        freq = self.regresion(encoded)
        decoded = self.decoder(encoded)
        return encoded,perplexity,loss,freq, decoded
    
    
    
    
class AutoEncoder(nn.Module):
    def __init__(self, num_hiddens, kernel_size, data_len, latent_dim, regression_hidden, dropout):
        super().__init__()
        
        self.encoder = nn.Sequential( #784
                nn.Conv1d(in_channels=1,
                      out_channels=num_hiddens // 2,
                      kernel_size=kernel_size, stride=2, padding = kernel_size  // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens // 2),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                nn.Conv1d(in_channels=num_hiddens // 2,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 2, stride=2, padding = kernel_size  // 2 // 2 - 1, bias=False),
                nn.LeakyReLU(0.01),
                nn.Conv1d(in_channels=num_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 4, stride=2, padding = kernel_size  // 4 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens),
                nn.LeakyReLU(0.01),
                nn.Flatten(),
                nn.Linear(num_hiddens * (data_len // (2 ** 3)), latent_dim),
        )
        self.regresion = nn.Sequential(
                nn.Linear(latent_dim , regression_hidden),
                nn.Linear(regression_hidden , 6),
                nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
                torch.nn.Linear(latent_dim, num_hiddens * (data_len // (2 ** 3))),
                Reshape(-1, num_hiddens, (data_len // (2 ** 3))),
                nn.ConvTranspose1d(in_channels=num_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 4, stride=2, padding = kernel_size // 4 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose1d(in_channels=num_hiddens,
                      out_channels=num_hiddens // 2,
                      kernel_size=kernel_size // 2, stride=2, padding = kernel_size // 2 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens // 2),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                      out_channels=1,
                      kernel_size=kernel_size , stride=2, padding = kernel_size  // 2 - 1, bias=False),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )
    def forward(self, x):
        encoder = self.encoder(x)
        freq = self.regresion(encoder)
        decoder = self.decoder(encoder)
        return encoder,None, None, freq, decoder
    
class VAE(nn.Module):
    def __init__(self, num_hiddens, kernel_size, data_len, latent_dim, regression_hidden, dropout):
        super().__init__()
        
        self.encoder = nn.Sequential( #784
                nn.Conv1d(in_channels=1,
                      out_channels=num_hiddens // 2,
                      kernel_size=kernel_size, stride=2, padding = kernel_size  // 2 - 1, bias=False),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(num_hiddens // 2),
                nn.Dropout(dropout),
                nn.Conv1d(in_channels=num_hiddens // 2,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 2, stride=2, padding = kernel_size  // 2 // 2 - 1, bias=False),
                nn.LeakyReLU(0.01),
                nn.Conv1d(in_channels=num_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 4, stride=2, padding = kernel_size  // 4 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens),
                nn.LeakyReLU(0.01),
                nn.Flatten(),
        )
        self.z_mean = torch.nn.Linear(num_hiddens * (data_len // (2 ** 3)), latent_dim)
        self.z_log_var = torch.nn.Linear(num_hiddens * (data_len // (2 ** 3)), latent_dim)
        
        self.regresion = nn.Sequential(
                nn.Linear(latent_dim , regression_hidden),
                nn.Linear(regression_hidden , 6),
                nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
                torch.nn.Linear(latent_dim, num_hiddens * (data_len // (2 ** 3))),
                Reshape(-1, num_hiddens, (data_len // (2 ** 3))),
                nn.ConvTranspose1d(in_channels=num_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=kernel_size // 4, stride=2, padding = kernel_size // 4 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose1d(in_channels=num_hiddens,
                      out_channels=num_hiddens // 2,
                      kernel_size=kernel_size // 2, stride=2, padding = kernel_size // 2 // 2 - 1, bias=False),
                nn.BatchNorm1d(num_hiddens // 2),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                      out_channels=1,
                      kernel_size=kernel_size , stride=2, padding = kernel_size  // 2 - 1, bias=False),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    def latent_sample(self, z_mu, z_log_var):
        # the reparameterization trick
        std = z_log_var.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(z_mu)

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.latent_sample(z_mean, z_log_var)
        freq = self.regresion(encoded)#torch.cat((z_mean, z_log_var), dim = 1))
        decoded = self.decoder(encoded)
        return encoded,z_mean,z_log_var,freq, decoded
