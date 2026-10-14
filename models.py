import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, z_dim = 32):
        super().__init__()
        # 4 layer, kernel-4, stride 2, 32 -> 256, choice taken from world models paper
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size = 4, stride = 2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2)
        self.z_dim = z_dim
        self.fc_mu = nn.Linear(1024, self.z_dim) #flattened x
        self.fc_logvar = nn.Linear(1024, self.z_dim)
        self.dec_fc = nn.Linear(z_dim, 1024)
        self.dec_conv1 = nn.ConvTranspose2d(1024, 128, kernel_size = 5, stride =2) # number of channels after reshape
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 5, stride =2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size = 6, stride =2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size = 6, stride =2)

    def encode(self, x):
        x = torch.relu(self.enc_conv1(x))
        x = torch.relu(self.enc_conv2(x))
        x = torch.relu(self.enc_conv3(x))
        x = torch.relu(self.enc_conv4(x))
        x_flat = torch.flatten(x, start_dim = 1) # keeps batch dimension as is
        z_mu = self.fc_mu(x_flat)
        z_logvar = self.fc_logvar(x_flat)

        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        #cant sample directly, because gradients cant flow through a random draw
        sigma = torch.sqrt(torch.exp(z_logvar)) #standard deviation
        eps = torch.randn_like(sigma) #random noise, from distribution of same shape as sigma
        z = z_mu + sigma * eps
        return z

    def decode(self, z):
        x = torch.relu(self.dec_fc(z))
        x = x.view(-1, 1024, 1, 1)
        x = torch.relu(self.dec_conv1(x))
        x = torch.relu(self.dec_conv2(x))
        x = torch.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x



    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        recon = self.decode(z)
        return recon, z_mu, z_logvar

        
#loss function
def vae_loss(recon, x, z_mu, z_logvar, beta = 1.0, kl_tolerance = None):
    "when x is the input to the vae"
    #recon loss
    batch_loss = torch.nn.functional.mse_loss(recon, x, reduction = 'sum')
    recon_loss = batch_loss / x.shape[0]
    kl_loss = ((-0.5 * ( 1 + z_logvar - z_mu**2 - torch.exp(z_logvar))).sum(dim = 1)).mean()
    if kl_tolerance is None:
        total_loss = recon_loss + beta * kl_loss
    else:
        clamped_kl = torch.clamp(kl_loss, min=kl_tolerance)
        total_loss = recon_loss + beta * clamped_kl
    return total_loss, recon_loss, kl_loss


