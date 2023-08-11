import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class Criterion(nn.Module):
    def __init__(self, cfg):
        super(Criterion, self).__init__()
    
    def kl_divergence(self, mu, logvar):
        loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return loss
    
    def reconstruction_loss(self, input, target, b=None, eps=1e-06):
        if b is not None:
            b_clamped = torch.clamp(b, min=eps)
            loss = torch.mean(torch.mean(F.l1_loss(input, target, reduction='none') / b_clamped + torch.log(b_clamped), dim=(2,3)), dim=(0,1))
        else:
            loss = F.l1_loss(input, target)
    
        return loss
        
        
    
    def forward(self, depth, mu, logvar, target, b=None, image=None):
        kl_div = self.kl_divergence(mu, logvar)
        
        reconstruction_loss = self.reconstruction_loss(depth, target, b)
        edge_smoothness_loss = self.edge_aware_smoothness(depth, image) if image is not None else torch.tensor(0.0).to(mu.device)
        uncertainty_smoothness_loss = self.uncertainty_aware_smoothness(depth, b) if b is not None else torch.tensor(0.0).to(mu.device)
        


        loss_dict = dict(
            kl_div=kl_div,
            recon_loss=reconstruction_loss
        )

        return loss_dict