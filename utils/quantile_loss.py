import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        target = target.squeeze(-1) 

        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = preds[:, :, i]  
            error = target - pred_q 
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.unsqueeze(0))  
        return torch.cat(losses, dim=0).mean() 
