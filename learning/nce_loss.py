import torch
import torch.nn.functional as F
import pdb


def nce_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 1.0
        ) -> torch.Tensor:
    """
    PyTorch implementation of the NT-Xent loss introduced in 
    https://proceedings.mlr.press/v119/chen20j/chen20j.pdf

    Args:
        z1: embedding from view 1 (Tensor) of shape (bsz, dim).
        z2: embedding from view 2 (Tensor) of shape (bsz, dim).
        temperature: a floating number for temperature scaling.
    """
    LARGE_NUM = 1e9
    SMALL_NUM = 1e-9

    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  
    sim = torch.mm(z, z.T)
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets])  

    logits = sim / temperature
    loss = F.cross_entropy(logits, targets)
    
    return loss

    