import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from utils import deprocess_image

def compute_batch_activations(model, x, layer):
    """TODO: complete.
    """
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    return x.mean(dim=[2,3]).data.cpu()
                else:
                    current_layer += 1

def compute_activations_for_gradient_ascent(model, x, layer, filter_id):
    """TODO: complete.
    """
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    return x.mean(dim=[2,3])
                else:
                    current_layer += 1
                    
def compute_dataset_activations(model, dataset, layer, batch_size=64):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size
    )
    activations = []
    for data, _ in tqdm(loader, desc=f"Compute activations over dataset for layer {layer}"):
        batch_activation = compute_batch_activations(model, data, layer)
        activations.append(batch_activation)
    return torch.cat(activations)

def maximize_img_response(model, img_size, layer, filter_id, device='cuda', n_it=50000, wd=1e-5, lr=3, reg_step=5):
    """TODO: complete.
    A L2 regularization is combined with a Gaussian blur operator applied every reg_step steps.
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for param in model.parameters():
        param.requires_grad_(False)

    img = torch.nn.Parameter(
        data=torch.randn((1, 3, img_size, img_size))
        ).to(device)
    img.retain_grad()
    
    
    model = model.to(device)
    for it in tqdm(range(n_it), desc='Gradient ascent in image space'):

        out = compute_activations_for_gradient_ascent(
            model, img, layer=layer, filter_id=filter_id
            )
        target = torch.tensor([filter_id], dtype=torch.long, device=device)
        loss = F.cross_entropy(out, target) + wd * (img**2).sum()

        # compute gradient
        if img.grad is not None:
            img.grad.zero_()

        loss.backward()

        # normalize gradient
        grads = img.grad.data
        grads = grads.div(torch.norm(grads)+1e-8)

        # Update image
        with torch.no_grad():
            img.data += lr * grads


        # Apply gaussian blur
        if it % reg_step == 0:
            with torch.no_grad():
                blurred = gaussian_filter(
                    img.squeeze().cpu().numpy().transpose(1,2,0),
                    sigma=(0.3,0.3,0)
                )
                blurred = torch.from_numpy(blurred).float().permute(2,0,1).unsqueeze(0).to(device)
                img.data = blurred

    return deprocess_image(img.detach().cpu().numpy())


def maximize_img_response_bis(model, img_size, layer, filter_id, device='cuda', n_it=50000, wd=1e-5, lr=3, reg_step=5):
    """TODO: complete.
    A L2 regularization is combined with a Gaussian blur operator applied every reg_step steps.
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for param in model.parameters():
        param.requires_grad_(False)

    img = torch.nn.Parameter(
        data=torch.randn((1, 3, img_size, img_size))
        ).to(device)
    img.retain_grad()
    
    
    model = model.to(device)
    for it in tqdm(range(n_it), desc='Gradient ascent in image space'):

        out = compute_activations_for_gradient_ascent(
            model, img, layer=layer, filter_id=filter_id
            )
        #target = torch.tensor([filter_id], dtype=torch.long, device=device)
        loss = -out[0, filter_id] + wd * (img**2).sum()

        # compute gradient
        if img.grad is not None:
            img.grad.zero_()

        loss.backward()

        # normalize gradient
        grads = img.grad.data
        grads = grads.div(torch.norm(grads)+1e-8)

        # Update image
        with torch.no_grad():
            img.data += lr * grads


        # Apply gaussian blur
        if it % reg_step == 0:
            with torch.no_grad():
                blurred = gaussian_filter(
                    img.squeeze().cpu().numpy().transpose(1,2,0),
                    sigma=(0.3,0.3,0)
                )
                blurred = torch.from_numpy(blurred).float().permute(2,0,1).unsqueeze(0).to(device)
                img.data = blurred

    return deprocess_image(img.detach().cpu().numpy())