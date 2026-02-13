import os
import yaml
import argparse
import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms.v2 as T

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import ImageNetMnist, collate_views
from learning.transformations import DataTransformation
from learning.nce_loss import nce_loss
from utils import deprocess_image

from models.alexnet import alexnet
from models.projection_head import ProjectionHead


def main(cfg):
    os.makedirs(os.path.dirname(cfg['res_dir']), exist_ok=True)

    with open(os.path.join(cfg['res_dir'], 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    dataset = ImageNetMnist(
        imagenet_data_folder=cfg['imagenet_data_folder'], 
        imagenet_labels_file=cfg['imagenet_labels_file'], 
        imagenet_classes=cfg['imagenet_classes'], 
        mnist_data_folder=cfg['mnist_data_folder'],
        shared_feature=cfg['shared_feature'])
    
    transform = DataTransformation(cfg)
    if cfg['shared_feature'] == 'background' or 'background' in cfg['shared_feature']:
        dataset.transform1 = transform(['random_cropping', 'resize'])
        dataset.transform2 = transform(['gaussian_blur', 'normalize'])
    elif cfg['shared_feature'] == 'digit':
        dataset.transform1 = transform(['center_cropping'])
        dataset.transform2 = transform(['normalize'])
    else:
        raise ValueError("Shared feature must be background or digit")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_views
    )

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = alexnet(out=cfg['d_model'], sobel=True, freeze_features=False)
    pretrained_params = torch.load(cfg['pretrained_params'], map_location=torch.device(device))
    pretrained_params['top_layer.weight'] = torch.randn(
        (cfg['d_model'], pretrained_params['top_layer.weight'].shape[1])
        ).to(device)
    pretrained_params['top_layer.bias'] = torch.randn(cfg['d_model']).to(device)
    model.load_state_dict(pretrained_params)
    model = model.to(device)

    projection_head = ProjectionHead(d_in=cfg['d_model'], d_model=cfg['d_model']).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(projection_head.parameters()), 
        lr=float(cfg['lr'])
        )

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    
    metrics = {
            'train': {
                'loss': [],
            }
        }

    best_train_loss = np.inf
    model.train()
    for epoch in range(cfg['epochs']):
        train_loss = 0
        model.train()
        for imgs, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            img1 = imgs['view1'].to(device)
            img2 = imgs['view2'].to(device)

            h1, h2 = model(img1), model(img2)
            z1, z2 = projection_head(h1), projection_head(h2)

            loss = nce_loss(z1, z2)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(data_loader)

        print("Train metrics - Loss: {:.4f}".format(train_loss))
        metrics['train']['loss'].append(train_loss)

        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            torch.save({'epoch': epoch + 1, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(cfg['res_dir'], 'best_model.pth.tar'))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')
   
    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)