import os
import yaml
import torch
from tqdm import tqdm
import sys

# Ajouter le dossier racine du projet si nécessaire
sys.path.insert(1, 'path/to/folder')

from datasets import ImageNetMnist, collate_views
from learning.transformations import DataTransformation

# ----------------------------
# Background dataset
# ----------------------------
with open("configs/contrastive_training_background.yaml", 'r') as file:
    cfg = yaml.safe_load(file)

dataset = ImageNetMnist(
    imagenet_data_folder=cfg['imagenet_data_folder'],
    imagenet_labels_file=cfg['imagenet_labels_file'],
    imagenet_classes=cfg['imagenet_classes'],
    mnist_data_folder=cfg['mnist_data_folder'],
    shared_feature=cfg['shared_feature']  # 'background'
)

transform = DataTransformation(cfg)
dataset.transform1 = transform(['random_cropping', 'resize'])
dataset.transform2 = transform(['gaussian_blur', 'normalize'])

data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collate_views,
    batch_size=cfg.get('batch_size', 16)
)

data, imagenet_labels, digit_labels = [], [], []
for imgs, labels in tqdm(data_loader, desc="Building background dataset"):
    data.append(imgs['view1'])
    imagenet_labels.append(labels['view1']['imagenet_label'])
    digit_labels.append(labels['view1']['digit_label'])

    data.append(imgs['view2'])
    imagenet_labels.append(labels['view2']['imagenet_label'])
    digit_labels.append(labels['view2']['digit_label'])

data = torch.cat(data)
imagenet_labels = torch.cat(imagenet_labels)
digit_labels = torch.cat(digit_labels)

os.makedirs(cfg['res_dir'], exist_ok=True)
torch.save(data, os.path.join(cfg['res_dir'], 'test_data_background.pt'))
torch.save(imagenet_labels, os.path.join(cfg['res_dir'], 'test_imagenet_labels_background.pt'))
torch.save(digit_labels, os.path.join(cfg['res_dir'], 'test_digit_labels_background.pt'))

# ----------------------------
# Digit dataset
# ----------------------------
with open("configs/contrastive_training_digit.yaml", 'r') as file:
    cfg = yaml.safe_load(file)

dataset = ImageNetMnist(
    imagenet_data_folder=cfg['imagenet_data_folder'],
    imagenet_labels_file=cfg['imagenet_labels_file'],
    imagenet_classes=cfg['imagenet_classes'],
    mnist_data_folder=cfg['mnist_data_folder'],
    shared_feature=cfg['shared_feature']  # 'digit'
)

transform = DataTransformation(cfg)
dataset.view_transform = transform()

data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collate_views,
    batch_size=cfg.get('batch_size', 16)
)

data, imagenet_labels, digit_labels = [], [], []
for imgs, labels in tqdm(data_loader, desc="Building digit dataset"):
    data.append(imgs['view1'])
    imagenet_labels.append(labels['view1']['imagenet_label'])
    digit_labels.append(labels['view1']['digit_label'])

    data.append(imgs['view2'])
    imagenet_labels.append(labels['view2']['imagenet_label'])
    digit_labels.append(labels['view2']['digit_label'])

data = torch.cat(data)
imagenet_labels = torch.cat(imagenet_labels)
digit_labels = torch.cat(digit_labels)

os.makedirs(cfg['res_dir'], exist_ok=True)
torch.save(data, os.path.join(cfg['res_dir'], 'test_data_digit.pt'))
torch.save(imagenet_labels, os.path.join(cfg['res_dir'], 'test_imagenet_labels_digit.pt'))
torch.save(digit_labels, os.path.join(cfg['res_dir'], 'test_digit_labels_digit.pt'))

# ----------------------------
# Background + Digit dataset
# ----------------------------
with open("configs/contrastive_training_background_digit.yaml", 'r') as file:
    cfg = yaml.safe_load(file)

dataset = ImageNetMnist(
    imagenet_data_folder=cfg['imagenet_data_folder'],
    imagenet_labels_file=cfg['imagenet_labels_file'],
    imagenet_classes=cfg['imagenet_classes'],
    mnist_data_folder=cfg['mnist_data_folder'],
    shared_feature=['background', 'digit']  # <- passer une liste pour plusieurs features
)

transform = DataTransformation(cfg)
dataset.transform1 = transform(['random_cropping', 'resize'])
dataset.transform2 = transform(['gaussian_blur', 'normalize'])

data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collate_views,
    batch_size=cfg.get('batch_size', 16)
)

all_data, all_imagenet_labels, all_digit_labels = [], [], []
for imgs, labels in tqdm(data_loader, desc="Building background+digit dataset"):
    all_data.append(imgs['view1'])
    all_imagenet_labels.append(labels['view1']['imagenet_label'])
    all_digit_labels.append(labels['view1']['digit_label'])

    all_data.append(imgs['view2'])
    all_imagenet_labels.append(labels['view2']['imagenet_label'])
    all_digit_labels.append(labels['view2']['digit_label'])

all_data = torch.cat(all_data)
all_imagenet_labels = torch.cat(all_imagenet_labels)
all_digit_labels = torch.cat(all_digit_labels)

os.makedirs(cfg['res_dir'], exist_ok=True)
torch.save(all_data, os.path.join(cfg['res_dir'], 'test_data_digit_background.pt'))
torch.save(all_imagenet_labels, os.path.join(cfg['res_dir'], 'test_imagenet_labels_digit_background.pt'))
torch.save(all_digit_labels, os.path.join(cfg['res_dir'], 'test_digit_labels_digit_background.pt'))

print("Fichiers combinés background+digit générés avec succès !")
