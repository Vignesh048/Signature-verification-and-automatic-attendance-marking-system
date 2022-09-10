# import argparse
# import logging
# from os.path import dirname, abspath, join, isfile

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

import training.model as model
import training.losses as losses
import training.optimizer as optimizers

# Declare Siamese Network
net = model.Siamese_network().cuda()
# Decalre Loss Function
criterion = losses.ContrastiveLoss()
# Declare Optimizer
optimizer = optimizers.get_Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)