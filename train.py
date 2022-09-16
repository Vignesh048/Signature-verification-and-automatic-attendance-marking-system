import torch
import numpy as np
import pandas as pd
import training.model as model
import training.losses as losses
import training.optimizer as optimizers
from preprocess.preprocess import LoadDataset
from torch.utils.data import DataLoader

train_labels = pd.read_csv("data/train_data.csv", names = ["image_1", "image_2", "label"])
train_labels

processed_train_directory = "data/train_preprocess_vgg16/"
train_data = LoadDataset(train_labels, processed_train_directory)
train_data_loaded = DataLoader(train_data, shuffle = True, batch_size = 50)

# Declare Siamese Network
net = model.Siamese_network().cuda()
# Decalre Loss Function
criterion = losses.ContrastiveLoss()
# Declare Optimizer
optimizer = optimizers.get_Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

#train the model
def train(train_dataloader):
    loss=[] 
    for i, data in enumerate(train_dataloader):
      img0, img1 , label = data
      img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
      optimizer.zero_grad()
      output1,output2 = net(img0,img1)
      loss_contrastive = criterion(output1,output2,label)
      loss_contrastive.backward()
      optimizer.step()
      loss.append(loss_contrastive.item())
      print(loss)
    loss = np.array(loss)
    return loss.mean()/len(train_dataloader)

for epoch in range(10):
  train_loss = train(train_data_loaded)
  print(f"Training loss{train_loss}")

torch.save(net.state_dict(), "data/model_save/seamese.pt")