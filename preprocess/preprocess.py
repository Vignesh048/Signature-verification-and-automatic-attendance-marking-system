from PIL import Image
import torch
import os
from torchvision.transforms import transforms


def preprocess(img_path):
  
  img = Image.open(img_path)
  imgR = img.resize((224,224))

  return imgR

class LoadDataset:

  def __init__(self, train, train_directory):

      self.train_label = train
      self.train_dir = train_directory

  def __getitem__(self, index):

      # images locations

      img1_path = os.path.join(self.train_dir,self.train_label.iat[index,0])
      img2_path = os.path.join(self.train_dir,self.train_label.iat[index,1])

      # import images as tensors

      tensor = transforms.ToTensor()

      img1 = tensor(Image.open(img1_path))
      img2 = tensor(Image.open(img2_path))

      # label

      label = torch.tensor([float(self.train_label.iat[index,2])])


      return (img1,img2,label)

  def __len__(self):
        return len(self.train_label)