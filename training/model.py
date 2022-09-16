from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Siamese_network(nn.Module):

  def __init__(self):

      super(Siamese_network, self).__init__()
      model = models.vgg16(pretrained = True)

      self.features = model.features

      for param in self.features.parameters():

        param.requires_grad = False

      self.extract = nn.Sequential(
          
          nn.Dropout(),
          nn.Linear(25088, 1024),
          nn.ReLU(inplace = True),
          nn.Dropout2d(p=0.5),
          nn.Linear(1024, 128),
          nn.ReLU(inplace = True),
          nn.Linear(128,2)
      )

  def network_once(self,x):

    output = self.features(x)
    output = output.view(output.size()[0], -1)
    output = self.extract(output)
    return output

  
  def forward(self, input1, input2):

    output1 = self.network_once(input1)

    output2 = self.network_once(input2)

    return output1, output2

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive