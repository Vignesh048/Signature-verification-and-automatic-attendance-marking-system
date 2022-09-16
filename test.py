import os
import torch
from torch import nn
from torchvision.transforms import transforms
import training.model as model
from PIL import Image

net = model.Siamese_network()
net.load_state_dict(torch.load("data/model_save/seamese1.pt"))

net.cuda()
net.eval()
original = sorted(os.listdir("data/set1/"))
check = sorted(os.listdir("data/set2"))
output = []
for i in range(len(original)):

      print(original[i],check[i])
      img0 = "data/set1" + original[i]
      img1 = "data/set2" + check[7]
    #   cv2_imshow(cv2.resize(cv2.imread(img0), (224, 224)))
    #   cv2_imshow(cv2.resize(cv2.imread(img1), (224, 224)))
      tensor = transforms.ToTensor()
      img0 = Image.open(img0)
      img1 = Image.open(img1)
      
      rgbimg = Image.new("RGB", img0.size)
      rgbimg.paste(img0)
      rgbimg1 = Image.new("RGB", img1.size)
      rgbimg1.paste(img1)

      img0 = rgbimg.resize((224,224))
      img1 = rgbimg1.resize((224,224))

      
      img2 = tensor(img0).reshape(1,3,224,224).cuda()
      img3 = tensor(img1).reshape(1,3,224,224).cuda()
      output1,output2 = net(img2,img3)
      pd = nn.PairwiseDistance(p=2)
      print(pd(output1, output2)[0].item()*100)
