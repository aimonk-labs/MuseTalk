import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
import torchvision.transforms as transforms
MuseTalk_submodal_dir = os.path.dirname(os.path.abspath(__file__))
MuseTalk_submodal_parent_dir= os.path.dirname(os.path.dirname(MuseTalk_submodal_dir))

class FaceParsing():
    def __init__(self):
        self.net = self.model_init()
        self.preprocess = self.image_preprocess()

    def model_init(self, 
                   resnet_path=os.path.join(MuseTalk_submodal_parent_dir,'models/face-parse-bisent/resnet18-5c106cde.pth'), 
                   model_pth=os.path.join(MuseTalk_submodal_parent_dir,'models/face-parse-bisent/79999_iter.pth')):
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512)):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            # import pdb;pdb.set_trace()
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing[np.where(parsing>13)] = 0
            parsing[np.where(parsing>=1)] = 255
        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
    
