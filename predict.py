import torch
# from utils.helper import *
import  warnings
from PIL import Image
from torchvision import transforms
from models.ShuffleV2 import *


def image_transform(imagepath):
    test_transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                  std= [0.229, 0.224, 0.225])        
        ])
    
    image = Image.open(imagepath)
    imagetensor = test_transform(image)
    
    return imagetensor


def predict(imagepath, verbose = False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = "D:/Deep Learning Practice/DogCat_Classifier_Webapp/checkpoint/epoch100.pth"
    
    modelnet = load_model(model_path)
    
    # print(modelnet)

    if verbose and modelnet is not None:
        print("Model Loading...")

    image = image_transform(imagepath)
    image1 = image[None, : , : , : ]   #[1, channel, width, height]

    ps = torch.exp(modelnet(image1))

    topconf, topclass = ps.topk(1, dim= 1)

    if topclass.item() == 1:
        return {"Class" : "Dog", "Confidence" : str(topconf.item())}

    else:
        return {'Class':'Cat','Confidence':str(topconf.item())}
        

# print(predict('static/24.jpg'))
# print(predict('static/68.jpg'))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        