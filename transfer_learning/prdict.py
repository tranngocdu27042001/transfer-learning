import matplotlib.pyplot as plt
import torch

from lib import *

from model_transferlearning import mobilenet_v2
from labels import  labelss
from transform import transfrom

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def predict(image_path, model=mobilenet_v2, labels_list=labelss):
    model.eval()
    image = Image.open(image_path)
    transform_ = transfrom()
    image_transform = transform_(image, 'test').unsqueeze_(0).to(device)
    out = model(image_transform)
    label_idx = torch.argmax(out).item()
    label = labels_list[label_idx]
    print(label)
    plt.title(label)
    plt.imshow(image)
    plt.show()



if __name__ == '__main__':

    weights = torch.load('weight/weight (1).pth',map_location={'cuda:0':'cpu'})
    mobilenet_v2.load_state_dict(weights)
    predict('./image_test/african.jpg')