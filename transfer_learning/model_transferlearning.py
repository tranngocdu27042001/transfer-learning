from lib import *

#1. Transfer Learning with MobileNet_v2

mobilenet_v2 = models.mobilenet_v2(pretrained=True)

#fine_tuning

mobilenet_v2.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                        nn.Linear(1280,512,bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Linear(512,275))

name_parameter_update = [
'classifier.1.weight',
'classifier.2.weight',
'classifier.2.bias',
'classifier.4.weight',
'classifier.4.bias',
]

for name,value in mobilenet_v2.named_parameters():

    if name in name_parameter_update:
        value.requires_grad = True
        #print(name)
    else:
        value.requires_grad = False

if __name__ == '__main__':
    pass