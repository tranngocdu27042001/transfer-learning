from lib import *


class transfrom():
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transfrom = {
            'train': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop((resize, resize)),
                transforms.RandomResizedCrop((resize, resize)),
                # transforms.RandomRotation((-20,20)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'valid': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'test': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        }

    def __call__(self, image, phase='train'):
        return self.transfrom[phase](image)

if __name__ == '__main__':
    pass