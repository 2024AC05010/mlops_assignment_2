from torchvision import transforms
from PIL import Image
import torch


tVal_Transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def preprocess_image(pImage):
    tTensor = tVal_Transforms(pImage)
    tBatch = tTensor.unsqueeze(0)
    return tBatch


def apply_transforms(pImage, pIs_Training=False):
    if pIs_Training:
        tTransform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        tTransform = tVal_Transforms

    tResult = tTransform(pImage)
    return tResult