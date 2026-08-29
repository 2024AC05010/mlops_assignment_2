import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# download from: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
tRaw_Dir = "data/raw"
tProcessed_Dir = "data/processed"

tTrain_Transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tVal_Test_Transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CatDogDataset(Dataset):
    def __init__(self, pFile_Paths, pLabels, pTransform=None):
        self.tFile_Paths = pFile_Paths
        self.tLabels = pLabels
        self.tTransform = pTransform

    def __len__(self):
        return len(self.tFile_Paths)

    def __getitem__(self, pIdx):
        tImg_Path = self.tFile_Paths[pIdx]
        tLabel = self.tLabels[pIdx]
        tImage = Image.open(tImg_Path).convert("RGB")
        if self.tTransform:
            tImage = self.tTransform(tImage)
        return tImage, tLabel


def collect_image_paths(pBase_Dir):
    tPaths = []
    tLabels = []
    tCat_Dir = os.path.join(pBase_Dir, "Cat")
    tDog_Dir = os.path.join(pBase_Dir, "Dog")

    for tDir, tLabel in [(tCat_Dir, 0), (tDog_Dir, 1)]:
        for tFile_Name in os.listdir(tDir):
            if tFile_Name.lower().endswith(('.jpg', '.jpeg', '.png')):
                tPaths.append(os.path.join(tDir, tFile_Name))
                tLabels.append(tLabel)

    print(f"Cats: {len([l for l in tLabels if l == 0])} | Dogs: {len([l for l in tLabels if l == 1])}")
    return tPaths, tLabels

def split_and_save():
    tPaths, tLabels = collect_image_paths(tRaw_Dir)
    print(f"Total images found: {len(tPaths)}")

    # 80/10/10 split
    tTrain_Paths, tTemp_Paths, tTrain_Labels, tTemp_Labels = train_test_split(
        tPaths, tLabels, test_size=0.2, random_state=42, stratify=tLabels
    )
    tVal_Paths, tTest_Paths, tVal_Labels, tTest_Labels = train_test_split(
        tTemp_Paths, tTemp_Labels, test_size=0.5, random_state=42, stratify=tTemp_Labels
    )

    os.makedirs(tProcessed_Dir, exist_ok=True)

    for tName, tP, tL in [("train", tTrain_Paths, tTrain_Labels),
                           ("val", tVal_Paths, tVal_Labels),
                           ("test", tTest_Paths, tTest_Labels)]:
        tDf = pd.DataFrame({"path": tP, "label": tL})
        tSave_Path = os.path.join(tProcessed_Dir, f"{tName}.csv")
        tDf.to_csv(tSave_Path, index=False)
        print(f"{tName}: {len(tDf)} samples saved to {tSave_Path}")


def get_dataloaders(pBatch_Size=32):
    tTrain_Df = pd.read_csv(os.path.join(tProcessed_Dir, "train.csv"))
    tVal_Df = pd.read_csv(os.path.join(tProcessed_Dir, "val.csv"))
    tTest_Df = pd.read_csv(os.path.join(tProcessed_Dir, "test.csv"))

    tTrain_Dataset = CatDogDataset(
        tTrain_Df["path"].tolist(), tTrain_Df["label"].tolist(), tTrain_Transforms
    )
    tVal_Dataset = CatDogDataset(
        tVal_Df["path"].tolist(), tVal_Df["label"].tolist(), tVal_Test_Transforms
    )
    tTest_Dataset = CatDogDataset(
        tTest_Df["path"].tolist(), tTest_Df["label"].tolist(), tVal_Test_Transforms
    )

    tTrain_Loader = DataLoader(tTrain_Dataset, batch_size=pBatch_Size, shuffle=True)
    tVal_Loader = DataLoader(tVal_Dataset, batch_size=pBatch_Size, shuffle=False)
    tTest_Loader = DataLoader(tTest_Dataset, batch_size=pBatch_Size, shuffle=False)

    return tTrain_Loader, tVal_Loader, tTest_Loader


if __name__ == "__main__":
    split_and_save()