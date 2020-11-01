import torch
import PIL
import cv2
import numpy as np
import torchvision
import os
import matplotlib.pyplot as plt

fer_train = 'fer_data/fer_train/'
fer_test = 'fer_data/fer_test/'
fer_val = 'fer_data/fer_val/'

#CKP_Train = 'CK_48/train/'
#CKP_Valid = 'CK_48/val/'
#CKP_Test = 'CK_48/test/'

CKP_Train = 'CK_48_Original/train/'
CKP_Valid = 'CK_48_Original/val/'
CKP_Test = 'CK_48_Original/test/'


# Defining our ArtNetDataset with transforms

class FERDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        # Initialize certain variables that will be used later.
        self.directory = directory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.toTensor = torchvision.transforms.ToTensor()

        self.filenames = np.asarray(os.listdir(self.directory))

        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise  ', 'neutral']


        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomRotation(45),
            #torchvision.transforms.RandomCrop(36),
            #torchvision.transforms.Resize(48),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(0.5, 0.5)
        ])

    # Modify methods to return data in the required format.
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        X = PIL.Image.open(self.directory + f)
        Y = np.array([int(f[0])], dtype='long')
        return self.toTensor(X), torch.Tensor(Y).long()


class CKPlusDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        # Initialize certain variables that will be used later.
        self.directory = directory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.toTensor = torchvision.transforms.ToTensor()
        self.filenames = np.asarray(os.listdir(self.directory))
        self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomRotation(10),
            #torchvision.transforms.RandomCrop(36),
            #torchvision.transforms.Resize(48),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(0.5, 0.5)
        ])

    # Modify methods to return data in the required format.
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        X = PIL.Image.open(self.directory + f)
        X = self.transform(X)
        Y = np.array([int(f[0])], dtype='uint8')
        return X, torch.Tensor(Y).long()


def get_dataloader_FER():
    FERTrain = FERDataset(fer_train)
    FERValidation = FERDataset(fer_val)
    FERTest = FERDataset(fer_test)

    training_dl = torch.utils.data.DataLoader(FERTrain, batch_size=128, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(FERValidation, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(FERTest, batch_size=64, shuffle=True)

    return training_dl, valid_dl, test_dl


def get_dataloader_CKPlus():
    CKPTrain = CKPlusDataset(CKP_Train)
    CKPValidation = CKPlusDataset(CKP_Valid)
    CKPTest =  CKPlusDataset(CKP_Test)

    training_dl = torch.utils.data.DataLoader(CKPTrain, batch_size=64, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(CKPValidation, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(CKPTest, batch_size=64, shuffle=True)

    return training_dl, valid_dl, test_dl