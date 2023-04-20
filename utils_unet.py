from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torch_lr_finder import LRFinder

#For Unet
from PIL import Image
class Oxford_Pet(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root="./data", split='trainval', target_types='segmentation', transform1=None, transform2=None, download=True):
        super().__init__(root=root, split=split, target_types=target_types,download=download, transform=transform1)
        self.transform1 = transform1
        self.transform2 = transform2  

    def __getitem__(self, index):
        images, labels = self._images[index], self._segs[index]
        image = Image.open(images).convert("RGB")
        label = Image.open(labels)

        if self.transform is not None:
            image = self.transform1(image)
            label = self.transform2(label)
            label_one = torch.nn.functional.one_hot(label, 3).transpose(0, 2).squeeze(-1).transpose(1, 2).squeeze(-1)
        return image, label, label_one    

def tl_ts_mod_unet(batch_size=32):
    train_transform = transforms.Compose([transforms.Resize((128, 128),interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Compose([transforms.PILToTensor(),                                        
                                        transforms.Resize((128, 128),interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                                        transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor))])
    trainset = Oxford_Pet(root='./data', split='trainval', target_types='segmentation', transform1=train_transform, transform2=target_transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = Oxford_Pet(root='./data', split='test', target_types='segmentation', transform1=train_transform, transform2=target_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainset,trainloader,testset,testloader      

def show_sample_unet(dataset):
    dataiter = iter(dataset)

    index = 0
    fig = plt.figure(figsize=(20,10))
    for i in range(4):
        images, labels, label_one = next(dataiter)
        actual = 'Original' 
        image = images 
        ax = fig.add_subplot(2, 4, index+1)
        index = index + 1
        ax.set_title(f'\n Label : {actual}',fontsize=6) 
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax = fig.add_subplot(2, 4, index+1)
        index = index + 1
        lbl = 'Ground Truth'
        ax.set_title(f'\n Label : {lbl}',fontsize=6) 
        ax.imshow(labels)
        images, labels, label_one = next(dataiter)

def plot_acc_loss_unet(train_losses):
    fig, axs = plt.subplots(1,1,figsize=(10,5))

    axs.plot(train_losses, label='Training Losses')
    axs.legend(loc='upper right')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.set_title("Loss")

    plt.show()   

def show_sample_output_unet(model,loader,device,image_no=2):
    dataiter = iter(loader)

    with torch.no_grad():
        index = 0
        fig = plt.figure(figsize=(10,10))
        for i in range(image_no):
            images, labels, label_one = next(dataiter)
            labels = labels.to(torch.float)
            data, target = images.to(device), labels.to(device)
            output = model(data).squeeze()
            predicted_masks = torch.argmax(output, 1)
            predicted_masks = predicted_masks.cpu().numpy()
            actual = 'Original' 
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            ax.set_title(f'\n Label : {actual}',fontsize=6) 
            ax.imshow(np.transpose(images[0], (1, 2, 0))) 
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            lbl = 'Ground Truth'
            ax.set_title(f'\n Label : {lbl}',fontsize=6)
            ax.imshow(labels[0])
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            lbl = 'Predicted'
            ax.set_title(f'\n Label : {lbl}',fontsize=6)
            ax.imshow(predicted_masks[0])
            images, labels, label_one = next(dataiter)