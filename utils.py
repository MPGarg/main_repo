from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder

class cifar_ds10(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def tl_ts_mod(transform_train,transform_valid,batch_size=512):
    #for determining LR
    trainset_lr = cifar_ds10(root='./data', train=True, download=True, transform=transform_valid)
    trainloader_lr = torch.utils.data.DataLoader(trainset_lr, batch_size=batch_size, shuffle=True, num_workers=2)
    #for main training
    trainset = cifar_ds10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = cifar_ds10(root='./data', train=False, download=True, transform=transform_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainset,trainloader,testset,testloader,trainset_lr,trainloader_lr
  
def set_albumen_params(mean, std):
    num_holes= 1
    cutout_prob= 0.5
    max_height = 8
    max_width = 8

    transform_train = A.Compose(
        [
        A.PadIfNeeded(min_height=36, min_width=36),
        A.RandomCrop(width=32, height=32,always_apply=False),
        A.HorizontalFlip(p=0.2),
        A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=max_height, max_width=max_width, 
        p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
        min_height=max_height, min_width=max_width, mask_fill_value = None),
        A.Normalize(mean = mean, std = std,p=1.0, max_pixel_value=255, always_apply = True),
        ToTensorV2()
        ])
    
    transform_valid = A.Compose(
        [
        A.Normalize(
                mean=mean,
                std=std,
                p=1.0,
                max_pixel_value=255,
            ),
        ToTensorV2()
        ])
    return transform_train, transform_valid 

def load_data():
    transform = transforms.Compose(
      [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
    return trainloader, trainset  

def show_sample(dataset):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    dataiter = iter(dataset)

    index = 0
    fig = plt.figure(figsize=(20,10))
    for i in range(10):
        images, labels = next(dataiter)
        actual = classes[labels]
        image = images.squeeze().to('cpu').numpy()
        ax = fig.add_subplot(2, 5, index+1)
        index = index + 1
        ax.axis('off')
        ax.set_title(f'\n Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        images, labels = next(dataiter) 
    
def process_dataset(batch_size=512,visualize = ''):
    trl, trs = load_data()
    
    mean = list(np.round(trs.data.mean(axis=(0,1,2))/255., 4))
    std = list(np.round(trs.data.std(axis=(0,1,2))/255.,4))
        
    transform_train, transform_valid = set_albumen_params(mean, std)
    trainset_mod, trainloader_mod, testset_mod, testloader_mod,trainset_lr,trainloader_lr = tl_ts_mod(transform_train,transform_valid,batch_size=batch_size)

    if visualize == 'X':
        show_sample(trs)

    return trainset_mod, trainloader_mod, testset_mod, testloader_mod , mean, std ,trainset_lr,trainloader_lr


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def plot_acc_loss(train_acc,train_losses,test_acc,test_losses):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(train_losses, label='Training Losses')
    axs[0].plot(test_losses, label='Test Losses')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title("Loss")

    axs[1].plot(train_acc, label='Training Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title("Accuracy")

    plt.show()    

def display_incorrect_pred(mismatch, n=20 ):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    display_images = mismatch[:n]
    index = 0
    fig = plt.figure(figsize=(20,20))
    for img in display_images:
        image = img[0].squeeze().to('cpu').numpy()
        pred = classes[img[1]]
        actual = classes[img[2]]
        ax = fig.add_subplot(4, 5, index+1)
        ax.axis('off')
        ax.set_title(f'\n Predicted Label : {pred} \n Actual Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        #ax.imshow(image, cmap='gray_r')
        index = index + 1
    plt.show()

def find_lr(net, optimizer, criterion, train_loader):
    """Find learning rate for using One Cyclic LRFinder
    Args:
        net (instace): torch instace of defined model
        optimizer (instance): optimizer to be used
        criterion (instance): criterion to be used for calculating loss
        train_loader (instance): torch dataloader instace for trainig set
    """
    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot()
    lr_finder.reset()

#GradCam copied from https://github.com/kazuto1011/grad-cam-pytorch

from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
          def forward_hook(module, input, output):
              self.fmap_pool[key] = output.detach()

          return forward_hook

        def save_grads(key):
          def backward_hook(module, grad_in, grad_out):
              self.grad_pool[key] = grad_out[0].detach()

          return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        print(one_hot.shape)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:] # HxW
        self.nll = self.model(image)
        #self.probs = F.softmax(self.logits, dim=1)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size duign forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def generate_gradcam(misclassified_images, model, target_layers,device):
    images=[]
    labels=[]
    for i, (img, pred, correct) in enumerate(misclassified_images):
        images.append(img)
        labels.append(correct)
    
    model.eval()
    
    # map input to device
    images = torch.stack(images).to(device)
    
    # set up grad cam
    gcam = GradCAM(model, target_layers)
    
    # forward pass
    probs, ids = gcam.forward(images)
    
    # outputs agaist which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images),-1).to(device)
    
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
        
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids

def plot_gradcam(gcam_layers, target_layers, class_names, image_size,predicted, misclassified_images,mean,std):

    images=[]
    labels=[]
    for i, (img, pred, correct) in enumerate(misclassified_images):
      images.append(img)
      labels.append(correct)

    c = len(images)+1
    r = len(target_layers)+2
    fig = plt.figure(figsize=(30,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = plt.subplot(r, c, 1)
    ax.text(0.3,-0.5, "INPUT", fontsize=14)
    plt.axis('off')
    for i in range(len(target_layers)):
      target_layer = target_layers[i]
      ax = plt.subplot(r, c, c*(i+1)+1)
      ax.text(0.3,-0.5, target_layer, fontsize=14)
      plt.axis('off')

      for j in range(len(images)):
        image_cpu = images[j].cpu().numpy().astype(dtype=np.float32)
        for k in range(image_cpu.shape[0]):
          image_cpu[k] = image_cpu[k] * std[k] + mean[k]
        image_cpu = np.transpose(image_cpu, (1,2,0))
        img = np.uint8(255*image_cpu)
        if i==0:
          ax = plt.subplot(r, c, j+2)
          ax.text(0, 0.2, f"actual: {class_names[labels[j]]} \npredicted: {class_names[predicted[j][0]]}", fontsize=12)
          plt.axis('off')
          plt.subplot(r, c, c+j+2)
          plt.imshow(img)
          plt.axis('off')
          
        
        heatmap = 1-gcam_layers[i][j].cpu().numpy()[0] # reverse the color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128,128))
        plt.subplot(r, c, (i+2)*c+j+2)
        plt.imshow(superimposed_img, interpolation='bilinear')
        
        plt.axis('off')
    plt.show()
     
#For Unet
from PIL import Image
class Oxford_Pet(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root="./data", split='trainval', target_types='segmentation', transform=None, download=True):
        super().__init__(root=root, split=split, target_types=target_types,download=download, transform=transform)

    def __getitem__(self, index):
        #print(index)
        images, labels = self._images[index], self._segs[index]
        image = Image.open(images).convert("RGB")
        label = Image.open(labels)

        if self.transform is not None:
            image, _= self.transforms(image, label)
            label, _ = self.transforms(label, label)

        return image, label

def tl_ts_mod_unet(batch_size=64,transform_train=None):
    transform = transforms.Compose(
      [transforms.Resize(size=(128,128),interpolation=transforms.InterpolationMode.NEAREST_EXACT),
       transforms.ToTensor()])
    trainset = Oxford_Pet(root='./data', split='trainval', target_types='segmentation', transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = Oxford_Pet(root='./data', split='test', target_types='segmentation', transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainset,trainloader,testset,testloader      

def show_sample_unet(dataset):
    dataiter = iter(dataset)

    index = 0
    fig = plt.figure(figsize=(20,10))
    for i in range(4):
        images, labels = next(dataiter)
        actual = 'Original' 
        image = images 
        ax = fig.add_subplot(2, 4, index+1)
        index = index + 1
        ax.set_title(f'\n Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax = fig.add_subplot(2, 4, index+1)
        index = index + 1
        lbl = 'Ground Truth'
        ax.set_title(f'\n Label : {lbl}',fontsize=10) 
        ax.imshow(np.transpose(labels, (1, 2, 0)))
        images, labels = next(dataiter)

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
        fig = plt.figure(figsize=(20,10))
        for i in range(image_no):
            images, labels = next(dataiter)
            labels = labels.to(torch.float)
            data, target = images.to(device), labels.to(device)
            output = model(data)
            actual = 'Original' 
            image = output.to('cpu').numpy() 
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            ax.set_title(f'\n Label : {actual}',fontsize=10) 
            ax.imshow(np.transpose(images[0], (1, 2, 0))) 
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            lbl = 'Ground Truth'
            ax.set_title(f'\n Label : {lbl}',fontsize=10)
            ax.imshow(np.transpose(labels[0], (1, 2, 0)))
            ax = fig.add_subplot(image_no, 3, index+1)
            index = index + 1
            lbl = 'Predicted'
            ax.set_title(f'\n Label : {lbl}',fontsize=10)
            ax.imshow(np.transpose(image[0], (1, 2, 0)))
            images, labels = next(dataiter)