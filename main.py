from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from torch.optim import Adam
from torch.optim import SGD

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc,criterion,scheduler, lr_trend, lambda_l1=0):

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        if(lambda_l1 > 0):
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1*l1

        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # updating LR
        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])
        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()   
    
    train_losses.append(train_loss/len(train_loader.dataset))
    train_acc.append(100*correct/len(train_loader.dataset))

    print(f'\nAverage Training Loss={train_loss/len(train_loader.dataset)}, Accuracy={100*correct/len(train_loader.dataset)}')

def test(model, device, test_loader,test_losses, test_acc,epoch,criterion,target_acc=95,save_file=''):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy_epoch = 100. * correct / len(test_loader.dataset)

    if(accuracy_epoch > target_acc) and save_file == 'X':
        model_name_file = "Model_" + str(epoch) + "_acc_" + str(round(accuracy_epoch,2)) + ".pth"
        path = "/content/drive/MyDrive/" + model_name_file
        torch.save(model.state_dict(), path)
        print(f'Saved Model weights in file:  {model_name_file}')

    test_acc.append(100. * correct / len(test_loader.dataset))
    return accuracy_epoch

def fit_model(model, optimizer, criterion, trainloader, testloader, EPOCHS, device,lambda_l1=0,target_acc=100,scheduler=None):
    wrong_prediction_list = []
    right_prediction_list = []
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    lr_trend = []
    
    for epoch in range(EPOCHS):
        print("EPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train(model, device, trainloader, optimizer, epoch, train_losses, train_acc, criterion,scheduler,lr_trend, lambda_l1)

        eval_test_acc = test(model, device, testloader, test_losses, test_acc, epoch, criterion)
        if(eval_test_acc >= target_acc):
            break

    model.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        match = pred.eq(labels.view_as(pred)).to('cpu').numpy()
        for j, i in enumerate(match):
            if(i == False):
                wrong_prediction_list.append((images[j], pred[j].item(), labels[j].item()))
            else:
                right_prediction_list.append((images[j], pred[j].item(), labels[j].item()))

    print(f'Total Number of incorrectly predicted images by model is {len(wrong_prediction_list)}')
    return model, wrong_prediction_list, right_prediction_list, train_losses, train_acc, test_losses, test_acc

#Unet
def train_unet(model, device, train_loader, optimizer, epoch, train_losses,criterion,scheduler,loss_cr='BCE'):

    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0

    for batch_idx, (data, target, label_one) in enumerate(pbar):
        # get samples
        label_one = label_one.type(torch.FloatTensor)
        data, target, label_one = data.to(device), target.to(device), label_one.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        if loss_cr == 'BCE':
            loss = criterion(y_pred, label_one)
        else:
            loss = criterion(y_pred, target)

        train_loss += loss.item()

        if loss_cr == 'BCE':
            pred = torch.argmax(y_pred, 1)
        else:
            _, pred = torch.max(y_pred, 1)

        correct += torch.mean((pred == target).type(torch.float64))

        # Backpropagation
        loss.backward()
        optimizer.step() 
    
    train_losses.append(train_loss)

    print(f'Training Loss={train_loss} Accuracy={correct}')

def test_unet(model, device, test_loader,test_losses,criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target, label_one in test_loader:
            label_one = label_one.type(torch.FloatTensor)
            data, target, label_one = data.to(device), target.to(device), label_one.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.argmax(output, 1)
            correct += torch.mean((pred == target).type(torch.float64))

    #test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f'Test set: Average loss={test_loss} Accuracy={correct}')
    

def fit_model_unet(model, optimizer, criterion, trainloader, testloader, EPOCHS, device,scheduler=None,loss_cr='BCE'):
    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):
        print("\n EPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train_unet(model, device, trainloader, optimizer, epoch, train_losses, criterion,scheduler,loss_cr=loss_cr)

    return model, train_losses, test_losses

def dice_loss(pred, target):
    smooth = 1e-5
    #pred = F.sigmoid(pred)

    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice   

def train_vae(model, device, train_loader, optimizer, epoch, train_losses,criterion,scheduler):

    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        X_hat, mean, logvar = model(data,target)

        # Calculate loss
        reconstruction_loss = criterion(X_hat, data)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
        loss_tot = reconstruction_loss + KL_divergence

        train_loss += loss_tot.item()

        # Backpropagation
        loss_tot.backward()
        optimizer.step() 
    
    train_losses.append(train_loss/len(train_loader.dataset))

    print(f'\nAverage Training Loss={train_loss/len(train_loader.dataset)}')

def fit_model_vae(model, optimizer, criterion, trainloader, EPOCHS, device,scheduler=None):
    train_losses = []
    
    for epoch in range(EPOCHS):
        print("\nEPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train_vae(model, device, trainloader, optimizer, epoch, train_losses, criterion,scheduler)

    return model, train_losses