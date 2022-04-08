import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from torchvision.models.segmentation import fcn
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

"""Defining functions"""


def save_model():
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict()
    }
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y-%H%M")
    torch.save(checkpoint, f"checkpoint_resnet50_{date_time}_cityscapes.pth")
    print("Model is saved. Stopping script.")


def new_model_optimiser_criterion_epoch():
    '''Loading new model'''
    model = fcn.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes,
                             aux_loss=False, pretrained_backbone=True).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[
                                 beta1, beta2], eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, 0


def load_saved_model_optimiser_criterion_epoch():
    '''Loading saved model'''
    model = fcn.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes,
                             aux_loss=False, pretrained_backbone=True).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[
                                 beta1, beta2], eps=1e-08)

    loaded_checkpoint = torch.load(
        "checkpoint_resnet50_0.5epoch_cityscapes.pth")

    # for param in model.parameters():    # Freezing the startign layers
    #     # param.requires_grad = False
    #     param.requires_grad = False

    model.load_state_dict(loaded_checkpoint["model_state"])
    optimiser.load_state_dict(loaded_checkpoint["optimiser_state"])
    epoch = loaded_checkpoint["epoch"]
    criterion = nn.CrossEntropyLoss()

    model, optimiser, criterion, epoch


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0")
print("Device chosen GPU:", torch.cuda.get_device_name(device))

# Parameters
num_classes = 35

# Hyper parameters
num_epochs = 12
batch_size = 2
learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999
log_directory = f"runs/Cityscapes/resnet50/v0.1.4 Adam lr = {learning_rate}, epochs = {num_epochs}, batchsize ={batch_size}"
writer = SummaryWriter(log_directory)

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


def pil_to_np(img):
    return np.array(img)


# The Cityscapes dataset is avaliable in PyTorch
train_dataset = torchvision.datasets.Cityscapes(
    root='./cityscapesDataset', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=pil_to_np)
#test_dataset  = torchvision.datasets.Cityscapes(root='./cityscapesDataset', split='test',  mode='fine', target_type='semantic', transform=pil_to_tensor, target_transform=transforms.ToTensor())
val_dataset = torchvision.datasets.Cityscapes(root='./cityscapesDataset', split='val',
                                              mode='fine', target_type='semantic', transform=transform, target_transform=pil_to_np)

# Splitting the training and testing datasets into smaller batches
#workers = 5
# ,  num_workers=workers)#, pin_memory=True))
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)#, num_workers=workers)#, pin_memory=True))
# , num_workers=workers)#, pin_memory=True))
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,  batch_size=batch_size, shuffle=False)

#model, optimiser, criterion, epoch = load_saved_model_optimiser_criterion_epoch()
model, optimiser, criterion, epoch = new_model_optimiser_criterion_epoch()

'''Training'''
# Tensorboard
#writer.add_graph(model.cpu(), val_dataset[0][0])
# writer.close()

# Doing the training now
n_total_steps = len(train_loader)

steps_until_print = batch_size

stop_training = False


def signal_handler(sig, frame):
    print('\nDetected Ctrl+C, stopping training')
    stop_training = True
    print('Saving model')


signal.signal(signal.SIGINT, signal_handler)

model.train()
print('Starting training')
for epoch in range(num_epochs):
    if stop_training:
        break

    # Check for stop - read file for boolean to stopping safely
    with open("train.json") as train_json:
        train_dict = json.load(train_json)
        if train_dict["train"] == "False":
            break

    # Every epoch tests the whole dataset once
    testing_batches = iter(val_loader)

    for i, (images, targets) in enumerate(train_loader):
        # Check for stop - read file for boolean to stopping safely
        with open("train.json") as train_json:
            train_dict = json.load(train_json)
            if train_dict["train"] == "False":
                break

        images = images.to(device)
        targets = targets.to(device)
        # print('images  shape:', images.shape)
        # print('targets shape:', targets.shape)

        # Forward pass
        outputs = model(images)['out']
        # print("outputs shape:", outputs.shape)

        loss = criterion(outputs, targets.long())

        # Backward pass
        optimiser.zero_grad()   # Clear old gradient values
        loss.backward()         # Calculate the gradients
        optimiser.step()        # Update the model's weights - seen at model.parameters()

        with torch.no_grad():

            # Logging the train accuracy
            # Evaluate along the 1st dimension
            pred = torch.argmax(outputs, dim=1)
            batch_pixel_accuracy = (pred == targets).sum(
            ).item()/(batch_size*pred.shape[1]*pred.shape[2])
            # label of the scalar, actual loss mean, current global step
            writer.add_scalar(
                'Accuracy/training', batch_pixel_accuracy, epoch * n_total_steps + i)

            # Logging the train loss
            # label of the scalar, actual loss mean, current global step
            writer.add_scalar('Loss/training', loss.item() /
                              steps_until_print, epoch * n_total_steps + i)

            # For every 5 batches, test one batch. (test:train data ratio is split 1:5)
            if (i+1) % 5 == 0:  # Logging the testing loss
                test_images, test_targets = testing_batches.next()

                test_images = test_images.to(device)
                test_targets = test_targets.to(device).squeeze(1)

                model.eval()
                test_outputs = model(test_images)['out']
                model.train()

                test_pred = torch.argmax(test_outputs, dim=1)

                # print('test_images  shape:', test_images.shape)
                # print('test_targets shape:', test_targets.shape)
                # print('test_outputs shape:', test_outputs.shape)
                # print('test_pred    shape:', test_pred.shape)

                # writer.add_images('test/images',      test_images                  , epoch * n_total_steps + i)
                # writer.add_images('test/targets',     test_targets.unsqueeze(dim=1), epoch * n_total_steps + i)
                # writer.add_images('test/predictions', test_pred.unsqueeze(dim=1)   , epoch * n_total_steps + i)

                # Logging the test accuracy
                test_batch_pixel_accuracy = (test_pred == test_targets).sum(
                ).item()/(batch_size*test_pred.shape[1]*test_pred.shape[2])
                #print('Test batch pixel accuracy', test_batch_pixel_accuracy)
                # label of the scalar, actual loss mean, current global step
                writer.add_scalar(
                    'Accuracy/testing', test_batch_pixel_accuracy, epoch * n_total_steps + i)

                # Logging the test loss
                test_loss = criterion(test_outputs, test_targets.long())
                # label of the scalar, actual loss mean, current global step
                writer.add_scalar('Loss/testing', test_loss.item() /
                                  len(test_targets), epoch * n_total_steps + i)

                #writer.add_scalars("Accuracy", {"train": batch_pixel_accuracy, "test": test_batch_pixel_accuracy}, epoch * n_total_steps + i)

                print(
                    f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')

    #print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')

print("Training is done, saving model...")
