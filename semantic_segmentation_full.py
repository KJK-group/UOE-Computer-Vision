import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torchsummary import summary
import torchinfo
import json
from torchvision.models.segmentation import fcn
import glob
from PIL import Image
import time
from torch.utils.data import DataLoader
import airsim

from torch.utils.tensorboard import SummaryWriter

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0")
print("Device chosen GPU:", torch.cuda.get_device_name(device))

# Parameters
# num_classes = 9 # automatically calculated in the next cell
test_ratio = 20  # 1/test_ratio of samples are for testing

# Hyper parameters
num_epochs = 50
batch_size = 2
learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999
log_directory = f"runs/MDI-CustomDataset/resnet50/v0.0.0 Adam lr = {learning_rate}, epochs = {num_epochs}, batchsize ={batch_size}"
writer = SummaryWriter(log_directory)

torch.cuda.empty_cache()
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


def pil_to_np(img):
    return np.array(img)


# Help with glob from https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array
file_list_images = sorted(glob.glob('MDI-CustomDataset/img_0_0_*.png'))
#file_list_depths  = sorted(glob.glob('MDI-CustomDataset/img_0_1_*.pfm'))
file_list_targets = sorted(glob.glob('MDI-CustomDataset/img_0_5_*.png'))

dataset_images = np.array([np.array(Image.open(filename))
                          for filename in file_list_images])
#dataset_depths  = np.array([np.array(airsim.utils.read_pfm(filename)[0])  for filename in file_list_depths])
dataset_targets = np.array([np.array(Image.open(filename))
                           for filename in file_list_targets])

images = dataset_images[:, :, :, 0:3]
targets_RGB = dataset_targets[:, :, :, 0:3]

with torch.no_grad():

    def unique_rgb_colors_in_img(img):
        return np.unique(img.reshape(-1, 3), axis=0)

    # Going from 3 target channels into 1
    unique_colors = unique_rgb_colors_in_img(targets_RGB[0])
    num_classes = len(unique_colors)
    print('Number of classes/unique RGB values in given image:', num_classes)
    # finds the unique RGB colors and lists them # 1.1GB
    color_map = torch.tensor(unique_colors, device=device)

    # putting it on the GPU and garbage collrcting the cpu one # 1.6GB
    torch_targets_RGB = torch.tensor(targets_RGB, device=device)
    # Create target on the gpu
    targets = torch.zeros(targets_RGB.shape[:3], device=device)  # 14.4GB
    # convert to torch tensor and put it on gpu
    #mid = time.perf_counter()
    #print(f"time gpu moving: {mid - start}")
    for i, c in enumerate(color_map):
        # find all pixel equal to the color
        indices = torch.where(torch.all(torch_targets_RGB == c, dim=-1))
        # set class
        # targets[indices] = i + 1
        targets[indices] = i
    #print(f"time: {time.perf_counter() - mid}")

    del color_map
    torch.cuda.empty_cache()
    # del(targets_RGB)

    #targets = torch.unsqueeze(targets, dim=-1)
    # Permutes from [84, 720, 1280, 3] to [84, 3, 720, 1280], as that's how PyTorch likes it
    images = torch.tensor(images, device=device).permute(0, 3, 1, 2)

    # Pairing the images with the labels - train to test ratio of 5 to 1
    # num_test = int(len(targets)/5) # will return 1/5 integer of the total sample size
    # will return 1/5 integer of the total sample size
    num_test = int(len(targets)/test_ratio)
    # train_data = (images[num_test:len(targets)], targets[num_test:len(targets)])
    # test_data = (images[:num_test], targets[:num_test])
    train_data = [(img, t) for img, t in zip(
        images[num_test:len(targets)], targets[num_test:len(targets)])]
    test_data = [(img, t)
                 for img, t in zip(images[:num_test], targets[:num_test])]

    # Dataloaders used to batch the paired images
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    '''Plot from dataset'''
    # fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(24, 16))
    # ax[0][0].imshow(images[0].to('cpu').permute(1,2,0))
    # ax[0][1].imshow(targets[0].to('cpu'))
    # ax[1][0].imshow(images[83].to('cpu').permute(1,2,0))
    # ax[1][1].imshow(targets[83].to('cpu'))

    '''Plot from dataloader'''
    # plot_img, plot_target = iter(train_loader).next()
    # plot_img = plot_img[0]
    # plot_target = plot_target[0]
    # fig, ax = plt.subplots(ncols=2, figsize=(24, 16))
    # ax[0].imshow(np.array(plot_img.to('cpu')).transpose(1,2,0)) # transpose(1,2,0) changes the order of the dimensions
    # ax[1].imshow(np.array(plot_target.to('cpu')))
    # 18.1GB


'''Loading saved model'''
# model = fcn.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=False, pretrained_backbone=True).to(device)
# optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[beta1, beta2], eps=1e-08)

# loaded_checkpoint = torch.load("UPDATE PATH")

# # for param in model.parameters():    # Freezing the startign layers
# #     # param.requires_grad = False
# #     param.requires_grad = False

# model.load_state_dict(loaded_checkpoint["model_state"])
# optimiser.load_state_dict(loaded_checkpoint["optimiser_state"])
# epoch = loaded_checkpoint["epoch"]
# criterion = nn.CrossEntropyLoss()

'''Loading new model'''
model = fcn.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes,
                         aux_loss=False, pretrained_backbone=True).to(device)

# Finetuning
# for param in model.parameters():    # Freezing/unfreezing the starting layers
#     # param.requires_grad = False
#     param.requires_grad = False

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[
                             beta1, beta2], eps=1e-8)
criterion = nn.CrossEntropyLoss()


'''Training'''
# Tensorboard
#writer.add_graph(model.cpu(), val_dataset[0][0])
# writer.close()

# Doing the training now

n_total_steps = len(train_loader)

steps_until_print = batch_size

# stop_training = False
# def signal_handler(sig, frame):
#     print('\nDetected Ctrl+C, stopping training')
#     stop_training = True
#     print('Saving model')
# signal.signal(signal.SIGINT, signal_handler)

model.train()
print('Starting training')
for epoch in range(num_epochs):
    # if stop_training: break

    # Check for stop - read file for boolean to stopping safely
    with open("train.json") as train_json:
        train_dict = json.load(train_json)
        if train_dict["train"] == "False":
            break

    # Every epoch tests the whole dataset once
    testing_batches = iter(test_loader)

    for i, (images, targets) in enumerate(train_loader):
        # Check for stop - read file for boolean to stopping safely
        with open("train.json") as train_json:
            train_dict = json.load(train_json)
            if train_dict["train"] == "False":
                break

        images = images.to(device, torch.float32)
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
            writer.add_scalar('Accuracy/training',
                              batch_pixel_accuracy, epoch * n_total_steps + i)

            # Logging the train loss
            # label of the scalar, actual loss mean, current global step
            writer.add_scalar('Loss/training', loss.item() /
                              steps_until_print, epoch * n_total_steps + i)

            # For every 5 batches, test one batch. (test:train data ratio is split 1:5)
            if (i+1) % test_ratio == 0:  # Logging the testing loss
                test_images, test_targets = testing_batches.next()

                test_images = test_images.to(device)
                test_targets = test_targets.to(device)  # .squeeze(1)

                model.eval()
                test_outputs = model(test_images.float())['out']
                model.train()

                test_pred = torch.argmax(test_outputs, dim=1)

                '''Plot test results'''
                fig, ax = plt.subplots(ncols=3, figsize=(24, 16))
                # transpose(1,2,0) changes the order of the dimensions
                ax[0].imshow(test_images[0].to(
                    'cpu', torch.uint8).permute(1, 2, 0))
                ax[1].imshow(test_targets[0].to('cpu'))
                ax[2].imshow(test_pred[0].to('cpu').detach())
                plt.pause(0.01)
                '''                     '''

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

print("Training is done")


'''Saving the model'''
# checkpoint = {
#    "epoch": epoch,
#    "model_state": model.state_dict(),
#    "optimiser_state": optimiser.state_dict()
# }
#torch.save(checkpoint, "checkpoint_resnet50___.pth")

'''Plotting'''
# with torch.no_grad():
#     iterator = iter(test_loader)
#     images, targets = next(iterator)
#     images = images.to(device)

#     model.eval().to(device)
#     output = model(images.to(device))['out']
#     pred = torch.argmax(output, dim=1)

#     images = images.to('cpu')
#     targets = targets.to('cpu')
#     output = output.to('cpu')
#     pred = pred.to('cpu')
#     print('image: ', images.shape)
#     print('target:', targets.shape)
#     print('output:', output.shape)
#     print('pred:  ', pred.shape)
#     fig, ax = plt.subplots(ncols=3, figsize=(24, 16))
#     ax[0].imshow(images[1].squeeze().permute(1,2,0))  # .squeeze() does the same thing as .numpy().transpose(1,2,0)
#     ax[1].imshow(targets[1].squeeze()) # .squeeze() does the same thing as .numpy().transpose(1,2,0)
#     ax[2].imshow(pred[1].squeeze())
