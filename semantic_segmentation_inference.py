from urllib import response
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torchvision.models.segmentation import fcn
import glob
from PIL import Image
import time
from torch.utils.data import DataLoader
import rospy
from mdi_msgs.srv import Model, ModelResponse
from sensor_msgs.msg import Image

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0")
print("Device chosen:", torch.cuda.get_device_name(device))

# Parameters
num_classes = 9  # Automatically calculated in the next cell
# Optimiser parameters
learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999

torch.cuda.empty_cache()


# Loading saved model
model = fcn.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes,
                         aux_loss=False, pretrained_backbone=True).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[
                             beta1, beta2], eps=1e-08)

loaded_checkpoint = torch.load(
    "checkpoint_resnet50_28epochs_customdataset_v0.0.3.pth")

model.load_state_dict(loaded_checkpoint["model_state"])
model.eval()
model.to(device=device)
optimiser.load_state_dict(loaded_checkpoint["optimiser_state"])
epoch = loaded_checkpoint["epoch"]
criterion = nn.CrossEntropyLoss()


# Service Inference Callback
# sensor_msgs/Image Message
def semantic_segmentation_inference_CB(image_message):  # Consider nograd
    # Image digestion and resizing
    image = torch.tensor(image_message.data, dtype=torch.uint8)
    image.to(device=device)
    image.resize(3, image_message.height, image_message.width)  # [3, H, W]

    # Inference
    output = model(image)['out']
    print('Image inference completed')

    # Flatten into message and return
    resp = ModelResponse()
    resp.height = image_message.height
    resp.width = image_message.width
    resp.data = output.resize(-1)

    print('Returning the ')
    return resp


# ROS service
rospy.init_node('computer_vision_node')
inference_service = rospy.Service('semantic_segmentation', Model,
                                  semantic_segmentation_inference_CB)

print('Ready to recieve requests')
rospy.spin()
