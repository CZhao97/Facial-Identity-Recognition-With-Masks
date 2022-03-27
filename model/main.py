from os import listdir
from os.path import isfile, join
import os
import re
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

use_cuda = True


def split_data(dataset_path, batch_size=64):
    torch.manual_seed(1)
    np.random.seed(1)

    transform = transforms.Compose([transforms.Resize((224, 224)),  # (224, 224)
                                    # transforms.ColorJitter(), # change image color
                                    transforms.RandomHorizontalFlip(),  # flip images
                                    # transforms.RandomAffine(30), # Random affine transformation of the image keeping center invariant.
                                    # transforms.CenterCrop(224),
                                    # TODO: standardization
                                    transforms.ToTensor()])

    training_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    indices = list(range(len(training_dataset)))

    train_indices = []

    val_indices = []

    # split data with ratio 8:2
    for i in indices:
        if i % 10 > 7:
            val_indices.append(i)
        else:
            train_indices.append(i)

    np.random.shuffle(train_indices)

    np.random.shuffle(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader, val_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                           sampler=train_sampler), \
                               torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


# set data path
# ori_path = '/content/drive/MyDrive/ut/MIE1517_project_dataset/orl/images/'
# group_folder_path = '/content/drive/MyDrive/ut/MIE1517_project_dataset/orl/grouped_images/'
ori_path = 'E:/Codes/1517Project/Dataset/ORL_dataset/Original/'
group_folder_path = 'E:/Codes/1517Project/Dataset/ORL_dataset/grouped_images/'

files = [f for f in listdir(ori_path) if isfile(join(ori_path, f))]

# extract images and labels and store into target folder
for single_file in files:
    group = single_file.split("_")[1].split('.')[0]
    folder_path = group_folder_path + group

    if not os.path.exists(folder_path):
        print("Directory '{}' is created!".format(group))
        os.makedirs(folder_path)

    shutil.copyfile(ori_path + single_file, folder_path + '/' + single_file)

train_loader, val_loader = split_data(group_folder_path, batch_size=1)

print('Training images we have:', len(train_loader))
print('Validation images we have:', len(val_loader))


# define model to adapt Alex Model
class AlexNetModel(nn.Module):

    def __init__(self):
        super(AlexNetModel, self).__init__()
        self.name = 'AlexNetModel'
        self.conv1 = nn.Conv2d(256, 30, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 41)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.name = 'simpleCNN'
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size, stride
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.fc1 = nn.Linear(32 * 33 * 54, 200)
        self.fc2 = nn.Linear(200, 41)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = x.view(-1, 32 * 33 * 54)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


# function to save checkpoint
def save_checkpoint(model, batch_size, lr, epoch):
    model_path = "{}_{}_{}_{}".format(model.name, batch_size, lr, epoch)
    torch.save(model, model_path)
    print('Checkpoint of {} has been stored successfully!', format(model_path))


def get_accuracy(model, data_loader, grey_images_flag=False):
    correct = 0
    total = 0

    # alexnet = torchvision.models.alexnet(pretrained=True)

    for imgs, labels in data_loader:
        #############################################
        # To Enable GPU Usage

        if grey_images_flag:
            grey_images = torchvision.transforms.Grayscale()(imgs)

            imgs = torch.tensor(np.tile(grey_images, [1, 3, 1, 1]))

        imgs = imgs[:, :, 0:140, :]
        # imgs = alexnet.features(imgs)
        # imgs = lbp_image(imgs)

        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def lbp_image(img):
    '''
    local binary pattern images of faces

    img: np.ndarray of the form (samples, rows, cols, channels) or (samples, rows, cols, channels)

    '''
    img = F.pad(img, (1, 1, 1, 1), "constant", 0)
    img = img.numpy()  # tensor to numpy
    spoint = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [-0, 1], [1, -1], [1, 0], [1, 1]])
    neighbor = 8
    minx = np.min(spoint[:, 0])
    maxx = np.max(spoint[:, 0])
    miny = np.min(spoint[:, 1])
    maxy = np.max(spoint[:, 1])
    bsizex = (np.ceil(np.max(maxx, 0)) - np.floor(np.min(minx, 0)) + 1).astype('int32')
    bsizey = (np.ceil(np.max(maxy, 0)) - np.floor(np.min(miny, 0)) + 1).astype('int32')
    originx = (0 - np.floor(np.min(minx, 0))).astype('int32')
    originy = (0 - np.floor(np.min(miny, 0))).astype('int32')

    batch, channel, xsize, ysize = img.shape
    assert xsize > bsizex and ysize > bsizey
    dx = xsize - bsizex
    dy = ysize - bsizey
    result = np.zeros((batch, channel, dx + 1, dy + 1), dtype='float32')
    C = img[:, :, originx:originx + dx + 1, originy:originy + dy + 1]

    for i in range(neighbor):
        x = spoint[i, 0] + originx
        y = spoint[i, 1] + originy
        N = img[:, :, x:x + dx + 1, y:y + dy + 1]
        D = N > C
        v = 2 ** i
        result = np.add(result, v * D)

    result = torch.Tensor(result)  # 0-255
    result = result / 255  # 0-1

    # image = result[0]
    # # place the colour channel at the end, instead of at the beginning
    # img = np.transpose(image, [1, 2, 0])
    # # normalize pixel intensity values to [0, 1]
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()

    return result


def train(model, dataset_path, opt, batch_size=64, learning_rate=0.001, epochs=30, grey_images_flag=False):
    #############################################
    # input: grey_images_flag: boolean if gray image conversion is needed
    #############################################

    train_loader, val_loader = split_data(dataset_path, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = opt(model.parameters(), lr=learning_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    # alexnet = torchvision.models.alexnet(pretrained=True)

    # n = 0  # the number of iterations
    for epoch in range(epochs):
        for imgs, labels in iter(train_loader):

            if grey_images_flag:
                grey_images = torchvision.transforms.Grayscale()(imgs)

                imgs = torch.tensor(np.tile(grey_images, [1, 3, 1, 1]))

            # imgs = lbp_image(imgs)

            # imgs = alexnet.features(imgs)

            imgs = imgs[:, :, 0:140, :]
            image = imgs[0]
            # place the colour channel at the end, instead of at the beginning
            img = np.transpose(image.cpu().numpy(), [1, 2, 0])
            # normalize pixel intensity values to [0, 1]
            plt.axis('off')
            plt.imshow(img)
            plt.show()

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # save the current training information
        iters.append(epoch + 1)
        losses.append(float(loss) / batch_size)
        train_acc.append(get_accuracy(model, train_loader, grey_images_flag=grey_images_flag))
        val_acc.append(get_accuracy(model, val_loader, grey_images_flag=grey_images_flag))
        print('Epoch{}, Train acc: {} | Val acc: {} '
              .format(epoch + 1, "%.5f" % train_acc[-1], "%.5f" % val_acc[-1]))
        # n += 1

        if (epoch + 1) % 25 == 0:
            save_checkpoint(model, batch_size, learning_rate, epoch + 1)

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


# model = AlexNetModel()
model = simpleCNN()

if use_cuda and torch.cuda.is_available():
    model.cuda()
    print('CUDA is available!  Training on GPU ...')
else:
    print('CUDA is not available.  Training on CPU ...')

train(model, group_folder_path, opt=optim.Adam, batch_size=32, learning_rate=0.001, epochs=100,
      grey_images_flag=True)
