import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import flask
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 256, 3)
        # self.conv3 = nn.Conv2d(128, 256, 2)
        self.pool = nn.MaxPool2d(2)
        self.adPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, len(classes))
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        # x = F.relu(self.pool(self.conv3(x)))
        x = torch.squeeze(self.adPool(x))
        return self.fc1(x)
#
# def processImage():
#
#     # Using tutorial from: https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1
#     # Using tutorial from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#     # Using tutorial from: https://www.youtube.com/watch?v=Vt5xsiJPDWM
#     # Using tutorial from: https://towardsdatascience.com/animal-classification-using-pytorch-and-convolutional-neural-networks-78f2c97ca160
#
#     transform = transforms.Compose([
#         # transforms.RandomRotation(20),
#         transforms.RandomResizedCrop(224),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()])
#
#     # load and iterate training dataset
#     data_dir = 'CatVsTiger/'
#     trainSet = torchvision.datasets.ImageFolder(data_dir, transform=transform)
#
#
#     train = DataLoader(trainSet, batch_size=32, shuffle=True)
#
#
classes = ('Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate')  # Defining the classes we have
#     #classes = ('Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel')
#     classes = ('Cat', 'Lion')
#     dataiter = iter(train)
#     images, labels = dataiter.next()
#     fig, axes = plt.subplots(figsize=(15, 4), ncols=5)
#     for i in range(5):
#         ax = axes[i]
#         ax.imshow(images[i].permute(1, 2, 0))
#         ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
#     plt.show()
#
#     net = Net()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=3e-4, momentum=0.9)
#
#     losses = []
#     for epoch in range(2):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in enumerate(train, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             losses.append(loss)
#             running_loss += loss.item()
#             if i % 100 == 99:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.10f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#     plt.plot(losses, label='Training loss')
#     plt.show()
#     print('Finished Training')
#
#     PATH = './catOrLion.pth'
#     torch.save(net.state_dict(), PATH)
#
#     # Loading the trained network
#     net.load_state_dict(torch.load(PATH))
#
#     # load and iterate training dataset
#     data_dir = 'test/'
#     testSet = torchvision.datasets.ImageFolder(data_dir, transform=transform)
#
#     test = DataLoader(testSet, batch_size=32, shuffle=True)
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             print(predicted)
#     print(classes[predicted[0]])
#
#     print('Accuracy of the network on the %d test images: %d %%' % (len(test),
#                                                                     100 * correct / total))
#
#
#
# # make sure that the file is being called directly
# if __name__ == '__main__':
#     processImage()
#    # train_imshow()
#

transform = transforms.Compose([
    # transforms.RandomRotation(20),
    transforms.RandomResizedCrop(128),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# # load and iterate training dataset
# data_dir = 'images/'
# trainSet = torchvision.datasets.ImageFolder(data_dir, transform=transform)
# n = len(trainSet)  # total number of examples
# n_test = int(0.1 * n)  # take ~10% for test
# test_set = torch.utils.data.Subset(trainSet, range(n_test))  # take first 10%
# train_set = torch.utils.data.Subset(trainSet, range(n_test, n))  # take the rest
# classes = ('Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate')  # Defining the classes we have
#
# train_dataLoader = DataLoader(train_set, batch_size = 32)
# test_dataLoader = DataLoader(test_set, batch_size = 32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
PATH = './cat_breeds4_b.pth'

# Loading the trained network
model.load_state_dict(torch.load(PATH))

imsize = 224
# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

image = image_loader("test/download.jpg")

pred = model(image)
predIdx = torch.argmax(pred)
print(predIdx)
print(classes[predIdx])