import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import math
from CustomDataSet import PaintingDataset
# from CSV_Creator import build_csv


# Hyper Parameters #
learning_rate = 0.0001
num_epochs = 20
batch_size = 200
num_class = 27
img_size = (224, 224)


# Models #

# BaseLine CNN Architecture
class BLNet(nn.Module):

    def __init__(self):
        super(BLNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.max2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=6272, out_features=228)
        self.fc2 = nn.Linear(in_features=228, out_features=num_class)

        # Batch Normalization 
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# ResNet-18 Architecture
class ResNet(nn.Module):

    def __init__(self, ):
        super(ResNet, self).__init__()
        self.net = models.resnet18(pretrained=False, progress=True)
        self.fc = nn.Linear(in_features=1000, out_features=num_class)

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(self.fc(x), dim=0)) 
        return x


# Set models
model1 = BLNet()                   # BaseLine CNN
model2 = ResNet()                  # ResNet-18 from scratch

# Models name attribute
model1.name = 'Base Line CNN'
model2.name = 'ResNet-18'

# Models dictionary
my_models = [model1, model2]
model = my_models[0]


# Data #

# Create CSV
csvs = ['C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style_small.csv',
        'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style_large.csv',
        'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style_synthetic.csv',
        'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style_control.csv']
data_root = 'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/Custom_Data'

csv_loc = csvs[0]

csv_dict = {csvs[0]: 'Small', csvs[1]: 'Large', csvs[2]: 'Synthetic', csvs[3]: 'Control'}


# Transform
transform = transforms.Compose([transforms.ToPILImage(),
                                # transforms.Resize((448, 448)),
                                transforms.CenterCrop((448, 448)),
                                transforms.RandomCrop(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

# Transform for Synthetic dataset (csvs[2]):                                                     # set transforms --> transform_synth
data_augment =[transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
               transforms.RandomRotation(degrees=30), transforms.RandomGrayscale(p=0.5)]

transform_synth = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop((448, 448)),
                                      transforms.RandomChoice(data_augment), transforms.RandomCrop(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])


# Get data
data = PaintingDataset(csv_file=csv_loc, root_dir=data_root, transforms=transform)

# Divide data to sets
train_set, validation_set, test_set = torch.utils.data.random_split(data,              # split data in 80-10-10 ratio
                                    [int(len(data)*0.8), int(len(data)*0.1)+1, int(len(data)*0.1)+1])

# Data Loader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Loss & Optimization #
# Loss function - Cross Entropy Loss
loss_function = nn.CrossEntropyLoss()
# Optimizer - Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('\nThe device is {}\n'.format(device))
model.to(device)


# calculate accuracy #

def accuracy(loader):
    """Measures top-1 accuracy"""
    correct = 0
    total = 0

    model.eval()    # set model to evaluation mode

    for batch in loader:
        labels, inputs = batch[1].to(device), batch[0].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)   # top-1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return (100 * correct)/total


# Precision, Recall, F1, top-3

def calculate_performances(loader):
    """Measures top-1, top-3, precision, recall, F1
    Args: DataLoader
    Outputs: for current model & DataLoader (Args) test accuracy
            performance array: [top-1, top-3, Precision, Recall, F1]"""
    confusion_matrix = torch.zeros((num_class, num_class))  # init confusion matrix
    stats_matrix = torch.zeros((num_class, 3))
    all_perf = torch.zeros(5)                               # [top-1, top-3, precision, recall, f1] - averages
    correct = 0
    correct3 = 0
    total = 0

    model.eval()    # set model to evaluation mode

    for i, batch in enumerate(loader):
        labels, inputs = batch[1].to(device), batch[0].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)       # top-1
        correct += (predicted == labels).sum().item()
        _, predicted3 = torch.topk(outputs.data, 3)     # top-3
        for j in range(labels.shape[0]):
            if labels[j] in predicted3[j]:
                correct3 += 1
        total += labels.size(0)

        # create confusion matrix
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
      
    # calculate stats per class
    for cls in range(num_class):
        t_positive = confusion_matrix[cls, cls]                                   # True Positive
        stats_matrix[cls, 0] = confusion_matrix[:, cls].sum()                     # True Positive + False Positive
        stats_matrix[cls, 1] = confusion_matrix[cls, :].sum()                     # True Positive + False Negative
        stats_matrix[cls, 0] = t_positive / stats_matrix[cls, 0]                  # Precision
        if math.isnan(stats_matrix[cls, 0]):
            stats_matrix[cls, 0] = 0
        stats_matrix[cls, 1] = t_positive / stats_matrix[cls, 1]                            # Recall
        if math.isnan(stats_matrix[cls, 1]):
            stats_matrix[cls, 1] = 0
        stats_matrix[cls, 2] = 2 * (stats_matrix[cls, 1] * stats_matrix[cls, 2]) / (        # F1
                    stats_matrix[cls, 1] + stats_matrix[cls, 2])
    # total run stats - all_stats = [top-1, top-3, precision, recall, f1]
    all_perf[0] = (100 * correct) / total                  # Top-1
    all_perf[1] = (100 * correct3) / total                 # Top-3
    all_perf[2] = (stats_matrix[:, 0].sum()) / num_class   # Precision
    all_perf[3] = (stats_matrix[:, 1].sum()) / num_class   # Recall
    all_perf[4] = (stats_matrix[:, 2].sum()) / num_class   # F1

    return all_perf


# Train function

def train():
    """Training function that use global variables to train the chosen model and calculate accuracies
    of types: Top-1, Top-3, Precision, Recall and F1
    No Arg**
    Outputs: epoch average loss, training set accuracy, validation set accuracy &
            test accuracy performance array: [top-1, top-3, Precision, Recall, F1]"""
    ep_loss = []
    train_acc = []
    validation_acc = []
    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        running_loss = 0

        model.train()   # set model to training mode

        for i, batch in enumerate(train_loader):
            labels, inputs = batch[1].to(device), batch[0].to(device)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward pass & back propagation
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # accuracy
            _, predicted = torch.max(outputs.data, 1)  # top-1
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        ep_loss.append(running_loss / len(train_loader)) 
        train_acc.append((100 * correct_train) / total_train)
        validation_acc.append(accuracy(validation_loader))
        print('Epoch: %d of %d, Train Acc: %0.3f, Validation Acc: %0.3f, Loss: %0.3f\n'
              % (epoch + 1, num_epochs, train_acc[epoch], validation_acc[epoch], running_loss / len(train_loader)))

    performances = calculate_performances(test_loader)

    return ep_loss, train_acc, validation_acc, performances


## Run ##

print('\nStart Training...\n')

# Train selected model
epoch_loss, train_accuracy, validation_accuracy, accuracy_performance = train()


# Print performance
print('\nTrain Acc: %0.3f\nValidation Acc: %0.3f\nTest Acc: %0.3f\nTop-3 Acc: %0.3f'
      '\nPrecision: %0.3f\nRecall: %0.3f\nF1 Acc: %0.3f\n' % (train_accuracy[num_epochs-1],
                                                              validation_accuracy[num_epochs-1],
                                                              accuracy_performance[0], accuracy_performance[1],
                                                              accuracy_performance[2], accuracy_performance[3],
                                                              accuracy_performance[4]))


with open('performance_net.txt', "w", newline='') as file:
    file.write('\nTrain Acc: %0.3f\n'
               'Validation Acc: %0.3f\n'
               'Test Acc: %0.3f\n'
               'Top-3 Acc: %0.3f\n'
               'Precision: %0.3f\n'
               'Recall: %0.3f\n'
               'F1 Acc: %0.3f\n' % (train_accuracy[num_epochs-1], validation_accuracy[num_epochs-1], accuracy_performance[0],
                                    accuracy_performance[1], accuracy_performance[2],
                                    accuracy_performance[3], accuracy_performance[4]))


# Plot train & validation accuracy

plt.figure()
plt.plot(train_accuracy, 'y')   
plt.plot(validation_accuracy, 'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('{}_Accuracy on {} Dataset'.format(model.name, csv_dict[csv_loc]))
plt.savefig('{}_Accuracy.png'.format(model.name))
plt.show()


print("Thailand")
