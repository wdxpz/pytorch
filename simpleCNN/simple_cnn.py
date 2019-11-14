import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12*6*6, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, *input):
        x = self.pool1(F.relu(self.conv1(input[0])))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x



def train(network, dataset, device, save_path, epoch=4, lr=1e-4, watch_step=100):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    average_loss = 0
    average_correct = 0

    network.train()

    for e in range(epoch):
        for i, data in enumerate(dataset, 1):
            inputs, labels = data[0].to(device), data[1].to(device)

            #zero grad
            optimizer.zero_grad()

            logits = network(inputs)

            _, predicts = torch.max(logits, 1)
            average_correct += (predicts==labels).sum().item()*1.0 / labels.shape[0]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            average_loss += loss
            if i % watch_step == 0:
                print('[%2d, %5d] loss: %.3f, accuracy: %3f' % (e+1, i, average_loss/watch_step, average_correct/watch_step))
                average_loss = 0
                average_correct = 0

    torch.save(network.state_dict(), os.path.join(save_path, 'cifar10.pth'))


def predict(network, inputs):

    with torch.no_grad():
        logits = network(inputs)
        _, predicts = torch.max(logits, 1)

    return predicts


def test(network, testloader):

    network.eval()

    correct = 0
    total =0

    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        predicts = predict(network, inputs)
        total += len(inputs)
        correct += (predicts==labels).sum().item()

    print('accuracy on {} test images: {:.3f}'.format(total, correct*1.0/total))



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
Model_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(Model_DIR):
    os.mkdir(Model_DIR)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)
Batch_Size = 32
Num_Workers = 4
CIFAR_train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
CIFAR_train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=Batch_Size, shuffle=True, num_workers=Num_Workers)
CIFAR_test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, transform=transform, download=True)
CIFAR_test_loader = torch.utils.data.DataLoader(CIFAR_test_set, batch_size=Batch_Size, shuffle=True, num_workers=Num_Workers)

net = CNN().to(device)

if device.type == 'cuda':
    net = nn.DataParallel(net, list(range(2)))

train(net, CIFAR_train_loader, device=device, save_path= Model_DIR, epoch=100, lr=0.5e-4)

# net.load_state_dict(torch.load(os.path.join(Model_DIR, 'cifar10.pth')))
test(net, CIFAR_test_loader)





