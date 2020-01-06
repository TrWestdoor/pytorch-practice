import matplotlib.pyplot as plt
import torch
import torchvision
import os
import torch.utils.data as Data
import torch.nn as nn


BATCH_SIZE = 50
EPOCH = 1
LR = 0.01

DOWNLOAD_CIFAR10 = False
if not(os.path.exists('./CIFAR10/')) or not os.listdir('./CIFAR10/'):
    DOWNLOAD_CIFAR10 = True


train_data = torchvision.datasets.CIFAR10(
    root='./CIFAR10',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_CIFAR10
)

print(train_data.data.shape)
print(train_data.targets.__len__())
# plt.imshow(train_data.data[0])
# plt.show()


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False)

test_x = torch.unsqueeze(torch.from_numpy(test_loader.data), dim=1).type(torch.FloatTensor)
test_y = test_loader.targets


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = cnn(test_x)
            pred_y = torch.max(test_out, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

