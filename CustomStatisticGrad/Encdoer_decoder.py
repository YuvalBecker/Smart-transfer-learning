import numpy as np
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from tqdm import tqdm

from datasets.data_utils import cifar_part, kmnist_part, mnist_part, Fmnist_part

import matplotlib.pyplot as plt
# Creating a DeepAutoencoder class
class DeepAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64 * 64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64 * 64),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Instantiating the model and hyperparameters

if __name__ == '__main__':
    model = DeepAutoencoder().cuda()
    model.load_state_dict(torch.load(r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\_encoder_decoder99'), strict=True)
    criterion = torch.nn.MSELoss()
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    transform= transforms.Compose([transforms.ToTensor(),transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])

    dataset_pre_kmnist = datasets.KMNIST(root='.', train=True, download=True,transform=transform)
    dataset_pre_fmnist = datasets.FashionMNIST(root='.', train=True, download=True,transform=transform)

    dataset_pre_kmnist.data = torch.cat([dataset_pre_kmnist.data, dataset_pre_fmnist.data], dim = 0  )
    dataset_pre_kmnist.targets = torch.cat([dataset_pre_kmnist.targets, dataset_pre_fmnist.targets], dim = 0  )

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset_pre_kmnist, batch_size=batch_size,
                                                     shuffle=True)


    # List that will store the training loss
    train_loss = []

    # Dictionary that will store the
    # different images and outputs for
    # various epochs
    outputs = {}

    batch_size = len(train_loader)

    # Training loop starts
    for epoch in range(num_epochs):

        # Initializing variable for storing
        # loss
        running_loss = 0

        # Iterating over the training dataset
        for batch in tqdm(train_loader):

            # Loading image(s) and
            # reshaping it into a 1-d vector
            img, _ = batch
            img = img.reshape(-1, 64*64)

            # Generating output
            out = model(img.cuda())

            # Calculating loss
            loss = criterion(out, img.cuda())

            # Updating weights according
            # to the calculated loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Incrementing loss
            running_loss += loss.item()

        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(running_loss)

        # Storing useful images and
        # reconstructed outputs for the last batch
        outputs[epoch+1] = {'img': img, 'out': out}
        torch.save(model.state_dict(),r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\_encoder_decoder'+str(epoch))
        print(train_loss)

    # Plotting the training loss
    plt.plot(range(1,num_epochs+1),train_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.show()