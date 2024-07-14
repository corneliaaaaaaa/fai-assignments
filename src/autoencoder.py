import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import ipdb
import numpy as np
import random

"""
Implementation of Autoencoder
"""


class Autoencoder(nn.Module):
    """
    I referred to the following resources to implement the code.
    - Autoencoder in Pytorch: https://www.kaggle.com/code/weka511/autoencoder-implementation-in-pytorch
    - Vivian's class slides
    - ChatGPT
    """

    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        # original version
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
        # deeper network
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim),
        #     nn.Linear(encoding_dim, encoding_dim // 2),
        #     nn.Linear(encoding_dim // 2, encoding_dim // 4),
        #     nn.Linear(encoding_dim // 4, encoding_dim // 8),
        #     nn.ReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim // 8, encoding_dim // 4),
        #     nn.Linear(encoding_dim // 4, encoding_dim // 2),
        #     nn.Linear(encoding_dim // 2, encoding_dim),
        #     nn.Linear(encoding_dim, input_dim),
        # )
        # different activation function
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim),
        #     nn.Linear(encoding_dim, encoding_dim // 2),
        #     nn.Sigmoid(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim // 2, encoding_dim),
        #     nn.Linear(encoding_dim, input_dim),
        # )

    def forward(self, x):
        # TODO: 5%
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def fit(self, X, epochs=10, batch_size=32):
        """
        Given epochs and batch size, fit the model using MSE as loss function
        and Adam as optimizer.
        """
        # TODO: 5%
        set_seed(0)  # set seed for reproductibility

        # convert input data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # create data loader for batching
        dataset = TensorDataset(X_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # fit the model (training)
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs in data_loader:
                optimizer.zero_grad()  # reset all gradients for each batch, or else it'll accumulate
                outputs = self(
                    inputs[0]
                )  # put batch of input in the model and get the output (encode then decode)
                loss = loss_function(outputs, inputs[0])  # measure averaged mse
                loss.backward()  # compute gradients using backward propagation
                optimizer.step()  # update parameters using gradients computed previously
                running_loss += loss.item()  # accumulate loss for current epoch
            averaged_loss = running_loss / len(data_loader)
            losses.append(averaged_loss)
            print(f"Epoch {epoch+1}/{epochs}, Averaged squared error: {averaged_loss}")

        # plot_ase(losses, "autoencoder")

    def transform(self, X):
        """
        Obtain the encoded (compressed) representation of input data.

        We use detach to ceate a new tensor that shares the same data as the
        original tensor but without tracking any of the operations applied
        to it. This can help us save memory and improve computational efficiency.
        """
        # TODO: 2%
        X_tensor = torch.tensor(X, dtype=torch.float32)
        encoded = self.encoder(X_tensor)

        return encoded.detach().numpy()

    def reconstruct(self, X):
        # TODO: 2%

        X_tensor = torch.tensor(X, dtype=torch.float32)
        decoded = self(X_tensor)  # evokes the forward method (encode then decode)

        return decoded.detach().numpy()


def set_seed(seed):
    """
    Set seeds for random number generators in Python, NumPy, PyTorch etc.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU users
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_ase(losses, mode):
    """
    Plot averaged squared error
    """
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.ylabel("Averaged Squared Error")
    plt.xlabel("Epoch")
    plt.title(f"Averaged Squared Error vs. Epoch ({mode})")
    plt.savefig(f"autoencoder_ase_{mode}.png")
    # plt.show()


"""
Implementation of DenoisingAutoencoder
"""


class DenoisingAutoencoder(Autoencoder):
    """
    I referred to the following resources to implement the code.
    - Denoising Autoencoder in Pytorch: https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
    - Vivian's class slides
    - ChatGPT
    """

    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim, encoding_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        """
        Add Gaussian noise ϵ to each x. Each component of ϵ comes from an
        independent Gaussian distribution of standard deviation noise factor.

        We use torch.randn_like(x) to generate a tensor of random numbers with
        the same shape as x. So each component of the noise is independently
        sampled from a Gaussian distribution.
        """
        # TODO: 3%
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        noisy_x = torch.clip(noisy_x, 0.0, 1.0)  # constrain values within a range

        return noisy_x

    def fit(self, X, epochs=10, batch_size=32):
        """
        Similar to the fit method of Autoencoder(). Given epochs and batch size, fit
        the model using MSE as loss function and Adam as optimizer. But we add noise
        to x before training the model.
        """
        # TODO: 5%
        set_seed(0)  # set seed for reproductibility

        # convert input data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # create data loader for batching
        dataset = TensorDataset(X_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # optimizer = optim.SGD(self.parameters(), lr=0.001)

        # fit the model (training)
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs in data_loader:
                optimizer.zero_grad()  # reset all gradients for each batch, or else it'll accumulate
                inputs[0]
                outputs = self(
                    inputs[0]
                )  # put batch of input in the model and get the output (encode then decode)
                loss = loss_function(outputs, inputs[0])  # measure averaged mse
                loss.backward()  # compute gradients using backward propagation
                optimizer.step()  # update parameters using gradients computed previously
                running_loss += loss.item()  # accumulate loss for current epoch
            averaged_loss = running_loss / len(data_loader)
            losses.append(averaged_loss)
            print(f"Epoch {epoch+1}/{epochs}, Averaged squared error: {averaged_loss}")

        # plot_ase(losses, "denoising autoencoder")
