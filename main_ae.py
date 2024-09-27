import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tqdm.notebook import tqdm
from utils import MEAN, STD


def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    train_loss_batch = 0
    pbar = tqdm(train_loader, unit='batch')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.mse_loss(output, data)

        train_loss_batch += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(desc="Epoch = {} batch_loss={:.4f} batch_id={}" \
                             .format(epoch+1, loss.item(), batch_idx))

    train_loss_epoch = train_loss_batch / len(train_loader)
    train_losses.append(train_loss_epoch)

    print("\nTrain set: Average loss: {:.4f}\n".format(train_loss_epoch))


def test(model, device, test_loader, test_losses):
    model.eval()
    test_loss_batch = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_batch += F.mse_loss(output, data).item()

    test_loss_epoch = test_loss_batch / len(test_loader)
    test_losses.append(test_loss_epoch)

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss_epoch))


def train_and_evaluate(model, device, train_loader, test_loader, optimizer, epochs, scheduler=None):

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, train_losses)
        test(model, device, test_loader, test_losses)
        if scheduler is not None:
            scheduler.step()

    return train_losses, test_losses


def reconstruct_images(model, test_loader, device, sample_size=128):
    """
    Reconstructs images from a normal distribution fitted to the latent space of a given autoencoder.
    
    Parameters:
    - model: The trained autoencoder model
    - test_loader: DataLoader containing the test dataset
    - device: Device to run the computations on (e.g., 'cpu' or 'cuda')
    - sample_size: Number of latent vectors to sample from the normal distribution
    
    Returns:
    - img_recon: Reconstructed images tensor
    """
    model.eval()

    with torch.no_grad():
        # Approx. fit a multivariate Normal distribution (with diagonal cov.) to the latent vectors of a random part of the test set
        images, labels = next(iter(test_loader))
        images = images.to(device)
        latent = model.encoder(images)
        latent = latent.cpu()

        mean = latent.mean(dim=0)
        std = (latent - mean).pow(2).mean(dim=0).sqrt()

        # Sample latent vectors from the normal distribution
        latent = torch.randn(sample_size, latent.shape[1]) * std + mean

        # Reconstruct images from the latent vectors
        latent = latent.to(device)
        img_recon = model.decoder(latent)
        img_recon = img_recon.cpu()

    return img_recon


def interpolation(lambda1, model, device, img1, img2):
    model.eval()

    with torch.no_grad():

        # latent vector of first image
        img1 = img1.to(device)
        latent_1 = model.encoder(img1)

        # latent vector of second image
        img2 = img2.to(device)
        latent_2 = model.encoder(img2)

        # interpolation of the two latent vectors
        inter_latent = lambda1 * latent_1 + (1- lambda1) * latent_2

        # reconstruct interpolated image
        inter_image = model.decoder(inter_latent)
        inter_image = inter_image.cpu()
    
    return inter_image


def interpolate_and_display(autoencoder, test_loader, device, num_classes=4, num_images=5, num_lambdas=10):

    # Sort part of test set by digit
    ball = [[] for _ in range(num_classes)]
    for img_batch, label_batch in test_loader:
        for i in range(img_batch.size(0)):
            ball[label_batch[i]].append(img_batch[i:i+1])
        if sum(len(d) for d in ball) >= num_images:
            break
    
    # Interpolation lambdas
    lambda_range = np.linspace(0, 1, num_lambdas)

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    axs = axs.ravel()

    for ind, lam in enumerate(lambda_range):
        inter_image = interpolation(float(lam), autoencoder, device, ball[1][0], ball[2][0])
        inter_image = inter_image * STD[:, None, None] + MEAN[:, None, None]
        inter_image = inter_image.clamp(0, 1)
        image = inter_image.squeeze().numpy()
        
        axs[ind].imshow(np.transpose(image, (1, 2, 0)), interpolation="none")
        axs[ind].set_title('lambda_val=' + str(round(lam, 1)))

    plt.show()