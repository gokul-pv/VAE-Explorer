import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

MEAN = torch.tensor([0.9573, 0.9379, 0.9213])
STD = torch.tensor([0.1288, 0.1431, 0.1771])


def calculate_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_images = 0
    
    for i in range(len(dataset)):
        image, _ = dataset[i]
        _, height, width = image.shape
        n_pixels = height * width

        # Sum and sum of squares
        mean += image.sum(dim=[1, 2])
        std += (image ** 2).sum(dim=[1, 2])
        n_images += n_pixels

    # Compute the mean and standard deviation
    mean /= n_images
    std = torch.sqrt(std / n_images - mean ** 2)

    return mean, std


def plot_training_history(train_losses, test_losses):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_losses, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.grid(linestyle='-.')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(test_losses, 'b', label='Test Loss')
    plt.title('Test Loss')
    plt.grid(linestyle='-.')
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_image(img):
    img = img * STD[:, None, None] + MEAN[:, None, None]
    image_grid = torchvision.utils.make_grid(img, 10, 5).numpy()

    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='none')
    plt.axis('off')
    plt.show()


def visualise_output(images, model, device, num_samples, is_vae=True):
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        if is_vae:
            output, _, _ = model(images)
        else:
            output = model(images)
        output = output.cpu() * STD[:, None, None] + MEAN[:, None, None]

        image_grid = torchvision.utils.make_grid(output[:num_samples], 10, 5).numpy()
        plt.figure(figsize=(10,10))
        plt.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation="none")
        plt.axis('off')
        plt.show()
