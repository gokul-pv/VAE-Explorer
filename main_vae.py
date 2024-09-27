import torch
import torch.nn.functional as F

from tqdm.notebook import tqdm


# def vae_loss(x, x_reconstructed, z_mean, z_logvar):
#     recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
#     kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
#     return recon_loss + kl_divergence


def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    train_loss_batch = 0
    pbar = tqdm(train_loader, unit='batch')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, z_mean, z_logvar = model(data)

        recon_loss = model.gaussian_likelihood(output, model.log_scale, data)

        z = model.reparameterize(z_mean, z_logvar)
        std = torch.exp(z_logvar / 2)
        kl = model.kl_divergence(z, z_mean, std)

        elbo = kl - recon_loss
        loss = elbo.mean()

        # loss = vae_loss(data, output, z_mean, z_logvar)

        train_loss_batch += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(desc="Epoch = {} batch_loss={:.4f} batch_id={}" \
                             .format(epoch+1, loss.item(), batch_idx))

    train_loss_epoch = train_loss_batch / len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    print("\nTrain set: Average loss: {:.4f}\n".format(train_loss_epoch))


def test(model, device, test_loader, test_losses):
    model.eval()
    test_loss_batch = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, z_mean, z_logvar  = model(data)

            recon_loss = model.gaussian_likelihood(output, model.log_scale, data)

            z = model.reparameterize(z_mean, z_logvar)
            std = torch.exp(z_logvar / 2)
            kl = model.kl_divergence(z, z_mean, std)

            elbo = kl - recon_loss
            loss = elbo.mean()
            test_loss_batch += loss.item()
            # test_loss_batch += vae_loss(data, output, z_mean, z_logvar ).item()

    test_loss_epoch = test_loss_batch / len(test_loader.dataset)
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


def reconstruct_samples(model, device, latent_dim=512, num_samples=50):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        sample = model.decoder(z).cpu()
    return sample
    