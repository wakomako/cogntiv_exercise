from pathlib import Path
from typing import Optional

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from siren.baseline.dataset import SIRENDataset, ImageEncoding
from siren.baseline.siren_model import Siren
from siren.utils import gradient, laplace


def init_dataset(sidelength: int, image_encoding: ImageEncoding):
    data_path = Path(__file__).parent.parent / 'Data' / str(sidelength)

    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img_folder = SIRENDataset(data_path, sidelength=sidelength, image_encoding=image_encoding, transform=transform)

    return img_folder


def init_baseline_model() -> Siren:
    in_features = 3 #baseline

    # hidden_features =256 #baseline
    hidden_features =512

    # hidden_layers=3 # baseline
    hidden_layers=6
    multi_img_siren = Siren(in_features=in_features, out_features=3, hidden_features=hidden_features,
                      hidden_layers=hidden_layers, outermost_linear=True)

    multi_img_siren.cuda()

    return multi_img_siren


def visulize_grid(model: nn.Module, dataset: Dataset) -> None:
    sidelength = dataset.sidelength
    dataloader = DataLoader(dataset, batch_size=3, pin_memory=True, num_workers=0, shuffle=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 6))
    inputs = next(iter(dataloader))

    for i in range(len(inputs[0])):

        index, model_input, ground_truth = (inputs[j][i] for j in range(len(inputs)))

        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        model_output, coords = model(model_input)

        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)

        axes[i,0].imshow(model_output.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[i,1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength, sidelength).detach().numpy())
        axes[i,2].imshow(img_laplacian.cpu().view(sidelength, sidelength).detach().numpy())
        plt.show(block=False)
        plt.pause(1)




def train(dataset: Dataset, model: nn.Module, save_checkpoint: bool = False, checkpoint_tag: str = ''):
    """ train the baseline SIREN"""
    num_epochs = 500
    dataloader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, num_workers=0, shuffle=True)

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    visualize_epoch_frequency = 100


    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):

            index, model_input, ground_truth = batch
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

            model_output, coords = model(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        generalization_loss = eval_noisy_coords(model, dataset)

        print(f"epoch {epoch}, train loss = {loss} , generalization_loss = {generalization_loss}")

        if (epoch+1) % visualize_epoch_frequency == 0:
            visulize_grid(model, dataset)

    print("finished training")
    if save_checkpoint:
        save_chkpt(epoch, model, loss, optim, checkpoint_tag=checkpoint_tag)

    return model




def save_chkpt(epoch: int, net: nn.Module, loss: float, optimizer: Optional[torch.optim.Optimizer] = None, checkpoint_tag: str = ''):

    save_dir = Path(__file__).parent.parent / 'checkpoints'

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
    }, save_dir / f"{checkpoint_tag}_epoch_{epoch}_loss_{loss}.chkpt")


def eval_noisy_coords(model: nn.Module, dataset: Dataset) -> float:
    """ return the reconstruction loss of the model on noisy inputs"""

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, num_workers=0, shuffle=True)

        samples = next(iter(dataloader))

        index, model_input, ground_truth = samples

        noisy_inputs = model_input + torch.normal(torch.zeros(model_input.shape), 0.01)

        noisy_model_output, coords = model(noisy_inputs.cuda())

        loss = ((noisy_model_output.detach().cpu() - ground_truth) ** 2).mean()

        return loss







