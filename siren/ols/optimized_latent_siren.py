import os
from pathlib import Path
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from siren.baseline.dataset import OLSDataset
from siren.baseline.siren_model import OLSiren
from siren.utils import gradient, laplace


class _netZ(nn.Module):
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)  # initializes a learnable weights embedding from the normal distribution.
        self.nz = nz

    def normalize(self):
        """
        :return: normalized version of point z
        """
        wn = self.emb.weight.norm(p=2, dim=1).data.unsqueeze(1)
        self.emb.weight.data = \
            self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        idx = idx.type(torch.long)  # due to type error
        z = self.emb(idx).squeeze()
        return z

class OLS:

    def __init__(self, sidelength: int, dim_z: int = 10, batch_size: int = 20, num_epochs: int = 200):

        self.dim_z = dim_z
        self.sidelength = sidelength
        self.batch_size = batch_size

        self.dataset, self.dataloader = self.init_dataset(self.batch_size, self.sidelength)

        self.model = self.init_model(self.dim_z)

        self.netZ = self.init_Z(self.dim_z, len(self.dataset))

        self.num_epochs = num_epochs

    @staticmethod
    def init_dataset(batch_size: int, sidelength: int) -> Tuple[OLSDataset, DataLoader]:

        data_path = Path(__file__).parent.parent / 'Data' / str(sidelength)

        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        dataset = OLSDataset(data_path, sidelength=sidelength, transform=transform)

        dataloader = DataLoader(dataset, batch_size, pin_memory=True, num_workers=0)
        return dataset, dataloader

    @staticmethod
    def init_Z(dim_z: int, num_samples: int):
        netZ = _netZ(dim_z, num_samples)
        init.normal_(netZ.emb.weight, mean=0, std=0.01)
        netZ.cuda()

        return netZ


    @staticmethod
    def init_model(dim_z: int):
        in_features = dim_z + 2

        hidden_features = 512

        hidden_layers = 6
        multi_img_siren = OLSiren(in_features=in_features, out_features=3, hidden_features=hidden_features,
                                hidden_layers=hidden_layers, outermost_linear=True)

        multi_img_siren.cuda()

        return multi_img_siren

    def train(self, checkpoint_tag: str):
        model_optim = torch.optim.Adam(lr=1e-4, params=self.model.parameters())
        latent_optim = torch.optim.Adam(lr=1e-5, params=self.netZ.parameters())
        visualize_epoch_frequency = 50

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for step, batch in enumerate(self.dataloader):


                idxs, coords, pixels = batch
                coords, pixels = coords.cuda(), pixels.cuda()

                zi = self.netZ(idxs.cuda())

                model_output, coords = self.model(coords, zi)


                loss = ((model_output - pixels) ** 2).mean()


                model_optim.zero_grad()
                latent_optim.zero_grad()

                loss.backward()
                model_optim.step()
                latent_optim.step()

                epoch_loss += loss


            self.netZ.normalize()  # z's are normalized every epoch (not iteration)

            print("epoch %d, Total loss %0.6f" % (epoch, epoch_loss))

            if (epoch + 1) % visualize_epoch_frequency == 0:
                self.visulize_grid(self.model, self.dataset)

            self.save_chkpt(epoch, self.model, self.netZ, loss, checkpoint_tag=checkpoint_tag)

        print("finished training")
        self.save_chkpt(epoch, self.model, self.netZ, loss, checkpoint_tag=checkpoint_tag)



    def save_chkpt(self, epoch: int, model: nn.Module, embeddings: nn.Module, loss: float, checkpoint_tag: str):
        save_dir = Path(os.path.dirname(__file__)) / 'checkpoints'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'embeddings_state_dict': embeddings.state_dict(),
            'loss': loss,
        }, save_dir / f"{checkpoint_tag}_epoch_{epoch}_loss_{loss}.chkpt")


    def visulize_grid(self, model, dataset):
        dataloader = DataLoader(dataset, batch_size=3, pin_memory=True, num_workers=0, shuffle=True)

        fig, axes = plt.subplots(3, 3, figsize=(18, 6))
        inputs = next(iter(dataloader))

        for i in range(len(inputs[0])):
            idxs, coords, pixels = inputs[0][i], inputs[1][i], inputs[2][i]


            coords, pixels = coords.cuda(), pixels.cuda()

            zi = self.netZ(idxs.cuda())


            model_output, coords = model(coords.unsqueeze(0), zi.unsqueeze(0))


            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            axes[i, 0].imshow(model_output.cpu().view(self.sidelength, self.sidelength, 3).detach().numpy())
            axes[i, 1].imshow(img_grad.norm(dim=-1).cpu().view(self.sidelength, self.sidelength).detach().numpy())
            axes[i, 2].imshow(img_laplacian.cpu().view(self.sidelength, self.sidelength).detach().numpy())
            plt.show(block=False)
            plt.pause(1)





if __name__ == '__main__':
    sidelen = 96
    ols_trainer = OLS(sidelength=sidelen)
    ols_trainer.train(checkpoint_tag="ols_sidelen_96")

