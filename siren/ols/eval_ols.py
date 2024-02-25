from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import Dataset

from siren.ols.optimized_latent_siren import OLS



def load_chkpt(chkpt_path: Path, dataset: Dataset) -> Tuple[nn.Module, nn.Module]:
    model = OLS.init_model(dim_z=10)
    embeddings = OLS.init_Z(dim_z=10, num_samples=len(dataset))
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings.load_state_dict(checkpoint['embeddings_state_dict'])

    return model, embeddings



def eval_ols_interpolation(chkpt_path: Path, dataset: Dataset, sidelength: int, num_samples: int = 3) -> None:

    out_sidelenght = sidelength
    model, embeddings = load_chkpt(chkpt_path, dataset)

    nnbrs = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')

    np_embeddings = embeddings.emb.weight.detach().cpu().numpy()

    nbrs = nnbrs.fit(np_embeddings)

    nn_distances, nn_indices = nbrs.kneighbors(np_embeddings)

    for i in range(num_samples):
        rand_idx = torch.randint(0,len(dataset), [1]).item()

        sample_idx, nn_idx = nn_indices[rand_idx]

        idxs_0, inputs_0, gt_0 = dataset[sample_idx]
        idxs_1, inputs_1, gt_1 = dataset[nn_idx]

        z0 = embeddings(torch.tensor([idxs_0]).cuda())
        z1 = embeddings(torch.tensor([idxs_1]).cuda())

        interpolated_embedding = (z0 + z1) / 2
        normalized_interpolated = F.normalize(interpolated_embedding.unsqueeze(0))


        output_0, _  = model(inputs_0.unsqueeze(0).cuda(), z0.unsqueeze(0))
        output_1, _  = model(inputs_1.unsqueeze(0).cuda(), z1.unsqueeze(0))

        model_output, coords = model(inputs_0.unsqueeze(0).cuda(), normalized_interpolated)

        fig, axes = plt.subplots(1, 5, figsize=(18, 6))

        axes[0].imshow(gt_0.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[0].set_title('GT_0')

        axes[1].imshow(output_0.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[1].set_title('output_0')

        axes[2].imshow(gt_1.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[2].set_title('GT_1')

        axes[3].imshow(output_1.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[3].set_title('output_1')

        axes[4].imshow(model_output.cpu().view(out_sidelenght, out_sidelenght, 3).detach().numpy())
        axes[4].set_title('output_interpolated')

        plt.show(block=False)
        plt.pause(1)

    plt.pause(10)

