from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset

from siren.baseline.dataset import generate_coords, ImageEncoding
from siren.baseline.trainer import init_baseline_model


def load_chkpt(chkpt_path) -> nn.Module:
    model = init_baseline_model()
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model



def eval_model_interpolation(chkpt_path: Path, dataset: Dataset, sidelength: int, num_samples: int = 3) -> None:
    """ plots num_samples of random interpolations created by the inputted model on the inputted data"""
    out_sidelenght = sidelength
    model = load_chkpt(chkpt_path)

    dataloader = DataLoader(dataset, batch_size=2, pin_memory=True, num_workers=0, shuffle=False)

    rand_idxs = torch.randint(0, len(dataset) // 2, [num_samples])

    for j, inputs in enumerate(dataloader):

        if j not in rand_idxs:
            continue


        index_0, inputs_0, gt_0 = (inputs[i][0] for i in range(len(inputs)))
        index_1, inputs_1, gt_1 = (inputs[i][1] for i in range(len(inputs)))

        interpolated_input = 0.5 * inputs_0 + 0.5 * inputs_1

        output_0, _ = model(inputs_0.cuda())
        output_1, _ = model(inputs_1.cuda())


        model_output, coords = model(interpolated_input.cuda())

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

def eval_model_upsample(chkpt_path: Path, dataset: Dataset, sidelength: int, image_encoding: ImageEncoding, num_samples: int = 3):
    """ plots num_samples of random up-samples created by the inputted model on the inputted data"""

    model = load_chkpt(chkpt_path)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=True)

    for j, inputs in enumerate(dataloader):
        if j == num_samples:
            break
        index, coords_input, gt = inputs
        new_sidelenght = 256
        upsampled_input = generate_coords(img_index=index.item(), sidelength=new_sidelenght, image_encoding=image_encoding,
                                    class_embeddings=dataset.class_coords)

        upsampled_model_output, coords = model(upsampled_input.cuda())
        orig_model_output, coords = model(coords_input.cuda())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(gt.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[0].set_title('GT')
        axes[1].imshow(orig_model_output.cpu().view(sidelength, sidelength, 3).detach().numpy())
        axes[1].set_title('original_input')
        axes[2].imshow(upsampled_model_output.cpu().view(new_sidelenght, new_sidelenght, 3).detach().numpy())
        axes[2].set_title('upsampled')

        plt.show(block=False)
        plt.pause(1)

    plt.pause(10)





