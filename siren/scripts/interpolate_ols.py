import argparse
from pathlib import Path
from siren.ols.eval_ols import eval_ols_interpolation
from siren.ols.optimized_latent_siren import OLS


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolate OLS')
    parser.add_argument('--chkpt_path', default=Path(__file__).parent.parent / "checkpoints/ols_sidelen_96_epoch_199_loss_0.01371758058667183.chkpt", type=Path,
                        help='path to chkpt to evaluate')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sidelength = 96
    args = parse_args()
    chkpt_path = args.chkpt_path
    dataset, dataloader = OLS.init_dataset(batch_size=2, sidelength=sidelength)
    eval_ols_interpolation(chkpt_path, dataset, sidelength)
