import argparse
from pathlib import Path

from siren.baseline.dataset import ImageEncoding
from siren.baseline.eval import eval_model_upsample
from siren.baseline.trainer import init_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Interpolate OLS')
    parser.add_argument('--side_length', default=48, type=int,
                        help='the size of the image side to train on', choices=[24, 48, 96, 144, 192, 256, 512])
    parser.add_argument('--image_encoding', default="discrete", type=ImageEncoding,
                        help='the type of image encoding to train on', choices=["discrete", "continuous"])

    parser.add_argument('--chkpt_path', default=Path(__file__).parent.parent / "checkpoints/upsample_epoch_499_loss_0.0011897414224222302.chkpt", type=Path,
                        help='path to chkpt to evaluate')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sidelength = args.side_length
    data = init_dataset(sidelength=sidelength, image_encoding=args.image_encoding)
    eval_model_upsample(args.chkpt_path, data, sidelength, args.image_encoding)
