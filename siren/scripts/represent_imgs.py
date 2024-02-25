import argparse

from siren.baseline.dataset import ImageEncoding
from siren.baseline.trainer import init_baseline_model, init_dataset, train


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolate OLS')
    parser.add_argument('--image_encoding', default="discrete", type=ImageEncoding,
                        help='the type of image encoding to train on', choices=["discrete", "continuous"])
    parser.add_argument('--side_length', default=48, type=int,
                        help='the size of the image side to train on', choices=[24, 48, 96, 144, 192, 256, 512])
    parser.add_argument('--save_checkpoint', default=False, action='store_true',
                        help='whether to save the checkpoint at the end of training')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    sidelength = args.side_length
    data = init_dataset(sidelength=sidelength, image_encoding=args.image_encoding)
    model = init_baseline_model()
    train(data, model, save_checkpoint=args.save_checkpoint, checkpoint_tag=f"baseline_{args.image_encoding}_sidelen_{args.side_length}")
