import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="Parsing parameters for 3D occupancy.")

    # config file
    parser.add_argument("--config", type=str, required=True,
                        default="config/sdf/config.json",
                        help="network configuration")

    # data file
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="output directory")

    # training options
    parser.add_argument('--batch_size', type=int, default=49152,
                        help='number of points in a batch')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training')

    # validation options
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to perform marching cubes for input shapes')

    # misc
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load')
    parser.add_argument('--clamp_distance', type=float, default=0.1,
                        help='the value range for sdfs')


    args = parser.parse_args()
    return args