import configargparse


def data_args(parser):
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--out_dir_root", type=str)


def model_args(parser):
    parser.add_argument("--max_len", type=int, default=50)


def train_args(parser):
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--monitor", type=str)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--n_gpu", type=int, default=1)

    # logging
    parser.add_argument("--verbosity", type=int, default=2)


def test_args(parser):
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--has_targets", action="store_true")

    # logging
    parser.add_argument("--verbosity", type=int, default=2)

    # model
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--n_gpu", type=int, default=1)
