import argparse    

def load_args():
    # HyperGAN args
    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--z', default=512, type=int, help='latent space width')
    parser.add_argument('-ze', '--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=20, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='mednet', type=str)
    parser.add_argument('-d', '--dataset', default='cifar', type=str)
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool, help='sample from real layers')
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--n_actions', default=6, type=int)
    # A3C args
    parser.add_argument('--env', default='PongDeterministic-v4', type=str, help='')
    parser.add_argument('--processes', default=1, type=int, help='')
    parser.add_argument('--render', default=False, type=bool, help='')
    parser.add_argument('--test', default=False, type=bool, help='')
    parser.add_argument('--rnn_steps', default=20, type=int, help='')
    parser.add_argument('--lr', default=1e-4, type=float, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--tau', default=1.0, type=float, help='')
    parser.add_argument('--horizon', default=0.99, type=float, help='')
    parser.add_argument('--hidden', default=256, type=int, help='')
    parser.add_argument('--frame_skip', default=-1, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='')
    parser.add_argument('--exp', default='0', type=str, help='')
    parser.add_argument('--scratch', default=False, type=bool, help='')
    args = parser.parse_args()
    return args


