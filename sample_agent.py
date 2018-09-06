import sys
import args
import models_small as models
#import models
import hypera2c as H
import utils
from atari_data import MultiEnvironment

import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, os, gym, time, glob, argparse, sys
import utils

from torch.optim import Adam
from scipy.misc import imresize
from scipy.signal import lfilter

os.environ['OMP_NUM_THREADS'] = '1'


class HyperNetwork(object):
    def __init__(self, args):
        #super(HyperNetwork, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'HyperNetwork'
        self.encoder = models.Encoder(args).cuda()
        self.adversary = models.DiscriminatorZ(args).cuda()
        self.generators = [
                models.GeneratorW1(args).cuda(),
                models.GeneratorW2(args).cuda(),
                models.GeneratorW3(args).cuda(),
                models.GeneratorW4(args).cuda(),
                models.GeneratorW5(args).cuda(),
                models.GeneratorW6(args).cuda()
                ]


    def save_state(self, optim, num_frames, mean_reward):
        path = 'models/{}/agent_{}.pt'.format(self.env, self.exp)
        if self.scratch:
            path = '/scratch/eecs-share/ratzlafn/' + path
        Hypernet_dict = {
                'E': utils.get_net_dict(self.encoder, optim['optimE']),
                'D': utils.get_net_dict(self.adversary, optim['optimD']),
                'W1': utils.get_net_dict(self.generators[0], optim['optimG'][0]),
                'W2': utils.get_net_dict(self.generators[1], optim['optimG'][1]),
                'W3': utils.get_net_dict(self.generators[2], optim['optimG'][2]),
                'W4': utils.get_net_dict(self.generators[3], optim['optimG'][3]),
                'W5': utils.get_net_dict(self.generators[4], optim['optimG'][4]),
                'W6': utils.get_net_dict(self.generators[5], optim['optimG'][5]),
                'num_frames': num_frames,
                'mean_reward': mean_reward
                }
        torch.save(Hypernet_dict, path)
        print ('saved agent to {}'.format(path))


    def load_state(self, optim):
        layers = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
        nets = [self.generators[0], self.generators[1], self.generators[2],
                self.generators[3], self.generators[4], self.generators[5]]
        opts = [optim['optimG'][0], optim['optimG'][1], optim['optimG'][2],
                optim['optimG'][3], optim['optimG'][4], optim['optimG'][5]]
        path = 'models/{}/agent_{}.pt'.format(self.env, self.exp)
        if self.scratch:
            path = '/scratch/eecs-share/ratzlafn/' + path
        print ('loading agent from {}'.format(path))
        HN = torch.load(path)
        self.encoder, optim['optimE'] = utils.open_net_dict(
            HN['E'], self.encoder, optim['optimE'])
        self.adversary, optim['optimD'] = utils.open_net_dict(
            HN['D'], self.adversary, optim['optimD'])
        for i in range(6):
            nets[i], opts[i] = utils.open_net_dict(HN[layers[i]], nets[i], opts[i])
        num_frames = HN['num_frames']
        mean_reward = HN['mean_reward']
        return optim, num_frames, mean_reward

    def sample_agents(self, arch, n=10):
        state  = arch.state_dict()
        path = '/scratch/eecs-share/ratzlafn/agents/'
        names = ['conv1', 'conv2', 'conv3', 'conv4', 'critic_linear', 'actor_linear']
        x_dist = H.create_d(args.ze)
        z_dist = H.create_d(args.z)
        for i in range(n):
            z = H.sample_d(x_dist, args.batch_size)
            codes = self.encoder(z)
            layers = []
            for (code, gen) in zip(codes, self.generators):
                layers.append(gen(code).mean(0))
            for name, layer in zip(names, layers):
                name = name + '.weight'
                state[name] = layer.detach()
                arch.load_state_dict(state)
            #torch.save(arch.state_dict(), path+'sampled_agent_{}'.format(i))

    
class Agent(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions):
        super(Agent, self).__init__()
        channels = 1
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.critic_linear = nn.Linear(32*5*5, 1, bias=False)
        self.actor_linear = nn.Linear(32*5*5, num_actions, bias=False)

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*5*5)
        x_c = self.critic_linear(x)
        x_a = self.actor_linear(x)
        return x_c, x_a


def load_optim(args, HyperNet):
    gen_optim = []
    w = 1e-4
    #if args.test: 
    #    lr_e, lr_d, lr_g = 0, 0, 0
    #else:
    lr_e, lr_d, lr_g = 1e-4, 1e-3, 1e-4
    for p in HyperNet.generators:
        gen_optim.append(Adam(p.parameters(), lr=lr_g, betas=(.5,.999), weight_decay=w))

    Optim = { 
        'optimE': Adam(HyperNet.encoder.parameters(), lr=lr_e, betas=(.5,.999),
            weight_decay=w, eps=1e-8),
        'optimD': Adam(HyperNet.adversary.parameters(), lr=lr_d, betas=(.9,.999),
            weight_decay=w),
        'optimG': gen_optim,
        }
    return Optim


if __name__ == "__main__":

    args = args.load_args()
    args.save_dir = '{}/'.format(args.env.lower()) 
    if args.render:  
        args.processes = 1 
        args.test = True 
    if args.test:  
        args.lr = 0
    if args.scratch:
        print ('training on server; saving to /scratch/eecs-share')
    args.n_actions = gym.make(args.env).action_space.n
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    torch.manual_seed(args.seed)
    torch.cuda.device(args.gpu)
    hypernet = HyperNetwork(args)
    optim = load_optim(args, hypernet)
    hypernet.load_state(optim)
    arch = Agent(args.n_actions)
    hypernet.sample_agents(arch)

