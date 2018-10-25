import torch
import torchvision
import torch.distributions.multivariate_normal as N

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np

import netdef
import utils


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad=grad
    return z


def batch_zero_grad(nets):
    for module in nets:
        module.zero_grad()


def batch_zero_optim_hn(optim):
    optim['optimE'].zero_grad()
    optim['optimG'][0].zero_grad()
    optim['optimG'][1].zero_grad()
    optim['optimG'][2].zero_grad()
    optim['optimG'][3].zero_grad()
    optim['optimG'][4].zero_grad()
    optim['optimG'][5].zero_grad()
    return optim


def batch_update_optim(optimizers):
    for optim in optimizers:
        optim.step()

def free_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = True


def frozen_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoding, prior):
    mean_p = torch.mean(prior, dim=0, keepdim=True)
    mean_q = torch.mean(encoding, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_p, mean_q)

    cov_p = torch.matmul((prior-mean_p).transpose(0, 1), prior-mean_p)
    cov_p /= 1000 - 1
    cov_q = torch.matmul((encoding-mean_q).transpose(0, 1), encoding-mean_q)
    cov_q /= 1000 - 1
    cov_loss = F.mse_loss(cov_p, cov_q)
    return mean_loss, cov_loss


def pretrain_encoder(args, HyperNet, optim):
    j = 0
    final = 100.
    e_batch_size = 1000
    x_dist = create_d(args.ze)
    z_dist = create_d(args.z)
    for j in range(500):
        x = sample_d(x_dist, e_batch_size)
        z = sample_d(z_dist, e_batch_size)
        codes = HyperNet.encoder(x)
        for i, code in enumerate(codes):
            code = code.view(e_batch_size, args.z)
            mean_loss, cov_loss = pretrain_loss(code, z)
            loss = mean_loss + cov_loss
            loss.backward(retain_graph=True)
        optim['optimE'].step()
        HyperNet.encoder.zero_grad()
        print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
            j, mean_loss.item(), cov_loss.item()))
        final = loss.item()
        if loss.item() < 0.1:
            print ('Finished Pretraining Encoder')
            break
    return HyperNet, optim


def get_policy_weights(args, HyperNet, optim):
    # generate embedding for each layer
    x_dist = create_d(args.ze)
    z_dist = create_d(args.z)
    #batch_zero_grad([HyperNet.encoder] + HyperNet.generators)
    z = sample_d(x_dist, args.batch_size)
    codes = HyperNet.encoder(z)
    layers = []
    # decompress to full parameter shape
    for (code, gen) in zip(codes, HyperNet.generators):
        rand = np.random.randint(args.batch_size)
        layers.append(gen(code)[rand])
    # Z Adversary 
    for code in codes:
        noise = sample_d(z_dist, args.batch_size)
        d_real = HyperNet.adversary(noise)
        d_fake = HyperNet.adversary(code.contiguous())
        d_real_loss = -1 * torch.log((1-d_real).mean())
        d_fake_loss = -1 * torch.log(d_fake.mean())
        d_real_loss.backward(retain_graph=True)
        d_fake_loss.backward(retain_graph=True)
        d_loss = d_real_loss + d_fake_loss
    optim['optimD'].step()
    return layers, HyperNet, optim


def update_hn(args, loss, optim):

    scaled_loss =args.beta * loss 
    scaled_loss.backward()
    optim['optimE'].step()
    optim['optimG'][0].step()
    optim['optimG'][1].step()
    optim['optimG'][2].step()
    optim['optimG'][3].step()
    optim['optimG'][4].step()
    optim['optimG'][5].step()
<<<<<<< HEAD
    return HyperNet, optim
=======
    return optim
>>>>>>> 7e140acd4f1b56ce5754e65fe3d85750092e28da
