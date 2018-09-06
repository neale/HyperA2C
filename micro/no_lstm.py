import argparse
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
import time
from math import log2
from atari_data import MultiEnvironment
from scipy.signal import lfilter

from scipy.misc import imsave

from collections import deque

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
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)
        return self.critic_linear(x), self.actor_linear(x)

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '/*')
        step = 0
        print (paths)
        self.load_state_dict(torch.load('../saved_model.pt'))
        print ('loaded agent')
        return 1

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongDeterministic-v0', type=str, help='gym environment')
    parser.add_argument('--save_dir', default='/home/neale/mount/agents', type=str, help='gym environment')
    parser.add_argument('--batch_size', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=True, action="store_true", help='renders the atari environment')
    parser.add_argument('--test', default=True, action="store_true", help='test mode sets lr=0, chooses most likely actions')
    parser.add_argument('--lstm_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount for gamma-discounted rewards')
    parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--frame_skip', default=-1, type=int, help='atari frame skipping, -1 to use default (2-4)')
    parser.add_argument('--gpu', default=0, type=int, help='gpu to use')
    return parser.parse_args()

map_loc = {
        'cuda:0': 'cuda:0',
        'cuda:1': 'cuda:0',
        'cuda:2': 'cuda:0',
        'cuda:3': 'cuda:0',
        'cuda:4': 'cuda:0',
        'cuda:5': 'cuda:0',
        'cuda:6': 'cuda:0',
        'cuda:7': 'cuda:0',
        'cpu': 'cpu',
}
def printlog(args, s, end='\n', mode='a'):
    print(s, end=end)
    f=open(args.save_dir+'log.txt',mode)
    f.write(s+'\n')
    f.close()

#discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner

# Input: batch_size x lstm_steps
def discount(rewards, gamma):
    rev_rewards = rewards[:, ::-1]
    result = lfilter([1], [1, -gamma], rev_rewards)
    return result[:, ::-1]


args = get_args()

print("initializing environments")
envs = MultiEnvironment(args.env, args.batch_size, args.frame_skip)
torch.manual_seed(args.seed)
torch.cuda.set_device(args.gpu)

print("initializing agent")
agent = Agent(envs.get_action_size()).cuda() #cuda is fine here cause we are just using it for perceptual loss and copying to discrim
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)


# values: batch_size x (lstm_steps + 1)
# logps: batch_size x lstm_steps x num_actions
# actions: batch_size x lstm_steps  (integer array)
# rewards: batch_size x lstm_steps
def cost_func(args, values, logps, actions, rewards):
    np_values = values.cpu().data.numpy()

    # generalized advantage estimation (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[:,1:] - np_values[:,:-1]
    gae = discount(delta_t, args.gamma * args.tau)
    logpys = logps.gather(2, Variable(actions).view(actions.shape[0],-1,1))
    policy_loss = -(logpys.view(gae.shape[0],-1) * Variable(torch.Tensor(gae.copy())).cuda()).sum()
    
    # l2 loss over value estimator
    rewards[:,-1] += (args.gamma * np_values[:,-1])
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = Variable(torch.Tensor(discounted_r.copy())).cuda()
    value_loss = .5 * (discounted_r - values[:,:-1]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum() # encourage lower entropy
    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    return loss


def train():
    print("Training has started")
    info = {k : torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'episodes', 'frames']}
    info['frames'] += agent.try_load(args.save_dir)*1e6
    if int(info['frames'][0]) == 0: printlog(args,'', end='', mode='w') # clear log file
    agent.load_state_dict(torch.load('../sampled_agent_100.pt'))
    
    bs = args.batch_size
    
    state = Variable(torch.Tensor(envs.reset()).view(bs,1,80,80)).cuda()
    start_time = last_disp_time = time.time()
    episode_length, epr, eploss  = np.zeros(bs), np.zeros(bs), np.zeros(bs)
    values, logps, actions, rewards = [], [], [], [] 

    i = 0
    print (agent.state_dict())
    while info['frames'][0] <= 4e7 or args.test:
        i+=1
        episode_length +=1
        #run agent on state
        value, logit = agent(state)
        #print (value)
        print (agent.state_dict().keys(), agent.state_dict()['actor_linear.weight'])
        logp = F.log_softmax(logit, dim=1)

        action = torch.exp(logp).multinomial(num_samples=1).data
        state, reward, done, _ = envs.step(action)
        if args.render:
            envs.envs[0].render()

        state = torch.tensor(state).view(args.batch_size,1,80,80).cuda()
        reward = np.clip(reward, -1, 1)
        epr+=reward

        info['frames'] += bs 
        num_frames = int(info['frames'][0])
        if num_frames % 2e6 == 0: # save every 2M frames
            printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
            #torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

        done_count = np.sum(done)
        if done_count > 0:
            if done[0] == True and time.time() - last_disp_time > 5:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, current reward {}'
                    .format(elapsed, info['episodes'][0], num_frames/1e6, epr[0]))
                last_disp_time = time.time()
            for j, d in enumerate(done):
                if d:
                    info['episodes'] += 1
                    interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                    info['run_epr'].mul_(1-interp).add_(interp * epr[j])
                    episode_length[j], epr[j], eploss = 0, 0, 0
            
        values.append(value)
        logps.append(logp)
        actions.append(action)
        rewards.append(reward)
        
        if i % args.lstm_steps == 0:
            next_value = agent(state)[0]
            if done_count > 0: 
                for j, d in enumerate(done):
                    if d: next_value[j] = 0
            values.append(next_value.cuda())

            optimizer.zero_grad()  
            loss = cost_func(args, torch.cat(values, dim=1), torch.stack(logps, dim=1), torch.cat(actions, dim=1), np.transpose(np.asarray(rewards)))
            #loss.backward()
            eploss += loss.data[0]
            #torch.nn.utils.clip_grad_norm(agent.parameters(), 40)

            #optimizer.step()
            values, logps, actions, rewards = [], [], [], []

def main():
    train()


if __name__ == '__main__':
    main()
