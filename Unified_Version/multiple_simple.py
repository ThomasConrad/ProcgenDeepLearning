#!/usr/bin/env python
# coding: utf-8

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

import csv


# ### Hyperparameters
# Leave unchanged between comparison runs

# Hyperparameters
total_steps = 8e6
num_envs = 96
num_levels = 100
num_steps = 256
num_epochs = 3
batch_size = 1024
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
feature_dim = 256

env_name = 'starpilot,coinrun,bigfish'
use_mixreg = False
gamma = 0.999
increase = 2 # How much to augment the dataset with mixreg
alpha = 0.5 # Alpha value to use for the beta-distribution in mixreg


# ### Network Definition
# Leave unchanged between comparison runs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureModel(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels, 
                 feature_dim,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=feature_dim)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


# ## Environment and training definition

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels,env_name=env_name)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

eval_environments = {}
# Define validation environments
env_names = env_name.split(',')
for e in env_names:
    eval_environments[e] = make_env(num_envs, start_level=num_levels, num_levels=num_levels,env_name=e)
print("Nr. of environments:", len(eval_environments))

# Define network
encoder = ImpalaModel(3,256)
policy = Policy(encoder, 256, env.action_space.n)
policy.cuda()

# Define optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5, weight_decay = 1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma=gamma
)

## Filename for checkpoints
checkpoint_file_name = 'checkpoint_multi'
data_log_file_name = 'training_stats_multi'
if use_mixreg:
    checkpoint_file_name += '_mixreg'
    
data_log_file_name += '.csv'
checkpoint_file_name += '.pt'


# ## Training Loop

# Run training
obs = env.reset()
v_obs = eval_environments[list(eval_environments)[0]].reset()
step = 0

data_log = []
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    if use_mixreg:
        generator = storage.get_mix_generator(increase, alpha, batch_size)
    else:
        generator = storage.get_generator(batch_size)
        
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      ratio = torch.exp(new_log_prob - b_log_prob)
      clipped_ratio = ratio.clamp(min=1.0 - eps,max=1.0 + eps)
      policy_reward = torch.min(ratio * b_advantage, clipped_ratio * b_advantage)
      pi_loss = -policy_reward.mean()

      # Clipped value function objective
      V_clip = b_value + (new_value-b_value).clamp(-eps,eps)
      vf_loss = torch.max((b_value - b_returns) ** 2, (V_clip - b_returns) ** 2)
      value_loss = 0.5*vf_loss.mean()

      # Entropy loss
      entropy_loss = new_dist.entropy().mean()

      # Backpropagate losses
      loss = pi_loss + value_coef * value_loss - entropy_coef * entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  ## VALIDATION ##
  # Evaluate policy
  policy.eval()
  validation_rewards = []
  for e in eval_environments:
      total_reward = []
      for _ in range(num_steps):
        # Use policy
        v_action, v_log_prob, v_value = policy.act(v_obs)

        # Take step in environment
        v_obs, v_reward, v_done, v_info = eval_environments[e].step(v_action)
        total_reward.append(torch.Tensor(v_reward))

      # Calculate average return
      total_reward = torch.stack(total_reward).sum(0).mean(0)
      validation_rewards.append(total_reward.item())
  ## END OF VALIDATION ##
      

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}\tMean validation rewards: {validation_rewards}')
  data_point = [step, storage.get_reward().item()]
  for r in validation_rewards:
    data_point.append(r)
  data_log.append(data_point)
    
with open(data_log_file_name, 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerows(data_log)

print('Completed training!')
torch.save(policy.state_dict, checkpoint_file_name)
