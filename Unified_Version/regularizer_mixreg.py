#!/usr/bin/env python
# coding: utf-8

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

import csv


# Hyperparameters
total_steps = 8e6
num_envs = 64
num_levels = 100
num_steps = 256
num_epochs = 3
batch_size = 1024
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
feature_dim = 256
env_name = 'starpilot'
use_mixreg  = False
with_background = False # Use backgrounds for the environments
increase = 2 # How much to augment the dataset with mixreg
alpha = 0.5 # Alpha value to use for the beta-distribution in mixreg


# ### Network Definition
# Leave unchanged between comparison runs

# Network definition
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#class RandomCrop(nn.Module, size=48):
#    def forward(self, x):
#        n, c, h, w = x.shape
#        w1 = torch.randint(0, w - size + 1, (n,))
#        h1 = torch.randint(0, h - size + 1, (n,))
#        cropped = torch.empty((n, c, size, size), dtype=x.dtype, device=x.device)
#        for i, (img, w11, h11) in enumerate(zip(x, w1, h1)):
#            cropped[i][:] = img[:, h11:h11 + size, w11:w11 + size]
#        return cropped

#class RandomCutout(nn.Module):
#    def __init__(self, min_cut=4, max_cut=24):
#      super().__init__()
#      self.min_cut = min_cut
#      self.max_cut = max_cut
#    def forward(self, x):
#        if not self.training:
#            return x
#        n, c, h, w = x.shape
#        w_cut = torch.randint(self.min_cut, self.max_cut + 1, (n,))
#        h_cut = torch.randint(self.min_cut, self.max_cut + 1, (n,))
#        fills = torch.randint(0, 255, (n, c, 1, 1)) # assume uint8.
#        for img, wc, hc, fill in zip(x, w_cut, h_cut, fills):
#            w1 = torch.randint(w - wc + 1, ()) # uniform over interior
#            h1 = torch.randint(h - hc + 1, ())
#            img[:, h1:h1 + hc, w1:w1 + wc] = fill
#        return x

class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.layers = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    return self.layers(x) + x

class ImpalaBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ResidualBlock(out_channels),
        ResidualBlock(out_channels)
    )

  def forward(self, x):
    return self.layers(x)

class SimpleImpalaEncoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        ImpalaBlock(in_channels, out_channels=16),
        ImpalaBlock(in_channels=16, out_channels=32),
        ImpalaBlock(in_channels=32, out_channels=32),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=2048, out_features=feature_dim),
        nn.ReLU()
    )
  
  def forward(self, x):
    return self.layers(x)

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.BatchNorm2d(num_features=128),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


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
env = make_env(num_envs, num_levels=num_levels, env_name=env_name, use_backgrounds=with_background)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define validation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels, env_name=env_name, use_backgrounds=with_background)

# Define network
encoder = SimpleImpalaEncoder(3,feature_dim)
policy = Policy(encoder, feature_dim, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5,weight_decay = 1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

## Filename for checkpoints
data_log_file_name = 'training_stats'
checkpoint_file_name = 'checkpoint'

if use_mixreg:
    data_log_file_name += '_mixreg'
    checkpoint_file_name += '_mixreg'
if with_background:
    data_log_file_name += '_background'
    checkpoint_file_name += '_background'

data_log_file_name += '.csv'
checkpoint_file_name += '.pt'


# ## Training Loop

# Run training
obs = env.reset()
v_obs = eval_env.reset()
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
  info_stack = []
  for _ in range(num_steps):
    # Use policy
    v_action, v_log_prob, v_value = policy.act(v_obs)

    # Take step in environment
    v_obs, v_reward, v_done, v_info = eval_env.step(v_action)
    info_stack.append(v_info)

  # Calculate average return
  valid_score = []
  for i in range(num_steps):
    info = info_stack[i]
    valid_score.append([d['reward'] for d in info])
  valid_score = torch.Tensor(valid_score)
  validation_reward = valid_score.mean(1).sum(0)
  ## END OF VALIDATION ##

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}\tMean validation reward: {validation_reward}')
  data_point = [step, storage.get_reward().item(), validation_reward.item()]
  data_log.append(data_point)
    
  with open(data_log_file_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_log)
  
  torch.save(policy.state_dict(), checkpoint_file_name)

print('Completed training!')

