import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

import imageio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print('using cpu')
else:
    print('using gpu')

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
      x = x.to(device).contiguous()
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


# Make evaluation environment
import sys
env_name = sys.argv[1]
nameToPath = {'bigfish':'BigFish','bossfight':'BossFight','caveflyer':'Caveflyer','chaser':'Chaser',\
    'climber':'Climber','coinrun':'CoinRun','dodgeball':'DodgeBall','fruitbot':'FruitBot',\
        'heist':'Heist','jumper':'Jumper','leaper':'Leaper','maze':'Maze','ninja':'Ninja',\
              'plunder':'Plunder','starpilot':'Starpilot'}
path = nameToPath[env_name]
env = make_env(64, num_levels=100,env_name=env_name)
eval_env = make_env(64, num_levels=100,start_level=100,env_name=env_name)
obs = eval_env.reset()
v_obs = eval_env.reset()
frames = []
v_frames = []
total_reward = []
v_total_reward = []
encoder = SimpleImpalaEncoder(3,256)
policy = Policy(encoder, 256, eval_env.action_space.n).to(device)
policy.load_state_dict(torch.load(f"Baselines/{path}/checkpoint.pt",map_location=device), strict=False)
# Evaluate policy
policy.eval()
info_stack = []
v_info_stack = []
for _ in range(512):
# Use policy
    action, log_prob, value = policy.act(obs)
    v_action, v_log_prob, v_value = policy.act(v_obs)

    # Take step in environment
    obs, reward, done, info = env.step(action)
    v_obs, v_reward, v_done, v_info = eval_env.step(v_action)
    info_stack.append(info)
    v_info_stack.append(v_info)
    # Render environment and store
    frame = (torch.Tensor(env.render(mode='rgb_array'))*255.).byte()
    v_frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()

    frames.append(frame)
    v_frames.append(v_frame)
# Calculate average return
test_score = []
valid_score = []

for i in range(256):
    info = info_stack[i]
    v_info = v_info_stack[i]
    test_score.append([d['reward'] for d in info])
    valid_score.append([d['reward'] for d in v_info])

test_score = torch.Tensor(test_score)
valid_score = torch.Tensor(valid_score)
test_reward = test_score.mean(1).sum(0)
validation_reward = valid_score.mean(1).sum(0)
print('Average train score:',test_reward,'Average test score:', validation_reward)

# Save frames as video
frames = torch.stack(frames)
v_frames = torch.stack(v_frames)
imageio.mimsave(f'test_{path}.mp4', frames, fps=25)
imageio.mimsave(f'train_{path}.mp4', v_frames, fps=25)
