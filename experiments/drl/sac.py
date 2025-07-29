# Permit to run this script from any directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports
import json, pickle, time, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from tabulate import tabulate
from stable_baselines3.common.buffers import ReplayBuffer

# Save logs for sac
def save_logs(run_name, history, folder=''):
    if not os.path.exists('data'):
        os.makedirs('data')
    if folder == '':
        with open(f'logs/{run_name}.json', 'w') as f:
            json.dump(history, f, indent=4)
    else:
        with open(f'logs/{folder}/{run_name}.json', 'w') as f:
            json.dump(history, f, indent=4)
    print("Logs")

# Args
@dataclass
class Args:
    env_id: str = 'HalfCheetah-v5'  # Default environment, can be overridden
    seed: int = 10
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate
    buffer_size: int = int(1e6) # Replay buffer size
    batch_size: int = 256
    learning_starts: int = 5000 # Steps before starting learning
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2  # Frequency of policy updates (and alpha if autotune)
    target_network_frequency: int = 1 # Frequency of target network updates (often equal to policy_frequency)
    alpha: float = 0.2  # Initial entropy regularization coefficient
    autotune: bool = True # Whether to automatically tune alpha
    max_grad_norm: float = 0.5 # Gradient clipping
    # num_steps for MetricsContainer, can be how often to log/show metrics
    log_interval_steps: int = 2048 # How often to show metrics
    run_name: str = 'run_name'

# Metrics container to store training metrics
@dataclass
class MetricsContainer:
    log_interval_steps: int 
    episode_rewards: list[float] = field(default_factory=list)
    qf1_losses: list[float] = field(default_factory=list)
    qf2_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    alpha_losses: list[float] = field(default_factory=list) 
    alphas: list[float] = field(default_factory=list)
    entropy_mean: list[float] = field(default_factory=list) 
    history: list[dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.global_step_at_last_log = 0
        self.num_logs_shown = 0
        
    def reset_epoch_metrics(self):
        self.qf1_losses = []
        self.qf2_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.alphas = []
        
    def show(self, global_step: int):
        time_since_last_log = time.time() - self.last_log_time
        steps_this_interval = global_step - self.global_step_at_last_log
        
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        datas = [
            ["Total Timesteps", f"{global_step:.0f}"],
            ["Log Entries", f"{self.num_logs_shown + 1:.0f}"],
            ["Mean Reward (last N eps)", f"{mean_reward:.2f}"],
            ["Time Elapsed", f"{(time.time() - self.start_time):.2f} s"],
            ["Steps per Second", f"{steps_this_interval / time_since_last_log:.2f}" if time_since_last_log > 0 else "N/A"],
            ["Mean Actor Loss", f"{np.mean(self.actor_losses):.2f}" if self.actor_losses else "N/A"],
            ["Mean QF1 Loss", f"{np.mean(self.qf1_losses):.2f}" if self.qf1_losses else "N/A"],
            ["Mean QF2 Loss", f"{np.mean(self.qf2_losses):.2f}" if self.qf2_losses else "N/A"],
            ["Mean Alpha", f"{np.mean(self.alphas):.2f}" if self.alphas else "N/A"],
        ]
        if self.alpha_losses: 
             datas.append(["Mean Alpha Loss", f"{np.mean(self.alpha_losses):.2f}"])
        if self.entropy_mean:
             datas.append(["Mean Policy Entropy", f"{np.mean(self.entropy_mean):.2f}"])

        print(tabulate(datas, headers=["Metrics", "Value"], tablefmt="fancy_grid"))
        
        current_metrics_dict = {
            "total_steps": global_step,
            "log_entries": self.num_logs_shown + 1,
            "mean_reward_last_n_eps": mean_reward,
            "time_elapsed": time.time() - self.start_time,
            "steps_per_second": steps_this_interval / time_since_last_log if time_since_last_log > 0 else 0,
            "mean_actor_loss": np.mean(self.actor_losses) if self.actor_losses else None,
            "mean_qf1_loss": np.mean(self.qf1_losses) if self.qf1_losses else None,
            "mean_qf2_loss": np.mean(self.qf2_losses) if self.qf2_losses else None,
            "mean_alpha": np.mean(self.alphas) if self.alphas else None,
            "mean_policy_entropy": np.mean(self.entropy_mean) if self.entropy_mean else None,
        }
        if self.alpha_losses:
            current_metrics_dict["mean_alpha_loss"] = np.mean(self.alpha_losses)

        self.history.append(current_metrics_dict)
        
        self.episode_rewards = [] 
        self.entropy_mean = []    
        self.reset_epoch_metrics()
        self.last_log_time = time.time()
        self.global_step_at_last_log = global_step
        self.num_logs_shown += 1

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Actor network
class Actor(nn.Module):
    def __init__(self, env, layer_init_fn=None):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(self.obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, self.action_dim)
        self.fc_logstd = nn.Linear(256, self.action_dim)

        if layer_init_fn: 
            self.fc1 = layer_init_fn(self.fc1)
            self.fc2 = layer_init_fn(self.fc2)
            self.fc_mean = layer_init_fn(self.fc_mean, std=0.01) 
            self.fc_logstd = layer_init_fn(self.fc_logstd, std=0.01)

        action_scale = torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        action_bias = torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) 
        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample() 
        
        y_t = torch.tanh(x_t) 
        action = y_t * self.action_scale + self.action_bias 
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        log_prob = log_prob.sum(1, keepdim=True)
        
        eval_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, eval_mean 
    
# Soft Q-Network
class SoftQNetwork(nn.Module):
    def __init__(self, env, layer_init_fn=None):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if layer_init_fn:
            self.fc1 = layer_init_fn(self.fc1)
            self.fc2 = layer_init_fn(self.fc2)
            self.fc3 = layer_init_fn(self.fc3, std=1.0) 

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Float32Wrapper to ensure observations and actions are float32
class Float32Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        if isinstance(self.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=self.observation_space.low.astype(np.float32),
                high=self.observation_space.high.astype(np.float32),
                shape=self.observation_space.shape,
                dtype=np.float32
            )
        
        if isinstance(self.action_space, spaces.Box):
            self.action_space = spaces.Box(
                low=self.action_space.low.astype(np.float32),
                high=self.action_space.high.astype(np.float32),
                shape=self.action_space.shape,
                dtype=np.float32
            )

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return next_obs.astype(np.float32), float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        seed = kwargs.pop('seed', None)
        if seed is not None:
            obs, info = self.env.reset(seed=seed, **kwargs)
        else:
            obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32), info

# Main SAC class
class SACTorch(nn.Module):
    def __init__(self, env_or_env_id: str | gym.Env, args: Args, 
                 actor_optimizer_cls=torch.optim.Adam, 
                 q_optimizer_cls=torch.optim.Adam,
                 alpha_optimizer_cls=torch.optim.Adam,
                 optimizer_params: dict = {}):
        super(SACTorch, self).__init__()
        
        self.args = args
        
        # Device setup
        self.device = torch.device('cpu')
        print(self.device)

        # Reproducibility
        if args.seed is not None:
            print(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if self.device != torch.device("mps"): 
                 torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True) 

        assert isinstance(env_or_env_id, (str, gym.Env)), "env_or_env_id must be a string or a gym.Env instance."
        
        # Environment setup
        if isinstance(env_or_env_id, str):
            _env = gym.make(args.env_id)
        else:
            _env = env_or_env_id 
        self.env = Float32Wrapper(_env) 
        if not hasattr(self.env, 'single_observation_space'):
            self.env.single_observation_space = self.env.observation_space
        if not hasattr(self.env, 'single_action_space'):
            self.env.single_action_space = self.env.action_space
        
        assert isinstance(self.env.action_space, gym.spaces.Box), "SAC requires continuous action space (gym.spaces.Box)."
        assert self.env.single_observation_space.dtype == np.float32, "Observation space dtype should be float32 after wrapping."
        if isinstance(self.env.single_action_space, spaces.Box): # Action space peut être autre chose
             assert self.env.single_action_space.dtype == np.float32, "Action space dtype should be float32 after wrapping for Box spaces."


        layer_initializer = None 

        # Actor networks
        self.actor = Actor(self.env, layer_initializer).to(self.device)
        self.qf1 = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf2 = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf1_target = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf2_target = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Set the optimizer
        self.actor_optimizer = actor_optimizer_cls(self.actor.parameters(), **optimizer_params)
        self.q_optimizer = q_optimizer_cls(list(self.qf1.parameters()) + list(self.qf2.parameters()), **optimizer_params)

        # Alpha networks
        self.alpha = args.alpha
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = alpha_optimizer_cls([self.log_alpha], lr=args.q_lr) 
            self.alpha = self.log_alpha.exp().item() 
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space, 
            self.device, 
            handle_timeout_termination=False, 
        )
        if args.seed is not None:
            self.replay_buffer._rng = np.random.default_rng(args.seed)
        
        self.metrics = MetricsContainer(args.log_interval_steps)
        self.global_step = 0

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device) 
        if len(obs_tensor.shape) == 1: 
             obs_tensor = obs_tensor.unsqueeze(0)
        action, _, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action.detach().cpu().numpy()[0]

    def learn(self, total_timesteps: int):
        obs, _ = self.env.reset(seed=self.args.seed if self.args.seed is not None else None)
        self.env.reset(seed=self.args.seed)       
        self.env.action_space.seed(self.args.seed)
        self.env.observation_space.seed(self.args.seed)
        current_episode_reward = 0
        current_episode_len = 0
        
        for step in range(total_timesteps):
            self.global_step +=1
            current_episode_len += 1

            if self.global_step < self.args.learning_starts:
                action = self.env.action_space.sample()
                if isinstance(self.env.action_space, spaces.Box) and self.env.action_space.dtype == np.float32:
                    action = action.astype(np.float32)
            else:
                action = self.get_action(obs) 

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            current_episode_reward += reward
            
            self.replay_buffer.add(obs, next_obs, action, reward, terminated, [info]) 

            obs = next_obs

            if done:
                self.metrics.episode_rewards.append(current_episode_reward)
                obs, _ = self.env.reset(seed=self.args.seed if self.args.seed is not None else None)
                current_episode_reward = 0
                current_episode_len = 0

            if self.global_step > self.args.learning_starts:
                self.train_step()

            if self.global_step % self.args.log_interval_steps == 0 and self.global_step > 0:
                self.metrics.show(self.global_step)
        
        self.env.close()
        return self.metrics.history

    def train_step(self):
        data = self.replay_buffer.sample(self.args.batch_size)
        
        observations = data.observations.float() 
        actions = data.actions.float()
        next_observations = data.next_observations.float()
        rewards = data.rewards.float() 
        dones = data.dones.float()     

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1.0 - dones.flatten()) * self.args.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        self.metrics.qf1_losses.append(qf1_loss.item())
        self.metrics.qf2_losses.append(qf2_loss.item())

        if self.global_step % self.args.policy_frequency == 0:
            for _ in range(self.args.policy_frequency): 
                pi, log_pi, _ = self.actor.get_action(observations) 
                qf1_pi = self.qf1(observations, pi) 
                qf2_pi = self.qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi.view(-1)) - min_qf_pi).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()
                self.metrics.actor_losses.append(actor_loss.item())
                self.metrics.entropy_mean.append(-log_pi.mean().item())
                if self.args.autotune:
                    alpha_loss = (-self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
                    self.metrics.alpha_losses.append(alpha_loss.item())
            self.metrics.alphas.append(self.alpha)

        if self.global_step % self.args.target_network_frequency == 0: 
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    def test(self, num_episodes: int = 10):
        test_rewards = []
        base_seed = self.args.seed if self.args.seed is not None else random.randint(0, 1e6)
        for i in range(num_episodes):
            obs, _ = self.env.reset(seed=base_seed + i) 
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs
            test_rewards.append(episode_reward)
            print(f"Test Episode {i+1}/{num_episodes} Reward: {episode_reward}")
        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        print(f"Test Results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

if __name__ == '__main__':
    from optimizers import * 

    total_training_steps = 1_000_000
    optimizer_to_test = [
                        (Anolog, 'Anolog', {'lr': 3e-4}),
                        #(Adan, "AdanTuned", {'lr': 3e-3, 'betas': (0.98, 0.92, 0.98)}),
                        #(Ano, 'AnoTuned', {'lr': 3e-4, 'betas': (0.95, 0.99)}),
                        #(torch.optim.Adam, 'AdamTuned', {'lr': 3e-4, 'betas': (0.95, 0.995)}),
                        #(torch.optim.Adam, 'AdamBaseline', {'lr': 3e-4}),
                        #(Adan, 'AdanBaseline', {'lr': 3e-4}),
                        #(Ano, 'AnoBaseline', {'lr': 3e-4}),
                        #(Lion, 'LionBaseline', {'lr': 3e-4}),
                        #(Lion, 'LionTuned', {'lr': 3e-4, 'betas': (0.92, 0.99)}),
                        #(Grams, 'GramsBaseline', {'lr': 3e-4}),
                        #(Grams, 'GramsTuned', {'lr': 3e-3, 'betas': (0.9, 0.95)}),
                         ] 
    # [2, 5, 10, 42, 52, 72, 108, 444, 512, 1000]
    #all_seeds = [5, 52, 72, 108, 512, 1000]
    all_seeds = [52, 72, 108, 444, 512, 1000]

    
    env_id_to_test = 'Ant-v5'

    for seed in all_seeds:
        for optimizer_cls, optimizer_name, optim_params in optimizer_to_test:
            print(f"Testing SAC with {optimizer_name} on {env_id_to_test} with seed {seed}")
            args = Args(seed=seed, 
                        env_id=env_id_to_test, 
                        q_lr=optim_params['lr'], 
                        policy_lr=optim_params['lr'],
                        run_name=f"{optimizer_name}-seed{seed}",)

            sac_agent = SACTorch(args.env_id, args, 
                                actor_optimizer_cls=optimizer_cls, 
                                q_optimizer_cls=optimizer_cls,
                                alpha_optimizer_cls=optimizer_cls,
                                optimizer_params=optim_params)
            
            history = sac_agent.learn(total_timesteps=total_training_steps)

            save_logs(args.run_name, history, folder=f'experiments/drl/logs/{env_id_to_test}')