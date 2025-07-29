import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from torch.distributions.normal import Normal
import numpy as np
import time
from tabulate import tabulate
import random
from stable_baselines3.common.buffers import ReplayBuffer
import os
import pickle
import json


def save_logs(run_name, history, folder=''):
    if not os.path.exists('data'):
        os.makedirs('data')
    if folder == '':
        with open(f'logs/{run_name}.json', 'w') as f:
            json.dump(history, f, indent=4)
    else:
        with open(f'{folder}/{run_name}.json', 'w') as f:
            json.dump(history, f, indent=4)
    print("Logs")

@dataclass
class Args:
    env_id: str = "Pendulum-v1"  # SAC is typically for continuous action spaces
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

@dataclass
class MetricsContainer:
    log_interval_steps: int # Renamed from num_steps to avoid confusion with PPO's num_steps
    episode_rewards: list[float] = field(default_factory=list)
    qf1_losses: list[float] = field(default_factory=list)
    qf2_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    alpha_losses: list[float] = field(default_factory=list) # Only if autotune
    alphas: list[float] = field(default_factory=list)
    entropy_mean: list[float] = field(default_factory=list) # Can represent policy entropy
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
        # episode_rewards and entropy_mean are accumulated across epochs until reset by user or new log display
        
    def show(self, global_step: int):
        time_since_last_log = time.time() - self.last_log_time
        steps_this_interval = global_step - self.global_step_at_last_log
        
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        # Calculate mean episode length: if we log every 'log_interval_steps', and have N episodes,
        # mean_ep_len ~ log_interval_steps / N. More accurately, sum lengths and divide by N.
        # For now, let's use a simpler approximation if we don't store individual lengths.
        # This part needs careful thought if precise mean episode length is critical.
        # For SAC, episodes can be long, so this metric might be less stable per log interval.
        # Let's assume for now we have an approximate number of episodes finished in this interval
        num_episodes_finished = len(self.episode_rewards) # Number of episodes finished *since last full reset*
        
        # A simple approach if episode_rewards is reset for each log interval
        # mean_ep_len = steps_this_interval / num_episodes_finished if num_episodes_finished > 0 else float('nan')
        # If episode_rewards accumulates, this is harder. Let's stick to what PPO had for now.
        mean_ep_len_approx = self.log_interval_steps / num_episodes_finished if num_episodes_finished > 0 else float('nan')


        datas = [
            ["Total Timesteps", f"{global_step:.0f}"],
            ["Log Entries", f"{self.num_logs_shown + 1:.0f}"],
            ["Mean Reward (last N eps)", f"{mean_reward:.2f}"],
            # ["Mean Episode Length (approx)", f"{mean_ep_len_approx:.2f}"], # Potentially misleading
            ["Time Elapsed", f"{(time.time() - self.start_time):.2f} s"],
            ["Steps per Second", f"{steps_this_interval / time_since_last_log:.2f}" if time_since_last_log > 0 else "N/A"],
            ["Mean Actor Loss", f"{np.mean(self.actor_losses):.2f}" if self.actor_losses else "N/A"],
            ["Mean QF1 Loss", f"{np.mean(self.qf1_losses):.2f}" if self.qf1_losses else "N/A"],
            ["Mean QF2 Loss", f"{np.mean(self.qf2_losses):.2f}" if self.qf2_losses else "N/A"],
            ["Mean Alpha", f"{np.mean(self.alphas):.2f}" if self.alphas else "N/A"],
        ]
        if self.alpha_losses: # Only show if autotuning alpha
             datas.append(["Mean Alpha Loss", f"{np.mean(self.alpha_losses):.2f}"])
        if self.entropy_mean: # Policy entropy
             datas.append(["Mean Policy Entropy", f"{np.mean(self.entropy_mean):.2f}"])

        print(tabulate(datas, headers=["Metrics", "Value"], tablefmt="fancy_grid"))
        
        current_metrics_dict = {
            "total_steps": global_step,
            "log_entries": self.num_logs_shown + 1,
            "mean_reward_last_n_eps": mean_reward,
            # "mean_episode_length_approx": mean_ep_len_approx,
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
        
        # Reset for next logging interval
        self.episode_rewards = [] # Reset rewards for the next interval's calculation
        self.entropy_mean = []    # Reset policy entropy for next interval
        self.reset_epoch_metrics()
        self.last_log_time = time.time()
        self.global_step_at_last_log = global_step
        self.num_logs_shown += 1

# From SAC example
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, layer_init_fn=None):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(self.obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, self.action_dim)
        self.fc_logstd = nn.Linear(256, self.action_dim)

        if layer_init_fn: # Optional layer initialization
            self.fc1 = layer_init_fn(self.fc1)
            self.fc2 = layer_init_fn(self.fc2)
            self.fc_mean = layer_init_fn(self.fc_mean, std=0.01) # Smaller std for action output
            self.fc_logstd = layer_init_fn(self.fc_logstd, std=0.01)


        # Action rescaling buffers
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        if deterministic:
            x_t = mean # Use mean for deterministic action
        else:
            x_t = normal.rsample()  # For reparameterization trick (mean + std * N(0,1))
        
        y_t = torch.tanh(x_t) # Squash to [-1, 1]
        action = y_t * self.action_scale + self.action_bias # Rescale to action space
        
        # Calculate log_prob accounting for tanh squashing
        # log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # More stable version from SAC paper appendix C / CleanRL
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        log_prob = log_prob.sum(1, keepdim=True)
        
        # For evaluation, the mean action is also squashed and rescaled
        eval_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, eval_mean # Return squashed & rescaled action, its log_prob, and squashed & rescaled mean

class SoftQNetwork(nn.Module):
    def __init__(self, env, layer_init_fn=None):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if layer_init_fn: # Optional layer initialization
            self.fc1 = layer_init_fn(self.fc1)
            self.fc2 = layer_init_fn(self.fc2)
            self.fc3 = layer_init_fn(self.fc3, std=1.0) # Larger std for value output

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0): # Your PPO layer init
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Float32Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Modifier l'espace d'observation
        if isinstance(self.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=self.observation_space.low.astype(np.float32),
                high=self.observation_space.high.astype(np.float32),
                shape=self.observation_space.shape,
                dtype=np.float32
            )
        
        # Modifier l'espace d'action (important si les actions sont aussi stockées en float64)
        if isinstance(self.action_space, spaces.Box):
            self.action_space = spaces.Box(
                low=self.action_space.low.astype(np.float32),
                high=self.action_space.high.astype(np.float32),
                shape=self.action_space.shape,
                dtype=np.float32
            )

    def step(self, action):
        # L'action de la politique devrait déjà être np.float32 grâce aux corrections précédentes
        # Si l'environnement sous-jacent attend strictement float64, il faudrait caster : action.astype(np.float64)
        # Mais la plupart des environnements MuJoCo acceptent float32.
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # Assurer que l'observation retournée et la récompense sont float32/float
        return next_obs.astype(np.float32), float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        seed = kwargs.pop('seed', None) # Extraire la seed pour la passer explicitement
        if seed is not None:
            # La méthode reset de gym.Wrapper ne propage pas la seed par défaut à self.env.reset
            # Il faut appeler self.env.reset directement si la seed est fournie.
            # Cependant, gym.Wrapper.reset gère la seed pour self.action_space.seed() etc.
            # Pour la plupart des envs, passer la seed à reset est la clé.
            obs, info = self.env.reset(seed=seed, **kwargs)
        else:
            obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32), info

class SACTorch(nn.Module):
    def __init__(self, env_or_env_id: str | gym.Env, args: Args, 
                 actor_optimizer_cls=torch.optim.Adam, 
                 q_optimizer_cls=torch.optim.Adam,
                 alpha_optimizer_cls=torch.optim.Adam,
                 optimizer_params: dict = {}):
        super(SACTorch, self).__init__()
        
        self.args = args
        
        # Configuration du périphérique, incluant MPS
        self.device = torch.device('cpu')
        print(self.device)


        if args.seed is not None:
            print(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if self.device != torch.device("mps"): # Deterministic algorithms not fully supported on MPS
                 torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True) # Peut causer des problèmes


        assert isinstance(env_or_env_id, (str, gym.Env)), "env_or_env_id must be a string or a gym.Env instance."
        
        # Création et wrapping de l'environnement
        if isinstance(env_or_env_id, str):
            _env = gym.make(args.env_id)
        else:
            _env = env_or_env_id # C'est déjà une instance d'environnement
        
        self.env = Float32Wrapper(_env) # Appliquer le wrapper ici

        # Assurer que single_observation_space et single_action_space sont définis
        # Ces attributs sont utilisés par Actor, SoftQNetwork et ReplayBuffer
        if not hasattr(self.env, 'single_observation_space'):
            self.env.single_observation_space = self.env.observation_space
        if not hasattr(self.env, 'single_action_space'):
            self.env.single_action_space = self.env.action_space
        
        assert isinstance(self.env.action_space, gym.spaces.Box), "SAC requires continuous action space (gym.spaces.Box)."
        assert self.env.single_observation_space.dtype == np.float32, "Observation space dtype should be float32 after wrapping."
        if isinstance(self.env.single_action_space, spaces.Box): # Action space peut être autre chose
             assert self.env.single_action_space.dtype == np.float32, "Action space dtype should be float32 after wrapping for Box spaces."


        layer_initializer = None 

        self.actor = Actor(self.env, layer_initializer).to(self.device)
        self.qf1 = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf2 = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf1_target = SoftQNetwork(self.env, layer_initializer).to(self.device)
        self.qf2_target = SoftQNetwork(self.env, layer_initializer).to(self.device)
        
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor_optimizer = actor_optimizer_cls(self.actor.parameters(), **optimizer_params)
        self.q_optimizer = q_optimizer_cls(list(self.qf1.parameters()) + list(self.qf2.parameters()), **optimizer_params)

        self.alpha = args.alpha
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = alpha_optimizer_cls([self.log_alpha], lr=args.q_lr) 
            self.alpha = self.log_alpha.exp().item() 
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
        
        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            self.env.single_observation_space, # Devrait maintenant être float32
            self.env.single_action_space,      # Devrait maintenant être float32 pour Box
            self.device, # Important: Le buffer saura qu'il opère avec des tenseurs MPS
            handle_timeout_termination=False, 
        )
        if args.seed is not None:
            self.replay_buffer._rng = np.random.default_rng(args.seed)
        
        self.metrics = MetricsContainer(args.log_interval_steps)
        self.global_step = 0

    # ... (get_action, learn, train_step, test methods comme avant) ...
    # Les conversions .float() dans train_step sont toujours une bonne défense,
    # mais devraient être des no-ops si le wrapper fonctionne correctement.
    # La conversion dans get_action est également toujours correcte.

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device) 
        if len(obs_tensor.shape) == 1: 
             obs_tensor = obs_tensor.unsqueeze(0)
        action, _, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action.detach().cpu().numpy()[0]

    def learn(self, total_timesteps: int):
        # Passer la seed ici si elle est définie dans les args
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
                # Assurer que l'action échantillonnée est aussi float32 si l'action_space l'est
                if isinstance(self.env.action_space, spaces.Box) and self.env.action_space.dtype == np.float32:
                    action = action.astype(np.float32)
            else:
                action = self.get_action(obs) # Déjà np.float32

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            # next_obs est déjà float32 grâce au wrapper
            done = terminated or truncated
            current_episode_reward += reward
            
            # action est np.float32
            # obs et next_obs sont np.float32
            self.replay_buffer.add(obs, next_obs, action, reward, terminated, [info]) 

            obs = next_obs

            if done:
                self.metrics.episode_rewards.append(current_episode_reward)
                obs, _ = self.env.reset() # La seed n'est passée qu'au premier reset
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
        
        # Les données du buffer devraient être float32 si le wrapper et l'init du buffer sont corrects
        # Les conversions .float() ici sont une double sécurité
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
        # Utiliser une seed différente pour chaque épisode de test pour plus de robustesse
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
    
    def save(self, folder_path: str):
        """Saves the model, optimizers, and replay buffer to a dedicated folder."""
        os.makedirs(folder_path, exist_ok=True)

        model_path = os.path.join(folder_path, "model.pt")
        buffer_path = os.path.join(folder_path, "replay_buffer.pkl")
        metrics_container_path = os.path.join(folder_path, "metrics_container.pkl")

        torch.save({
            'global_step': self.global_step,
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            'args': self.args,
        }, model_path)
        print(f"Agent checkpoint saved to {model_path}")

        with open(buffer_path, "wb") as f:
            pickle.dump(self.replay_buffer.__dict__, f)
        print(f"Replay buffer saved to {buffer_path}")
        
        with open(metrics_container_path, "wb") as f:
            pickle.dump(self.metrics, f)
        print(f"Metrics container saved to {metrics_container_path}")


    @staticmethod
    def load(folder_path: str, optimizer_cls: torch.optim.Optimizer, ignore_optimizers: bool = False):
        """Loads the agent and replay buffer from a folder."""
        model_path = os.path.join(folder_path, "model.pt")
        buffer_path = os.path.join(folder_path, "replay_buffer.pkl")
        metrics_container_path = os.path.join(folder_path, "metrics_container.pkl")

        print(f"Loading agent checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        args_to_use = checkpoint['args']
        sac_agent = SACTorch(
            args_to_use.env_id,
            args_to_use,
            actor_optimizer_cls=optimizer_cls,
            q_optimizer_cls=optimizer_cls,
            alpha_optimizer_cls=optimizer_cls
        )

        sac_agent.global_step = checkpoint['global_step']
        sac_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        sac_agent.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        sac_agent.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        sac_agent.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
        sac_agent.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])

        sac_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        sac_agent.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        if checkpoint['alpha_optimizer_state_dict'] is not None and sac_agent.alpha_optimizer:
            sac_agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            sac_agent.alpha = sac_agent.log_alpha.exp().item()
                    
        if os.path.exists(buffer_path):
            print(f"Loading replay buffer from {buffer_path}...")
            with open(buffer_path, "rb") as f:
                replay_data = pickle.load(f)
                sac_agent.replay_buffer.__dict__.update(replay_data)
            print("Replay buffer loaded successfully.")
        else:
            print(f"⚠️ Replay buffer not found at {buffer_path}. Starting with an empty buffer.")

        # Chargement des métriques depuis le JSON associé
        if os.path.exists(metrics_container_path):
            print(f"Loading metrics container from {metrics_container_path}...")
            with open(metrics_container_path, "rb") as f:
                sac_agent.metrics = pickle.load(f)
        else:
            print(f"⚠️ Metrics container not found at {metrics_container_path}. Starting with an empty metrics container.")
        
        return sac_agent


if __name__ == '__main__':
    from optimizers import * # Your custom optimizers
    # For SAC, optimizers are typically Adam. If you want to test custom ones,
    # you'll need to pass them to SACTorch for actor, q_functions, and alpha.
    # For simplicity, the SACTorch class defaults to Adam.
    # Here's how you might pass them if you adapt SACTorch to take optimizer classes:
    # Example: optimizer_to_test = [(torch.optim.Adam, "Adam"), (YourCustomOpt, "CustomOpt")]

    total_training_steps = 100_000

    optimizers_to_test = [#(torch.optim.Adam,'AdamW'),
                          (Ano, 'Ano'),
                          ]
    lr = [3e-4, 3e-5, 3e-3]
    beta1 = [0.9, 0.92, 0.95]
    beta2 = [0.99, 0.995, 0.999]
    optimizers = []

    """beta3 = [0.92, 0.94, 0.98]
    opt_class = Adan
    for learning_rate in lr:
            for b1 in beta1:
                for b3 in beta3:
                    opt_name = f"{opt_class.__name__}_lr{learning_rate}_b1{b1}_b2{b3}"
                    opt_params = {'lr': learning_rate, 'betas': (b1, 0.92, b3)}
                    optimizers.append((opt_class, opt_name, opt_params))"""
                    
    for opt_class, opt_name in optimizers_to_test:
        for learning_rate in lr:
            for b1 in beta1:
                for b2 in beta2:
                    opt_name_file = f"{opt_name}_lr{learning_rate}_b1{b1}_b2{b2}"
                    opt_params = {'lr': learning_rate, 'betas': (b1, b2)}
                    optimizers.append((opt_class, opt_name_file, opt_params))
    
    
                    
    env_id_to_test = 'HalfCheetah-v5'

    seeds_to_run = [42, 512]

    for seed in seeds_to_run:
        for optimizer_cls, optimizer_name, optim_params in optimizers:
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
            save_logs(args.run_name, history, folder=f'experiments/hyperparameters_tuning/logs/halfcheetah')
            
    print("All SAC tests done")