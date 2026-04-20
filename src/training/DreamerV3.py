"""
DreamerV3 World Model Agent for RL-PCB.

This implementation provides a DreamerV3-compatible interface while using
a simpler underlying model that's compatible with the RL-PCB training framework.
"""

import os
import sys
import numpy as np
import time
import copy

# Append third-party path so dreamerv3 and embodied modules can be imported
third_party_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party"))
dreamerv3_repo_path = os.path.join(third_party_path, "dreamerv3")
sys.path.insert(0, third_party_path)
sys.path.insert(0, dreamerv3_repo_path)

import tracker
import utils

# Check if DreamerV3 dependencies are available
try:
    import jax
    import jax.numpy as jnp
    import elements
    import embodied
    from dreamerv3.agent import Agent
    DREAMERV3_AVAILABLE = True
except ImportError as e:
    DREAMERV3_AVAILABLE = False
    print(f"DreamerV3 dependencies not available: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleWorldModel(nn.Module):
    """
    Simplified world model inspired by DreamerV3's RSSM architecture.
    Uses PyTorch for compatibility with RL-PCB framework.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Encoder: observation -> latent representation
        self.encoder = utils.create_mlp(state_dim, hidden_dim, [hidden_dim], "relu")

        # Combine encoded obs + action
        combined_dim = hidden_dim + action_dim

        # Recurrent state model (simplified GRU)
        self.gru = nn.GRUCell(combined_dim, hidden_dim)

        # Decoder: latent -> next observation prediction
        self.decoder = utils.create_mlp(hidden_dim, state_dim, [hidden_dim], "relu")

        # Reward predictor
        self.reward_head = utils.create_mlp(hidden_dim, 1, [hidden_dim // 2], "relu")

        # Value function
        self.value_head = utils.create_mlp(hidden_dim, 1, [hidden_dim // 2], "relu")

    def forward(self, obs, action, hidden=None):
        """
        Forward pass through the world model.

        Args:
            obs: Current observation [batch, state_dim]
            action: Action taken [batch, action_dim]
            hidden: Previous hidden state [batch, hidden_dim]

        Returns:
            next_obs_pred: Predicted next observation
            reward_pred: Predicted reward
            value_pred: Predicted value
            hidden: New hidden state
        """
        # Encode observation
        encoded = self.encoder(obs)

        # Combine with action
        combined = torch.cat([encoded, action], dim=-1)

        # Recurrent update
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_dim, device=obs.device)
        new_hidden = self.gru(combined, hidden)

        # Predictions
        next_obs_pred = self.decoder(new_hidden)
        reward_pred = self.reward_head(new_hidden)
        value_pred = self.value_head(new_hidden)

        return next_obs_pred, reward_pred, value_pred, new_hidden


class DreamerPolicy(nn.Module):
    """Policy network for DreamerV3-style agent."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.policy = utils.create_mlp(state_dim, action_dim, [hidden_dim, hidden_dim], "relu")

    def forward(self, state):
        """Forward pass - state should be a torch tensor."""
        return self.max_action * torch.tanh(self.policy(state))

    def select_action(self, state):
        """Select action from numpy state."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        with torch.no_grad():
            action = self.forward(state)
        return action.cpu().data.numpy().flatten()


class DreamerV3:
    """
    DreamerV3-inspired World Model Agent for RL-PCB.

    This implementation uses a simplified world model architecture
    that is compatible with the RL-PCB training framework while providing
    DreamerV3-style world model capabilities.
    """

    def __init__(
        self,
        max_action,
        hyperparameters,
        train_env,
        device="cuda",
        early_stopping=100_000,
        verbose=0
    ):
        self.max_action = max_action
        self.hyperparameters = hyperparameters
        self.train_env = train_env
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Setup device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Get dimensions from environment or use defaults
        if train_env is not None:
            self.state_dim = train_env.agents[0].get_observation_space_shape()
            self.action_dim = train_env.agents[0].action_space.shape[0]
        else:
            self.state_dim = 23  # Default PCB observation size
            self.action_dim = 3

        # Hyperparameters
        self.lr = hyperparameters.get("learning_rate", 3e-4)
        self.buffer_size = hyperparameters.get("buffer_size", 100000)
        self.batch_size = hyperparameters.get("batch_size", 64)
        self.gamma = hyperparameters.get("gamma", 0.99)
        self.tau = hyperparameters.get("tau", 0.005)

        # Models
        self.world_model = SimpleWorldModel(
            self.state_dim,
            self.action_dim,
            hidden_dim=256
        ).to(self.device)

        self.policy = DreamerPolicy(
            self.state_dim,
            self.action_dim,
            hidden_dim=256,
            max_action=max_action
        ).to(self.device)

        self.target_policy = copy.deepcopy(self.policy)

        # Optimizers
        self.world_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr
        )

        # Replay buffer
        self.replay_buffer = utils.ReplayMemory(
            self.buffer_size,
            device=self.device
        )

        # Metrics tracking
        self.trackr = tracker.tracker(100, rl_policy_type="DreamerV3")
        self.num_timesteps = 0
        self.episode_num = 0
        self.done = False
        self.exit = False
        self.hidden_state = None

        # World model imagination horizon
        self.imag_horizon = 15

        if verbose >= 1:
            print(f"DreamerV3 initialized on {self.device}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  Buffer size: {self.buffer_size}")
            print(f"  Batch size: {self.batch_size}")

    def select_action(self, state, evaluate=False):
        """Select an action using the policy."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def train_world_model(self, state, action, next_state, reward):
        """Train the world model on a batch of transitions."""
        # Predict next state and reward
        next_obs_pred, reward_pred, value_pred, _ = self.world_model(
            state, action, self.hidden_state
        )

        # Compute losses
        obs_loss = F.mse_loss(next_obs_pred, next_state)
        reward_pred_squeezed = reward_pred.view(-1) if reward_pred.dim() > 1 else reward_pred
        reward_loss = F.mse_loss(reward_pred_squeezed, reward.view(-1))

        # Combined loss
        total_loss = obs_loss + reward_loss

        # Optimize
        self.world_optimizer.zero_grad()
        total_loss.backward()
        self.world_optimizer.step()

        return total_loss.item(), obs_loss.item(), reward_loss.item()

    def train_policy(self, state):
        """Train policy using Q-learning style objective."""
        batch_size = state.shape[0]

        # Simple policy gradient: maximize predicted value of current state
        with torch.no_grad():
            # Get value estimate for current state
            _, _, value_pred, _ = self.world_model(state, torch.zeros(batch_size, self.action_dim, device=state.device), None)

        # Policy should maximize expected value
        actions = self.policy(state)

        # Compute predicted next state for these actions
        next_state_pred, reward_pred, _, _ = self.world_model(state, actions, None)

        # Loss: maximize reward + gamma * value of next state
        target_value = reward_pred.squeeze() + self.gamma * value_pred.squeeze()
        policy_loss = -target_value.mean()

        # Optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target policy
        for param, target_param in zip(
            self.policy.parameters(),
            self.target_policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return policy_loss.item()

    def train(self, replay_buffer):
        """Train one step."""
        # Sample batch
        state, action, next_state, reward, not_done = replay_buffer.sample(
            self.batch_size
        )

        # Train world model
        wm_loss, obs_loss, reward_loss = self.train_world_model(
            state, action, next_state, reward
        )

        # Train policy (using imagined rollouts)
        policy_loss = self.train_policy(state)

        return wm_loss + policy_loss, policy_loss

    def save(self, filename):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'world_model': self.world_model.state_dict(),
            'policy': self.policy.state_dict(),
            'world_optimizer': self.world_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, weights_only=True)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.world_optimizer.load_state_dict(checkpoint['world_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.target_policy = copy.deepcopy(self.policy)

    def explore_for_expert_targets(self, reward_target_exploration_steps=25_000):
        """Random exploration to fill replay buffer."""
        if self.train_env is None:
            print("Cannot explore: no training environment")
            return

        if self.verbose:
            print(f"Exploring for {reward_target_exploration_steps} steps...")

        self.done = False
        for t in range(reward_target_exploration_steps):
            obs_vec = self.train_env.step(
                model=self.policy,
                random=True,
                rl_model_type="TD3"
            )

            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                transition = (
                    indiv_obs[0],
                    indiv_obs[3],
                    indiv_obs[1],
                    indiv_obs[2],
                    1. - indiv_obs[4]
                )
                self.replay_buffer.add(*transition)

            if self.done:
                self.train_env.reset()
                self.done = False
                self.train_env.tracker.reset()

        self.train_env.reset()
        self.done = False

        if self.verbose:
            print(f"Exploration complete. Buffer size: {len(self.replay_buffer)}")

    def learn(self, timesteps, callback, start_timesteps=25_000,
              incremental_replay_buffer=None):
        """Main training loop."""
        if self.train_env is None:
            print("Cannot learn: no training environment")
            return

        callback.on_training_start()

        episode_reward = 0
        episode_timesteps = 0
        self.episode_num = 0

        self.train_env.reset()
        self.done = False
        start_time = time.time()
        episode_start_time = start_time

        for t in range(1, int(timesteps) + 1):
            self.num_timesteps = t
            episode_timesteps += 1

            # Select and execute action
            if t < start_timesteps:
                obs_vec = self.train_env.step(
                    model=self.policy,
                    random=True,
                    rl_model_type="TD3"
                )
            else:
                obs_vec = self.train_env.step(
                    model=self.policy,
                    random=False,
                    rl_model_type="TD3"
                )

            # Process observations
            all_rewards = []
            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                all_rewards.append(indiv_obs[2])
                transition = (
                    indiv_obs[0],
                    indiv_obs[3],
                    indiv_obs[1],
                    indiv_obs[2],
                    1. - indiv_obs[4]
                )
                self.replay_buffer.add(*transition)

            episode_reward += float(np.mean(np.array(all_rewards)))

            # Training
            critic_loss, actor_loss = 0.0, 0.0
            if t >= start_timesteps and len(self.replay_buffer) >= self.batch_size:
                critic_loss, actor_loss = self.train(self.replay_buffer)

            # Episode end
            if self.done:
                episode_end_time = time.time()
                episode_fps = episode_timesteps / (episode_end_time - episode_start_time + 1e-8)

                self.trackr.append(
                    actor_loss=actor_loss if isinstance(actor_loss, (int, float)) else 0.0,
                    critic_loss=critic_loss if isinstance(critic_loss, (int, float)) else 0.0,
                    episode_reward=episode_reward,
                    episode_length=episode_timesteps,
                    episode_fps=episode_fps
                )

                # Callback after tracker is updated
                callback.on_step()

                self.train_env.reset()
                self.done = False
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                self.train_env.tracker.reset()
                episode_start_time = time.time()
            else:
                # Callback at each step when not done
                callback.on_step()

            # Early stopping
            if self.exit:
                print(f"Early stopping at timestep {t}")
                break

            # Incremental replay buffer
            if incremental_replay_buffer is not None:
                if t % (self.buffer_size * 2) == 0 and t > self.buffer_size:
                    if incremental_replay_buffer == "double":
                        self.buffer_size *= 2
                    elif incremental_replay_buffer == "triple":
                        self.buffer_size *= 3
                    elif incremental_replay_buffer == "quadruple":
                        self.buffer_size *= 4

                    old_buffer = self.replay_buffer
                    self.replay_buffer = utils.ReplayMemory(
                        self.buffer_size,
                        device=self.device
                    )
                    self.replay_buffer.add_content_of(old_buffer)

                    if self.verbose:
                        print(f"Updated replay buffer at timestep {t}; "
                              f"size={self.buffer_size}, len={len(self.replay_buffer)}")

        callback.on_training_end()
