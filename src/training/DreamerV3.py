"""
DreamerV3 World Model Agent for RL-PCB using the real JAX-based implementation.

This wraps the official DreamerV3 from third_party/dreamerv3 to work with
RL-PCB's training framework.
"""

import os
import sys
import numpy as np
import time
import copy
from typing import Dict, Any
import jax

# Append third-party path so dreamerv3 and embodied modules can be imported
third_party_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party"))
dreamerv3_repo_path = os.path.join(third_party_path, "dreamerv3")
sys.path.insert(0, third_party_path)
sys.path.insert(0, dreamerv3_repo_path)

import tracker
import utils

# Import DreamerV3 dependencies
try:
    import elements
    import embodied
    from dreamerv3.agent import Agent as DreamerV3Agent
    from embodied.jax.agent import Agent as JAXAgent
    DREAMERV3_AVAILABLE = True
except ImportError as e:
    print(f"Error importing DreamerV3 dependencies: {e}")
    DREAMERV3_AVAILABLE = False
    raise


class DreamerV3:
    """
    DreamerV3 Agent for RL-PCB using the official JAX-based implementation.

    This wraps the real DreamerV3 algorithm with RSSM world model,
    categorical latents, symlog encoding, and imagination-based policy.
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
        if not DREAMERV3_AVAILABLE:
            raise ImportError("DreamerV3 dependencies not available. Please install JAX and related packages.")

        self.max_action = max_action
        self.hyperparameters = hyperparameters
        self.train_env = train_env
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Setup device (DreamerV3 uses JAX, so this is mainly for compatibility)
        self.device_str = device if device == "cuda" and self._check_cuda() else "cpu"
        self.device = self.device_str

        # Get dimensions from environment
        if train_env is not None:
            self.state_dim = train_env.agents[0].get_observation_space_shape()
            self.action_dim = train_env.agents[0].action_space.shape[0]
        else:
            self.state_dim = 23
            self.action_dim = 3

        # Hyperparameters with DreamerV3-appropriate defaults
        self.lr = hyperparameters.get("learning_rate", 1e-4)  # Lower LR for DreamerV3
        self.train_batch_size = hyperparameters.get("batch_size", 16)
        self.batch_length = hyperparameters.get("batch_length", 64)  # Sequence length
        self.gamma = hyperparameters.get("gamma", 0.99)

        # Create DreamerV3 config
        self.config = self._create_config()

        # Create observation and action spaces for DreamerV3
        self.obs_space = self._create_obs_space()
        self.act_space = self._create_act_space()

        # Initialize the DreamerV3 agent
        self.agent = self._create_agent()

        # Initialize policy carry state (for stateful inference)
        self.policy_carry = None
        
        # Initialize the policy carry (inference batch size is 1)
        self.policy_carry = self.agent.init_policy(1)

        # Replay buffer (DreamerV3-style with sequences)
        self.replay_buffer = utils.SequenceReplayBuffer(
            capacity=hyperparameters.get("buffer_size", 1_000_000),
            sequence_length=self.batch_length,
            device="cpu"
        )

        # Episode buffer for collecting transitions during rollout
        self.episode_buffer = utils.EpisodeBuffer()

        # Metrics tracking
        self.trackr = tracker.tracker(100, rl_policy_type="DreamerV3")
        self.num_timesteps = 0
        self.episode_num = 0
        self.done = False
        self.exit = False

        # JAX random seed for training
        self.jax_seed = jax.random.PRNGKey(hyperparameters.get("seed", 0))

        # Training state
        self.train_carry = None
        self.should_train = False
        self.train_steps = 0

        # Pre-allocate data structure for efficiency
        self._data_buffer = None

        if verbose >= 1:
            print(f"DreamerV3 (JAX) initialized on {self.device_str}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  Train batch size: {self.train_batch_size}")
            print(f"  Batch length: {self.batch_length}")

    def _check_cuda(self):
        """Check if CUDA is available for JAX."""
        try:
            import jax
            return len(jax.devices('gpu')) > 0
        except:
            return False

    def _create_config(self):
        """Create DreamerV3 configuration."""
        # Default DreamerV3 config - must be flat structure as expected by agent
        config_dict = {
            'seed': 0,
            'logdir': '/tmp/dreamerv3',
            'batch_size': self.train_batch_size,
            'batch_length': self.batch_length,
            'report_length': 32,
            'replay_context': 0,
            'random_agent': False,
            'jax': {
                'platform': 'cpu',
                'compute_dtype': 'float32',  # Use float32 for CPU
                'policy_devices': (0,),
                'train_devices': (0,),
                'mock_devices': 0,
                'prealloc': True,
                'jit': True,
                'debug': False,
                'expect_devices': 0,
                'enable_policy': True,
                'coordinator_address': '',
            },
            # Loss scales
            'loss_scales': {
                'rec': 1.0, 'rew': 1.0, 'con': 1.0, 'dyn': 1.0, 'rep': 0.1,
                'policy': 1.0, 'value': 1.0, 'repval': 0.3
            },
            # Optimizer
            'opt': {
                'lr': self.lr,
                'agc': 0.3,
                'eps': 1e-20,
                'beta1': 0.9,
                'beta2': 0.999,
                'momentum': True,
                'wd': 0.0,
                'schedule': 'const',
                'warmup': 1000,
                'anneal': 0,
            },
            'ac_grads': False,
            'reward_grad': True,
            'repval_loss': True,
            'repval_grad': True,
            'report': True,
            'report_gradnorms': False,
            # Dynamics (RSSM)
            'dyn': {
                'typ': 'rssm',
                'rssm': {
                    'deter': 256, 'hidden': 256, 'stoch': 32, 'classes': 32,
                    'act': 'silu', 'norm': 'rms', 'unimix': 0.01, 'outscale': 1.0,
                    'winit': 'trunc_normal_in', 'imglayers': 2, 'obslayers': 1,
                    'dynlayers': 1, 'absolute': False, 'blocks': 8,
                    'free_nats': 1.0
                }
            },
            # Encoder
            'enc': {
                'typ': 'simple',
                'simple': {
                    'depth': 32, 'mults': [1, 1], 'layers': 2, 'units': 256,
                    'act': 'silu', 'norm': 'rms', 'winit': 'trunc_normal_in',
                    'symlog': True, 'outer': False, 'kernel': 5, 'strided': False
                }
            },
            # Decoder
            'dec': {
                'typ': 'simple',
                'simple': {
                    'depth': 32, 'mults': [1, 1], 'layers': 2, 'units': 256,
                    'act': 'silu', 'norm': 'rms', 'outscale': 1.0,
                    'winit': 'trunc_normal_in', 'outer': False, 'kernel': 5,
                    'bspace': 8, 'strided': False
                }
            },
            # Heads
            'rewhead': {
                'layers': 2, 'units': 256, 'act': 'silu', 'norm': 'rms',
                'output': 'symexp_twohot', 'outscale': 0.0,
                'winit': 'trunc_normal_in', 'bins': 255
            },
            'conhead': {
                'layers': 2, 'units': 256, 'act': 'silu', 'norm': 'rms',
                'output': 'binary', 'outscale': 1.0, 'winit': 'trunc_normal_in'
            },
            'policy': {
                'layers': 3, 'units': 256, 'act': 'silu', 'norm': 'rms',
                'minstd': 0.1, 'maxstd': 1.0, 'outscale': 0.01,
                'unimix': 0.01, 'winit': 'trunc_normal_in'
            },
            'value': {
                'layers': 3, 'units': 256, 'act': 'silu', 'norm': 'rms',
                'output': 'symexp_twohot', 'outscale': 0.0,
                'winit': 'trunc_normal_in', 'bins': 255
            },
            'policy_dist_disc': 'categorical',
            'policy_dist_cont': 'bounded_normal',
            'imag_last': 0,
            'imag_length': 15,
            'horizon': 333,
            'contdisc': True,
            'imag_loss': {'slowtar': False, 'lam': 0.95, 'actent': 3e-4, 'slowreg': 1.0},
            'repl_loss': {'slowtar': False, 'lam': 0.95, 'slowreg': 1.0},
            'slowvalue': {'rate': 0.02, 'every': 1},
            'retnorm': {'impl': 'perc', 'rate': 0.01, 'limit': 1.0, 'perclo': 5.0, 'perchi': 95.0, 'debias': False},
            'valnorm': {'impl': 'none', 'rate': 0.01, 'limit': 0.00000001},
            'advnorm': {'impl': 'none', 'rate': 0.01, 'limit': 0.00000001},
        }
        return elements.Config(config_dict)

    def _create_obs_space(self):
        """Create observation space for DreamerV3."""
        spaces = {}
        # Vector observation
        spaces['vector'] = elements.Space(np.float32, (self.state_dim,))
        # Required metadata
        spaces['is_first'] = elements.Space(bool, ())
        spaces['is_last'] = elements.Space(bool, ())
        spaces['is_terminal'] = elements.Space(bool, ())
        spaces['reward'] = elements.Space(np.float32, ())
        return spaces

    def _create_act_space(self):
        """Create action space for DreamerV3."""
        spaces = {}
        # Continuous action - DreamerV3 expects 'action' key
        # Do NOT include 'reset' - DreamerV3 handles resets differently
        spaces['action'] = elements.Space(np.float32, (self.action_dim,), -1.0, 1.0)
        return spaces

    def _create_agent(self):
        """Create the DreamerV3 agent."""
        agent = DreamerV3Agent(self.obs_space, self.act_space, self.config)
        return agent

    def _init_policy_carry(self):
        """Initialize policy carry state for a batch of 1."""
        self.policy_carry = self.agent.init_policy(1)

    def _reset_policy_carry(self):
        """Reset policy carry for a new episode."""
        self.policy_carry = self.agent.init_policy(1)

    def _convert_obs_to_dreamer(self, obs, is_first=False, is_last=False, reward=0.0):
        """Convert RL-PCB observation to DreamerV3 format."""
        # Add batch dimension (1,) to match expected shapes
        return {
            'vector': np.asarray(obs, dtype=np.float32).reshape(1, -1),  # [1, state_dim]
            'is_first': np.array([is_first], dtype=bool),  # [1]
            'is_last': np.array([is_last], dtype=bool),  # [1]
            'is_terminal': np.array([is_last], dtype=bool),  # [1] - Terminal = episode end
            'reward': np.array([reward], dtype=np.float32),  # [1]
        }

    def _convert_action_from_dreamer(self, action_dict):
        """Convert DreamerV3 action to RL-PCB format."""
        action = action_dict['action']
        # Take first batch element and scale to max_action
        # action shape is [batch, action_dim]
        return action[0] * self.max_action

    def _convert_action_to_dreamer(self, action):
        """Convert RL-PCB action to DreamerV3 format."""
        # Normalize to [-1, 1]
        normalized = action / self.max_action
        return {'action': np.clip(normalized, -1.0, 1.0)}

    def select_action(self, state, evaluate=False):
        """Select an action using the DreamerV3 policy."""
        # Convert observation
        obs = self._convert_obs_to_dreamer(state, is_first=False)

        # Select mode
        mode = 'eval' if evaluate else 'train'

        # Get action from DreamerV3
        self.policy_carry, action_dict, _ = self.agent.policy(
            self.policy_carry, obs, mode=mode
        )

        # Convert action to RL-PCB format
        action = self._convert_action_from_dreamer(action_dict)
        return action.flatten()

    def train(self, replay_buffer):
        """Train one step (not used with DreamerV3 - training is done differently)."""
        # DreamerV3 uses sequence-based training, handled in learn()
        return 0.0, 0.0

    def save(self, filename):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = self.agent.save()
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        """Load model checkpoint."""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.agent.load(data)
        # Re-initialize policy carry
        self._init_policy_carry()

    def explore_for_expert_targets(self, reward_target_exploration_steps=25_000):
        """Random exploration to fill replay buffer."""
        if self.train_env is None:
            print("Cannot explore: no training environment")
            return

        if self.verbose:
            print(f"Exploring for {reward_target_exploration_steps} steps...")

        self.done = False
        for t in range(int(reward_target_exploration_steps)):
            obs_vec = self.train_env.step(
                model=self,
                random=True,
                rl_model_type="TD3"  # Use TD3 format for compatibility
            )

            for indiv_obs in obs_vec:
                state, next_state, reward, action, done = indiv_obs[0], indiv_obs[1], indiv_obs[2], indiv_obs[3], indiv_obs[4]
                if done is True:
                    self.done = True
                
                # Add to episode buffer
                self.episode_buffer.add(state, action, reward, next_state, done)

            if self.done:
                # Add completed episode to replay buffer
                self.replay_buffer.add_episode(self.episode_buffer.get_episode())
                self.episode_buffer.reset()
                
                self.train_env.reset()
                self.done = False
                self.train_env.tracker.reset()

        self.train_env.reset()
        self.done = False

        if self.verbose:
            print(f"Exploration complete. Buffer size: {len(self.replay_buffer)}")

    def learn(self, timesteps, callback, start_timesteps=25_000,
              incremental_replay_buffer=None):
        """Main training loop for DreamerV3."""
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

        # Reset policy carry for new episode
        self._reset_policy_carry()

        for t in range(1, int(timesteps) + 1):
            self.num_timesteps = t
            episode_timesteps += 1

            # Select and execute action
            if t < start_timesteps:
                obs_vec = self.train_env.step(
                    model=self,
                    random=True,
                    rl_model_type="TD3"
                )
            else:
                obs_vec = self.train_env.step(
                    model=self,
                    random=False,
                    rl_model_type="TD3"
                )

            # Process observations
            all_rewards = []
            for indiv_obs in obs_vec:
                state, next_state, reward, action, done = indiv_obs[0], indiv_obs[1], indiv_obs[2], indiv_obs[3], indiv_obs[4]
                if done is True:
                    self.done = True
                all_rewards.append(reward)
                
                # Add to episode buffer
                self.episode_buffer.add(state, action, reward, next_state, done)

            episode_reward += float(np.mean(np.array(all_rewards)))

            # Training - simplified version
            critic_loss, actor_loss = 0.0, 0.0
            if t >= start_timesteps and len(self.replay_buffer) >= self.train_batch_size:
                # TODO: Implement proper DreamerV3 sequence training
                # For now, just update periodically
                if t % 100 == 0:  # Train every 100 steps
                    critic_loss, actor_loss = self._train_step()

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

                # Add completed episode to replay buffer
                self.replay_buffer.add_episode(self.episode_buffer.get_episode())
                self.episode_buffer.reset()

                self.train_env.reset()
                self.done = False
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                self.train_env.tracker.reset()
                episode_start_time = time.time()

                # Reset policy carry for new episode
                self._reset_policy_carry()
            else:
                # Callback at each step when not done
                callback.on_step()

            # Early stopping
            if self.exit:
                print(f"Early stopping at timestep {t}")
                break

            # Incremental replay buffer
            if incremental_replay_buffer is not None:
                    # SequenceReplayBuffer handles capacity internally, 
                    # but we can print a message if capacity was supposedly increased
                    if self.verbose:
                        print(f"Incremental buffer update triggered at timestep {t}")

        callback.on_training_end()

    def _train_step(self):
        """Perform one training step with DreamerV3."""
        if len(self.replay_buffer) < self.train_batch_size:
            return 0.0, 0.0

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.train_batch_size)

        # Update JAX seed
        self.jax_seed, step_seed = jax.random.split(self.jax_seed)

        # Map buffer keys to Agent keys
        # Agent expects: vector, is_first, is_last, is_terminal, reward, action, stepid, consec, seed
        data = {
            'vector': batch['states'],
            'action': batch['actions'],
            'reward': batch['rewards'],
            'is_first': batch['is_first'],
            'is_last': batch['is_last'],
            'is_terminal': batch['dones'],  # dones in buffer are terminal flags
            'stepid': np.zeros((self.train_batch_size, self.batch_length, 20), dtype=np.uint8),
            'consec': np.tile(np.arange(self.batch_length), (self.train_batch_size, 1)).astype(np.int32),
            'seed': step_seed,
        }

        # Initialize train carry if needed
        if self.train_carry is None:
            self.train_carry = self.agent.init_train(self.train_batch_size)

        # Perform training step
        self.train_carry, outs, metrics = self.agent.train(self.train_carry, data)

        # Extract losses for tracking
        policy_loss = metrics.get('loss/policy', 0.0)
        value_loss = metrics.get('loss/value', 0.0)
        
        # Track world model losses (reward and continuity)
        model_loss = metrics.get('loss/rew', 0.0) + metrics.get('loss/con', 0.0)
        
        # Return policy loss and combined model/value loss as critic loss
        return float(policy_loss), float(value_loss + model_loss)
