import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.agent.observation import get_agent_observation, flatten_observation


class DummyModel:
    """Dummy model that returns a pre-set action."""
    def __init__(self):
        self.action = None

    def select_action(self, state, evaluate=True):
        return self.action


class PcbGymWrapper(gym.Env):
    """
    A unified Gymnasium Environment for RL_PCB that controls components sequentially as a single agent.
    To be used by Gym-expecting algorithms like DreamerV3.
    """
    def __init__(self, pcb_env):
        super().__init__()
        self.pcb_env = pcb_env
        self.current_agent_idx = 0
        self.dummy_model = DummyModel()

        # Action space: [-1, 1] scaled internally
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: flattened observation shape
        obs_shape = 23  # 8 los + 8 ol + 2 dom + 2 euc_dist + 2 pos + 1 orientation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.pcb_env.parameters.seed = seed
             self.pcb_env.rng = np.random.default_rng(seed=seed)

        self.pcb_env.reset()
        self.current_agent_idx = 0
        agent = self.pcb_env.agents[self.current_agent_idx]
        obs = get_agent_observation(agent.parameters)
        return np.array(flatten_observation(obs), dtype=np.float32), {}

    def step(self, action):
        self.dummy_model.action = action
        agent = self.pcb_env.agents[self.current_agent_idx]

        state, next_state, reward, model_action, done = agent.step(
            model=self.dummy_model,
            random=False,
            deterministic=True,  # Prevent internal noise, RL algorithm handles exploration
            rl_model_type="TD3"  # TD3 mode expects actions in [-1, 1]
        )

        self.current_agent_idx += 1

        # Process global steps if we've cycled through all agents
        if self.current_agent_idx >= len(self.pcb_env.agents):
            self.current_agent_idx = 0
            self.pcb_env.env_steps += 1
            if self.pcb_env.parameters.debug is True and (self.pcb_env.env_steps % 10 == 0 or done is True):
                self.pcb_env._render_debug_frame()

        next_agent = self.pcb_env.agents[self.current_agent_idx]
        next_obs = get_agent_observation(next_agent.parameters)

        # In a generic sequential formulation, if any agent hits done (like timeout), the whole env is done.
        return np.array(flatten_observation(next_obs), dtype=np.float32), float(reward), done, False, {}
