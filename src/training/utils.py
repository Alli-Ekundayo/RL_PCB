import torch
import numpy as np
from collections import namedtuple
import random

def create_mlp(input_dim, output_dim, architecture, activation_fn_name, intermediate_activations=True):
    """Creates a PyTorch MLP wrapped in nn.Sequential."""
    if activation_fn_name == "relu":
        activation_fn = torch.nn.ReLU
    elif activation_fn_name == "tanh":
        activation_fn = torch.nn.Tanh
    else:
        activation_fn = torch.nn.ReLU
        
    layers = []
    in_size = input_dim
    for layer_sz in architecture:
        layers.append(torch.nn.Linear(in_size, layer_sz))
        if intermediate_activations:
            layers.append(activation_fn())
        in_size = layer_sz
        
    if output_dim is not None:
        layers.append(torch.nn.Linear(in_size, output_dim))
        
    return torch.nn.Sequential(*layers)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
            )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """
    A replay memory buffer used in reinforcement learning algorithms to store\
          and sample transitions.

    Args:
        capacity (int): The maximum capacity of the replay memory.
        device (str): The device to store the tensors (e.g., 'cpu', 'cuda').

    Attributes:
        capacity (int): The maximum capacity of the replay memory.
        device (str): The device to store the tensors.
        memory (list): A list to store the transitions.
        position (int): The current position in the memory buffer.

    Methods:
        add(*args): Saves a transition to the replay memory.
        add_content_of(other): Adds the content of another replay buffer to\
              this replay buffer.
        get_latest(latest): Returns the latest elements from the replay memory.
        add_latest_from(other, latest): Adds the latest samples from another\
             buffer to this buffer.
        shuffle(): Shuffles the transitions in the replay memory.
        sample(batch_size): Samples a batch of transitions from the replay\
              memory.
        sample_from_latest(batch_size, latest): Samples a batch of transitions\
              from the latest elements.
        __len__(): Returns the number of transitions stored in the replay\
              memory.
        reset(): Resets the replay memory by clearing all stored transitions.
    """

    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        reshaped_args = []
        for arg in args:
            reshaped_args.append(np.reshape(arg, (1, -1)))

        self.memory[self.position] = Transition(*reshaped_args)
        self.position = (self.position + 1) % self.capacity

    def add_content_of(self, other):
        """
        Adds the content of another replay buffer to this replay buffer.

        Args:
            other (ReplayMemory): Another replay buffer.
        """
        latest_trans = other.get_latest(self.capacity)
        for transition in latest_trans:
            self.add(*transition)

    def get_latest(self, latest):
        """
        Returns the latest elements from the replay memory.

        Args:
            latest (int): The number of latest elements to return.

        Returns:
            list: A list containing the latest elements.
        """
        if self.capacity < latest:
            latest_trans = self.memory[self.position:].copy() + self.memory[:self.position].copy()
        elif len(self.memory) < self.capacity:
            latest_trans = self.memory[-latest:].copy()
        elif self.position >= latest:
            latest_trans = self.memory[:self.position][-latest:].copy()
        else:
            latest_trans = self.memory[-latest+self.position:].copy() + self.memory[:self.position].copy()
        return latest_trans

    def add_latest_from(self, other, latest):
        """
        Adds the latest samples from another buffer to this buffer.

        Args:
            other (ReplayMemory): Another replay buffer.
            latest (int): The number of elements to add.
        """
        latest_trans = other.get_latest(latest)
        for transition in latest_trans:
            self.add(*transition)

    def shuffle(self):
        """Shuffles the transitions in the replay memory."""
        random.shuffle(self.memory)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing the sampled tensors\
                  (state, action, next_state, reward, done).
        """
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.as_tensor(np.concatenate(batch.state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.concatenate(batch.action), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.concatenate(batch.next_state), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.concatenate(batch.reward), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.concatenate(batch.done), dtype=torch.float32, device=self.device)
        return state, action, next_state, reward, done

    def sample_from_latest(self, batch_size, latest):
        """
        Samples a batch of transitions from the latest elements in the\
              replay memory.

        Args:
            batch_size (int): The size of the batch to sample.
            latest (int): The number of latest elements to consider.

        Returns:
            tuple: A tuple containing the sampled tensors\
                  (state, action, next_state, reward, done).
        """
        latest_trans = self.get_latest(latest)
        transitions = random.sample(latest_trans, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.as_tensor(np.concatenate(batch.state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.concatenate(batch.action), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.concatenate(batch.next_state), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.concatenate(batch.reward), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.concatenate(batch.done), dtype=torch.float32, device=self.device)
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0


class SequenceReplayBuffer:
    """
    A replay buffer for sequence-based RL algorithms like DreamerV3.

    Stores full episodes and samples sequences of a specified length.
    Each sequence maintains temporal continuity for training world models.

    Args:
        capacity (int): Maximum number of timesteps to store across all episodes.
        sequence_length (int): Length of sequences to sample (batch_length).
        device (str): Device to store tensors on.
    """

    def __init__(self, capacity, sequence_length, device='cpu'):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device

        # Storage - list of episodes, each episode is a dict of sequences
        self.episodes = []
        self.total_timesteps = 0

    def add_episode(self, episode):
        """
        Add a complete episode to the buffer.

        Args:
            episode (dict): Episode data containing:
                - states: list/array of observations [T, state_dim]
                - actions: list/array of actions [T, action_dim]
                - rewards: list/array of rewards [T]
                - dones: list/array of done flags [T]
        """
        # Convert episode to numpy arrays
        episode_data = {
            'states': np.array(episode['states'], dtype=np.float32),
            'actions': np.array(episode['actions'], dtype=np.float32),
            'rewards': np.array(episode['rewards'], dtype=np.float32),
            'dones': np.array(episode['dones'], dtype=np.float32),
            'next_states': np.array(episode['next_states'], dtype=np.float32),
        }

        self.episodes.append(episode_data)
        self.total_timesteps += len(episode['states'])

        # Remove old episodes if over capacity
        while self.total_timesteps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_timesteps -= len(removed['states'])

    def sample(self, batch_size):
        """
        Sample a batch of sequences from the replay buffer.

        Args:
            batch_size (int): Number of sequences to sample.

        Returns:
            dict: Batch of sequences containing:
                - states: [batch_size, sequence_length, state_dim]
                - actions: [batch_size, sequence_length, action_dim]
                - next_states: [batch_size, sequence_length, state_dim]
                - rewards: [batch_size, sequence_length]
                - dones: [batch_size, sequence_length]
                - is_first: [batch_size, sequence_length] - True for first step
                - is_last: [batch_size, sequence_length] - True for last step
        """
        if len(self.episodes) == 0:
            raise ValueError("Replay buffer is empty")

        sequences = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'is_first': [],
            'is_last': [],
        }

        for _ in range(batch_size):
            # Sample random episode weighted by length
            episode_lengths = [len(ep['states']) for ep in self.episodes]
            total_len = sum(episode_lengths)
            episode_probs = [l / total_len for l in episode_lengths]
            episode_idx = np.random.choice(len(self.episodes), p=episode_probs)
            episode = self.episodes[episode_idx]

            ep_len = len(episode['states'])

            if ep_len < self.sequence_length:
                # Episode shorter than sequence length - pad or take full episode
                start_idx = 0
                end_idx = ep_len

                # Pad sequence
                pad_length = self.sequence_length - ep_len
                state_seq = np.concatenate([
                    episode['states'],
                    np.tile(episode['states'][-1:], (pad_length, 1))
                ])
                action_seq = np.concatenate([
                    episode['actions'],
                    np.tile(episode['actions'][-1:], (pad_length, 1))
                ])
                next_state_seq = np.concatenate([
                    episode['next_states'] if 'next_states' in episode else episode['states'][1:],
                    np.tile(episode['states'][-1:], (pad_length + 1, 1))
                ])[:self.sequence_length]
                reward_seq = np.concatenate([
                    episode['rewards'],
                    np.tile(episode['rewards'][-1:], (pad_length,))
                ])
                done_seq = np.concatenate([
                    episode['dones'],
                    np.ones(pad_length)  # Mark padded as done
                ])
            else:
                # Sample random starting position
                max_start = ep_len - self.sequence_length
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + self.sequence_length

                state_seq = episode['states'][start_idx:end_idx]
                action_seq = episode['actions'][start_idx:end_idx]
                next_state_seq = episode['next_states'][start_idx:end_idx]
                reward_seq = episode['rewards'][start_idx:end_idx]
                done_seq = episode['dones'][start_idx:end_idx]

            # Create is_first and is_last flags
            is_first = np.zeros(self.sequence_length, dtype=bool)
            is_last = np.zeros(self.sequence_length, dtype=bool)
            is_first[0] = True
            is_last[-1] = True

            # Handle episode boundaries within sequence
            if np.any(done_seq[:-1]):
                # Find where episode ends
                end_pos = np.where(done_seq[:-1])[0][0] + 1
                is_last[:] = False
                is_last[end_pos - 1] = True

            sequences['states'].append(state_seq)
            sequences['actions'].append(action_seq)
            sequences['next_states'].append(next_state_seq)
            sequences['rewards'].append(reward_seq)
            sequences['dones'].append(done_seq)
            sequences['is_first'].append(is_first)
            sequences['is_last'].append(is_last)

        # Stack to create batch
        batch = {
            'states': np.stack(sequences['states']),
            'actions': np.stack(sequences['actions']),
            'next_states': np.stack(sequences['next_states']),
            'rewards': np.stack(sequences['rewards']),
            'dones': np.stack(sequences['dones']),
            'is_first': np.stack(sequences['is_first']),
            'is_last': np.stack(sequences['is_last']),
        }

        return batch

    def __len__(self):
        """Return total timesteps stored."""
        return self.total_timesteps

    def reset(self):
        """Clear the buffer."""
        self.episodes = []
        self.total_timesteps = 0


class EpisodeBuffer:
    """
    A buffer for collecting a single episode during rollout.

    This accumulates transitions and creates an episode dict for SequenceReplayBuffer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def get_episode(self):
        """
        Get the collected episode as a dict.

        Returns:
            dict: Episode data ready for SequenceReplayBuffer.
        """
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'next_states': np.array(self.next_states, dtype=np.float32),
        }

    def __len__(self):
        return len(self.states)
