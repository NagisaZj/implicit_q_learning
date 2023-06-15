import collections
from typing import Optional

import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
from tqdm import tqdm
import tree
from acme import types

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def split_into_trajectories_transferred(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            types.Transition(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                discount=masks[i],
                next_observation=next_observations[i],extras=dones_float[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def merge_trajectories_transferred(trajs):
    # observation = observations[i],
    # action = actions[i],
    # reward = rewards[i],
    # discount = masks[i],
    # next_observation = next_observations[i]))
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, next_obs, done) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
        next_observations)

def merge_trajectories_transferred_selected(trajs,idx):
    # observation = observations[i],
    # action = actions[i],
    # reward = rewards[i],
    # discount = masks[i],
    # next_observation = next_observations[i]))
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []
    cnt=0
    for traj in trajs:
        for (obs, act, rew, mask, next_obs, done) in traj:
            observations.append(obs)
            actions.append(act)
            if cnt in idx:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)
        cnt+=1

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
        next_observations)

def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,reward_model_path="",disable_pds=False):
        # dataset = d4rl.qlearning_dataset(env,reward_model_path=reward_model_path,disable_pds=disable_pds)
        dataset = d4rl.qlearning_dataset(env)
        lengths = dataset['observations'].shape[0]
        num_selected = int(lengths*0.1)
        selected_args = np.argsort(dataset['rewards'])[-num_selected:]
        # dataset['observations'] = dataset['observations'][selected_args]
        # dataset['actions'] = dataset['actions'][selected_args]
        # dataset['rewards'] = dataset['rewards'][selected_args]
        # dataset['rewards'] = np.zeros_like(dataset['rewards'])
        # dataset['rewards'][selected_args] = 2
        # dataset['next_observations'] = dataset['next_observations'][selected_args]
        # dataset['terminals'] = dataset['terminals'][selected_args]
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


def qlearning_dataset_with_timeouts(env,
                                    dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
  if dataset is None:
    dataset = env.get_dataset(**kwargs)

  N = dataset['rewards'].shape[0]
  obs_ = []
  next_obs_ = []
  action_ = []
  reward_ = []
  done_ = []
  realdone_ = []
  if "infos/goal" in dataset:
    if not disable_goal:
      dataset["observations"] = np.concatenate(
          [dataset["observations"], dataset['infos/goal']], axis=1)
    else:
      pass
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)

  episode_step = 0
  for i in range(N - 1):
    obs = dataset['observations'][i]
    new_obs = dataset['observations'][i + 1]
    action = dataset['actions'][i]
    reward = dataset['rewards'][i]
    done_bool = bool(dataset['terminals'][i])
    realdone_bool = bool(dataset['terminals'][i])
    if "infos/goal" in dataset:
      final_timestep = True if (dataset['infos/goal'][i] !=
                                dataset['infos/goal'][i + 1]).any() else False
    else:
      final_timestep = dataset['timeouts'][i]

    if i < N - 1:
      done_bool += final_timestep

    if (not terminate_on_end) and final_timestep:
      # Skip this transition and don't apply terminals on the last step of an episode
      episode_step = 0
      continue
    if done_bool or final_timestep:
      episode_step = 0

    obs_.append(obs)
    next_obs_.append(new_obs)
    action_.append(action)
    reward_.append(reward)
    done_.append(done_bool)
    realdone_.append(realdone_bool)
    episode_step += 1

  return {
      'observations': np.array(obs_),
      'actions': np.array(action_),
      'next_observations': np.array(next_obs_),
      'rewards': np.array(reward_)[:],
      'terminals': np.array(done_)[:],
      'realterminals': np.array(realdone_)[:],
  }


class TopkD4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 env_name,k=10):
        if "antmaze" in env_name and True:
            dataset = qlearning_dataset_with_timeouts(env)
        else:
            dataset = d4rl.qlearning_dataset(env)
        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        if 'realterminals' in dataset:
            # We updated terminals in the dataset, but continue using
            # the old terminals for consistency with original IQL.
            masks = 1.0 - dataset['realterminals'].astype(np.float32)
        else:
            masks = 1.0 - dataset['terminals'].astype(np.float32)
        offline_traj = split_into_trajectories_transferred(
            observations=dataset['observations'].astype(np.float32),
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=masks,
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset['next_observations'].astype(np.float32))

        if "antmaze" in env_name:
            # 1/Distance (from the bottom-right corner) times return
            returns = [
                sum([t.reward
                     for t in traj]) /
                (1e-4 + np.linalg.norm(traj[0].observation[:2]))
                for traj in offline_traj
            ]
        else:
            returns = [sum([t.reward for t in traj]) for traj in offline_traj]
        idx = np.argpartition(returns, -k)[-k:]
        # idx2 = np.argpartition(returns, -k)[:-k]
        demo_returns = [returns[i] for i in idx]
        print(f"demo returns {demo_returns}, mean {np.mean(demo_returns)}, dattaset mean {np.mean(returns)}")
        expert_demo = [offline_traj[i] for i in idx]
        # expert_demo2 = [offline_traj[i] for i in idx2]
        #
        # for i in range(len(expert_demo)):
        #     expert_demo[i]['reward'] = np.ones_like( expert_demo[i]['reward'])+1
        # for i in range(len(expert_demo2)):
        #     expert_demo2[i]['reward'] = np.ones_like(expert_demo2[i]['reward'])-1
        # expert_demo = expert_demo + expert_demo2

        new_dataset = merge_trajectories_transferred(expert_demo)
        # new_dataset = merge_trajectories_transferred_selected(offline_traj,idx)
        print(dataset['observations'].shape,dataset['actions'].shape,dataset['rewards'].shape,dataset['next_observations'].shape,dones_float.shape,masks.shape,dataset['terminals'].shape,)
        print(new_dataset[0].shape,new_dataset[1].shape,new_dataset[2].shape,new_dataset[3].shape,new_dataset[4].shape,new_dataset[5].shape)

        # np.stack(observations), np.stack(actions), np.stack(
            # rewards), np.stack(masks), np.stack(dones_float), np.stack(
            # next_observations)

        super().__init__(new_dataset[0].astype(np.float32),
                         actions=new_dataset[1].astype(np.float32),
                         rewards=new_dataset[2].astype(np.float32),
                         masks=new_dataset[3].astype(np.float32),
                         dones_float=new_dataset[4].astype(np.float32),
                         next_observations=new_dataset[5].astype(
                             np.float32),
                         size=len(new_dataset[0]))

class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
