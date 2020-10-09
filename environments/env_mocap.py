import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import numpy as np
import torch


class EnvBase(gym.Env):
    def __init__(
        self,
        num_parallel,
        device="cpu",
    ):
        self.np_random = None
        self.seed()

        self.is_rendered = False
        self.num_parallel = num_parallel
        self.device = device

        self.root_facing = torch.zeros((num_parallel, 1)).to(device)
        self.root_xz = torch.zeros((num_parallel, 2)).to(device)
        self.reward = torch.zeros((num_parallel, 1)).to(device)
        self.done = torch.zeros((num_parallel, 1)).bool().to(device)

    def create_viewer(
        self,
        data_fps=30,
        use_params=False,
        camera_tracking=True,
    ):
        from common.bullet_utils import MocapViewer

        self.is_rendered = True
        self.data_fps = data_fps

        self.viewer = MocapViewer(
            self,
            num_characters=self.num_parallel,
            target_fps=self.data_fps,
            use_params=use_params,
            camera_tracking=camera_tracking,
        )

    def integrate_root_translation(self, pose):
        mat = self.get_rotation_matrix(self.root_facing)
        displacement = (mat * pose[:, 0:2].unsqueeze(1)).sum(dim=2)
        self.root_facing.add_(pose[:, [2]]).remainder_(2 * np.pi)
        self.root_xz.add_(displacement)

    def get_rotation_matrix(self, yaw):
        yaw = -yaw
        col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
        col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
        matrix = torch.stack((col1, col2), dim=-1)
        return matrix

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def calc_env_state(self, next_frame):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.is_rendered:
            self.viewer.close()


class MocapReplayEnv(EnvBase):
    def __init__(self, **kwargs):

        # Process keywords not used by EnvBase
        mocap_data = kwargs.pop("mocap_data", None)

        if mocap_data is None or not isinstance(mocap_data, np.ndarray):
            raise ValueError("MocapReplayEnv expects `mocap_data`: np.ndarray")

        super().__init__(**kwargs)

        self.mocap_data = torch.from_numpy(mocap_data).to(self.device)
        pose_dim = int(np.prod(self.mocap_data.shape[1:]))
        self.current_frame_index = torch.zeros(self.num_parallel).long().to(self.device)
        self.current_frame_index.random_(0, self.mocap_data.shape[0])

        # Action is a placeholder
        high = np.inf * np.ones(0)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Observation is the current frame
        high = np.inf * np.ones(pose_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def get_observation_components(self):
        self.current_pose = self.mocap_data[self.current_frame_index]
        return (self.current_pose.flatten(-2),)

    def reset(self):
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.reward.fill_(0)
        self.done.fill_(False)
        self.current_frame_index.random_(0, self.mocap_data.shape[0])

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def step(self, action):
        self.current_frame_index = (self.current_frame_index + 1) % len(self.mocap_data)

        obs_components = self.get_observation_components()

        self.render()

        state = torch.cat(obs_components, dim=1)
        return state, self.reward, self.done, {}

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(self.current_pose, self.root_facing, self.root_xz)
