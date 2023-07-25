from envs.dflex_env import DFlexEnv
import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu

class FrankaEnv(DFlexEnv):
    def __init__(self, render=False, device='cuda:0', num_envs=64, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency=1, early_termination=False):
        num_obs = 37
        num_act = 7
        self.steps = 0
        super(FrankaEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()

        #-----------------------
        # set up Usd renderer
        if (self.visualize):
            self.stage = Usd.Stage.CreateNew("outputs/" + "Franka_" + str(self.num_envs) + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 7
        self.num_joint_qd = 7
        self.start_joint_q = torch.tensor((0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4), device=self.device)
        self.start_joint_qd = torch.tensor((0, 0., 0.0, 0., 0., 0., 0.), device=self.device)

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):
            lu.urdf_load(self.builder,
                         os.path.join(asset_folder, 'franka_panda', 'panda.urdf'),
                         df.transform((0.0, 0.01, 0.0 + self.env_dist * i), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
                         floating=False,
                         shape_kd=1e4,
                         limit_kd=1.)
            self.builder.joint_q[i * self.num_joint_q:(i+1)*self.num_joint_q] = self.start_joint_q

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        # self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)
        self.model.gravity = torch.tensor((0.0, 0., 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if (self.model.ground):
            self.model.collide(self.state)


    def step(self, actions):
        with df.ScopedTimer("simulate", active=False, detailed=False):
            actions = actions.view((self.num_envs, self.num_actions))

            actions = torch.clip(actions, -1., 1.)

            self.actions = actions

            # self.state.joint_act.view(self.num_envs, -1)[:, :self.num_joint_qd] = actions
            self.state.joint_q.view(self.num_envs, -1)[:, 0] += 0.01

            self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
            self.sim_time += self.sim_dt
        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1
        self.steps += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            raise NotImplementedError

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        with df.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)

        with df.ScopedTimer("render", active=False, detailed=False):
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :self.num_joint_q] = self.start_joint_q.view(-1, self.num_joint_q)[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :self.num_joint_qd] = self.start_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()

        self.progress_buf[env_ids] = 0
        self.calculateObservations()
    # return self.obs_buf

    def calculateObservations(self):
        pass

    def calculateReward(self):
        pass

    def render(self, mode = 'human'):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)
            if (self.num_frames == 40):
                try:
                    self.stage.Save()
                except:
                    print('USD save error')
                self.num_frames -= 40
