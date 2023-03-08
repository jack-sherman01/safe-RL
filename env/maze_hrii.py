import os
import pickle
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np

from gym import Env
from gym import utils
from gym.spaces import Box

#to be deleted
# from mujoco_py import load_model_from_path, MjSim


# the mudule we can use from Gazeb


"""
Constants associated with the Maze env.
"""

HORIZON = 100
MAX_FORCE = # the threshold of max_force, the epispde should terminate when current_force_value >= MAX_FORCE
FAILURE_COST = 0
GOAL_THRESH = 3e-2

GT_STATE = True
DENSE_REWARD = True

# THE coordnate of start point of ee, END_POINT of the task, BOUNDARIES: the max/min position of maze walls (x, y, z) * (min, max) 
START_POINT = [0,0]
END_POINT = []
BOUNDARIES =[] 


def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


def get_offline_data(num_transitions, save_rollouts=False):
    env = MazeHrii()
    transitions = []
    num_constraints = 0
    total = 0
    rollouts = []

    for i in range(1 * num_transitions // 2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        

    for i in range(1 * num_transitions // 2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])
        action = env.expert_action() 
        next_state, reward, done, info = env.step(action)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state

    print("data dist", total, num_constraints)
    if save_rollouts:
        return rollouts
    else:
        return transitions


class MazeHrii(Env, utils.EzPickle):
    def __init__(self): 
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        # TODO: Start gazebo and load hrii maze world
        self.sim = # see example: openai_ros robot_gazebo_env.py, we may use GazeboConnection class or not
        
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_offline_data 
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2), ##### maybe we can use all of the basic code of Gym Box here, or we just redifine the function
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_offline_data
        obs = self._get_obs() 
        self.observation_space = obs
        # self.observation_space = Box(-0.3, 0.3, shape=obs.shape)
        self.dense_reward = DENSE_REWARD

        # TODO: initiate the values of the constants
        START_POINT = [0,0]
        END_POINT = []
        BOUNDARIES =[] 

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

    def step(self, action):
        # TODO: one step of simulation execution, example: openai_ros robot_gazebo_env.py's step() function
        action = process_action(action)
        self.sim.data.qvel[:] = 0 # forcefully sets agent velocity to zero
        # TODO: send action to sim to execute, so a func in Gazebo can receive the action value
        # example: openai_ros fetch_push.py's _set_action() function
        self.sim.data.ctrl[:] = action 

        cur_obs = self._get_obs()
        constraint = int(self.sim.data.ncon > 3)
        if not constraint:
            for _ in range(500):
                self.sim.step()
        obs = self._get_obs()
        self.sim.data.qvel[:] = 0
        self.steps += 1
        constraint = int(self.sim.data.ncon > 3)
        self.done = self.steps >= self.horizon or constraint or (
            self.get_distance_score() < GOAL_THRESH)
        if not self.dense_reward:
            reward = -(self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": cur_obs,
            "next_state": obs,
            "action": action,
            "success": reward>-0.03
        }

        return obs, reward, self.done, info

    def _get_obs(self):
        # TODO: Return state info from Gazebo:
        # End-effector positions (ee_pos[:],ee_vel[:]), end effector force (ee_force[:])
        
        # below was how it was done in Mujoco:
        # See also: openai_ros / fetch_push.py _get_obs() implementation for Gazebo example
        state = np.concatenate(
            [self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy(), self.sim.data.force[:].copy()])
        return state[:2]  # State is just (x, y) position

    def reset(self, difficulty='h', check_constraint=True, pos=()):
        # TODO: Return the robot to the start position, to run the exp. again
        # we use this function to start new RL trials
        # example: robot_gazebo_env.py
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.22, 0.22)

        self.steps = 0

        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)
            self.sim.data.qpos = START_POINT

        return self._get_obs()

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
            """
        # TODO: calculate the distance of ee_pos to end goal
        # we can use the data we got from _get_obs already
        d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
        return d

    def expert_action(self):
        # TODO: get ee pos
        st = self.sim.data.qpos[:]
        if st[0] <= -0.151:
            delt = (np.array([-0.15, -0.125]) - st)
        elif st[0] <= 0.149:
            delt = (np.array([0.15, 0.125]) - st)
        else:
            delt = (np.array([self.goal[0], self.goal[1]]) - st)
        act = self.gain * delt

        return act
