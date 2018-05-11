import numpy as np
from physics_sim import PhysicsSim
import copy
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class MyTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/  (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, done, rotor_speeds):
        """Uses current pose of sim to return reward."""
        diff = self.sim.pose[:3] - self.target_pos
        sdv = sum(np.fabs(self.sim.angular_v))
        dfav = sum(np.fabs(self.sim.v))
        sba = sum(np.fabs(self.sim.linear_accel))
        sbaa = sum(np.fabs(self.sim.angular_accels))
        # reward = max(-1, 1. - 0.02 * sum(np.fabs(diff)) - 0.02 * (sdv + dfav) + 0.01 * (sba + sbaa)) + (2 - 0.4 * self.sim.time)
        reward = max(-1, 1 - .3 * sum(np.abs(diff))  )
        # if done and self.sim.time < self.sim.runtime:
        #     reward = -1
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        rotor_speeds = np.repeat(rotor_speeds, 4)
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(done, rotor_speeds)
            inipose = copy.copy(self.sim.pose)
            inipose[:3] = inipose[:3] - self.target_pos
            pose_all.append(np.tanh(inipose))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def gettime(self):
        return self.sim.time

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        inipose = copy.copy(self.sim.pose)
        inipose[:3] = inipose[:3] - self.target_pos
        state = np.concatenate([np.tanh(inipose)] * self.action_repeat)
        return state


class MyTask4():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/  (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 4

        self.state_size = self.action_repeat * 6
        self.action_low = -1
        self.action_high = 1
        self.action_size = 5


        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, done, rotor_speeds):
        """Uses current pose of sim to return reward."""
        diff = self.sim.pose[:3] - self.target_pos
        sdv = sum(np.fabs(self.sim.angular_v))
        dfav = sum(np.fabs(self.sim.v))
        sba = sum(np.fabs(self.sim.linear_accel))
        sbaa = sum(np.fabs(self.sim.angular_accels))
        # reward = max(-1, 1. - 0.02 * sum(np.fabs(diff)) - 0.02 * (sdv + dfav) + 0.01 * (sba + sbaa)) + (2 - 0.4 * self.sim.time)
        reward = max(-1, 1 - .3 * sum(np.abs(diff))  )
        if done and self.sim.time < self.sim.runtime:
             reward = -1
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        rotor_speeds = np.repeat(rotor_speeds, 4)
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(done, rotor_speeds)
            inipose = copy.copy(self.sim.pose)
            inipose[:3] = inipose[:3] - self.target_pos
            pose_all.append(np.tanh(inipose))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def gettime(self):
        return self.sim.time

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        inipose = copy.copy(self.sim.pose)
        inipose[:3] = inipose[:3] - self.target_pos
        state = np.concatenate([np.tanh(inipose)] * self.action_repeat)
        return state

import gym
class TestTask():
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        self.env = gym.make('Pendulum-v0')
        self.state_size = len(self.env.reset())
        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]
        self.action_size = 1
        self.time = 0
    def step(self, act):
        self.time+=1
        s,r,d,_ = self.env.step(act)
        return s,r,d

    def gettime(self):
        return self.time

    def reset(self):
        self.time = 0
        return self.env.reset()