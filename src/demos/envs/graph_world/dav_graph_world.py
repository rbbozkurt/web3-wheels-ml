# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
import random as rnd

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx

import openstreetsmap_api as osm


class GraphNavigationEnv(gym.Env):
    """
    An environment where a road map is represented as a graph and the agent's goal is to navigate from one random node to another random.
    """

    def __init__(self, graph):
        self.graph = graph
        self.start_node_id = osm.get_random_nodes(self.graph, False)
        self.goal_node_id = osm.get_random_nodes(self.graph, False)
        self.current_node_id = self.start_node_id
        self.shortest_path = (
            nx.shortest_path(self.graph, self.current_node_id, self.goal_node_id)
            if nx.has_path(self.graph, self.current_node_id, self.goal_node_id)
            else -1
        )
        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))
        self.observation_space = gym.spaces.Discrete(len(self.graph))

    def reset(self):
        self.start_node_id = osm.get_random_nodes(self.graph, False)
        self.goal_node_id = osm.get_random_nodes(self.graph, False)
        self.current_node_id = self.start_node_id
        self.shortest_path = (
            nx.shortest_path(self.graph, self.current_node_id, self.goal_node_id)
            if nx.has_path(self.graph, self.current_node_id, self.goal_node_id)
            else None
        )
        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))
        return self._get_info()

    def _get_info(self):
        return {
            "start": self.start_node_id,
            "goal": self.goal_node_id,
            "current": self.current_node_id,
            "dist": nx.shortest_path_length(
                self.graph, self.current_node_id, self.goal_node_id
            )
            if nx.has_path(self.graph, self.current_node_id, self.goal_node_id)
            else -1,
        }

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # if random goal is not reachable from random selected start node then terminate the episode
        if not nx.has_path(self.graph, self.current_node_id, self.goal_node_id):
            truncated = True
            reward = -100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )
        if self.current_node_id == self.goal_node_id:
            terminated = True
            reward = 100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )

        self.current_node_id = list(self.graph[self.current_node_id].keys())[action]
        print("Current Node ID: ", self.current_node_id)
        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))

        if not nx.has_path(self.graph, self.current_node_id, self.goal_node_id):
            truncated = True
            reward = -100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )
        if self.current_node_id == self.goal_node_id:
            terminated = True
            reward = 100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )
        shortest_path_len = nx.shortest_path_length(
            self.graph, self.current_node_id, self.goal_node_id
        )
        if shortest_path_len < len(self.shortest_path):
            reward = +1
        reward = -1
        return self.observation_space, reward, terminated, truncated, self._get_info()

    def render(self):
        plt.ion()
        plt.show()
        fig, ax = osm.draw_plain_map(self.graph)
        osm.draw_passengers_and_vehicles(
            self.graph,
            pre_ax=ax,
            vehicles_node_ids=[self.current_node_id],
            passengers_node_ids=[self.goal_node_id],
        )

    def close(self):
        plt.close()
