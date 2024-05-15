# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import random as rnd

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx

import openstreetsmap_api as osm


class GraphNavigationEnv(gym.Env):
    rendering_started = False
    fig, ax = None, None

    def __init__(self, graph):
        self.graph = graph
        self.start_node_id = osm.get_random_nodes(self.graph, False)
        self.goal_node_id = osm.get_random_nodes(self.graph, False)
        self.current_node_id = self.start_node_id
        self.shortest_path = self._calculate_shortest_path()
        # TODO check when neighbors(current_node_id) is empty
        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))
        self.observation_space = gym.spaces.Discrete(len(self.graph))

    def reset(self):
        self.start_node_id = osm.get_random_nodes(self.graph, False)
        self.goal_node_id = osm.get_random_nodes(self.graph, False)
        self.current_node_id = self.start_node_id
        self.shortest_path = self._calculate_shortest_path()
        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))
        return self._get_info()

    def step(self, action):
        reward, terminated, truncated = 0, False, False

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

        self.current_node_id = list(self.graph[self.current_node_id].keys())[action]

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

        if len(self.graph[self.current_node_id]) == 0:
            truncated = True
            reward = -100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )

        self.action_space = gym.spaces.Discrete(len(self.graph[self.current_node_id]))

        if nx.has_path(self.graph, self.current_node_id, self.goal_node_id):
            shortest_path_len = nx.shortest_path_length(
                self.graph, self.current_node_id, self.goal_node_id
            )
            if shortest_path_len < len(self.shortest_path):
                reward = 1
            else:
                reward = -1
        else:
            truncated = True
            reward = -100
            return (
                self.observation_space,
                reward,
                terminated,
                truncated,
                self._get_info(),
            )

        return self.observation_space, reward, terminated, truncated, self._get_info()

    def render(self):
        if not self.rendering_started:
            self.rendering_started = True
            plt.ion()
            self.fig, self.ax = osm.draw_plain_map(self.graph)
            plt.show(block=False)
        else:
            self.ax.clear()
            _, self.ax = osm.draw_plain_map(self.graph, pre_ax=self.ax)

        osm.draw_passengers_and_vehicles(
            self.graph,
            pre_ax=self.ax,
            vehicles_node_ids=[self.current_node_id],
            passengers_node_ids=[self.goal_node_id],
        )
        osm.draw_outgoing_roads_to_neighbors(
            self.graph, self.current_node_id, pre_ax=self.ax
        )
        osm.draw_route_on_map(self.graph, [self.shortest_path], self.ax)

    def close(self):
        if self.rendering_started:
            plt.ioff()
            plt.close()
            self.rendering_started = False

    def _calculate_shortest_path(self):
        if nx.has_path(self.graph, self.current_node_id, self.goal_node_id):
            return nx.shortest_path(self.graph, self.current_node_id, self.goal_node_id)
        else:
            return None

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
