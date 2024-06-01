# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import json
import os
import sys

# Add the parent directory of 'tests' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from fastapi.testclient import TestClient

from src.ai_api.endpoint import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Web3 Wheels AI API!"}


def test_initial_get_graph():
    G = nx.MultiDiGraph()
    response = client.get("/graph")
    assert response.status_code == 200
    assert response.json() == nx.node_link_data(G)


def test_get_graph_after_update():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2)
    response = client.post("/graph", json=nx.node_link_data(G))
    assert response.status_code == 200
    assert response.json() == {"message": "Graph created/updated successfully"}
    response = client.get("/graph")
    assert response.status_code == 200
    assert response.json() == nx.node_link_data(G)
    assert nx.is_isomorphic(nx.node_link_graph(response.json()), G)
