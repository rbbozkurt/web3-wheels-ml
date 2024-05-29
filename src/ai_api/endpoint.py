# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

import networkx as nx
import numpy as np
import osmnx as ox
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from networkx.readwrite import json_graph
from pydantic import BaseModel

import openstreetsmap_api as osm

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # TODO: Fix paths for import if possible
sys.path.append(project_root)
from src.agents.coordinator_agent import AICoordinator
from src.envs.osm_env import RideShareEnv

# Initialize FastAPI app
app = FastAPI()

# Example graph for demonstration
city = "Piedmont, California, USA"
G = osm.G_map_from_address("350 5th Ave, New York, New York")  # TODO Try new city

# Initialize AI Coordinator
with open("src/training/ppo_config.yml", "r") as f:
    config = yaml.safe_load(f)
trained_coordinator = AICoordinator(RideShareEnv(map_area=city), config)
trained_coordinator.model = trained_coordinator.model.load(
    "src/training/saved_models/trained_coordinator"
)


# Function to serialize a graph to a dictionary
def serialize_graph(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    return nx.node_link_data(graph)


# Function to deserialize a dictionary to a graph
def deserialize_graph(data: Dict[str, Any]) -> nx.MultiDiGraph:
    return nx.node_link_graph(data)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Web3 Wheels AI API!"}


# Endpoint to get the serialized graph
@app.get("/graph", response_model=Dict[str, Any])
async def get_graph():
    try:
        graph_data = serialize_graph(G)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint to create or update the graph
@app.post("/graph")
async def update_graph(graph_data: Dict[str, Any]):
    try:
        global G
        G = deserialize_graph(graph_data)
        return {"message": "Graph created/updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint to perform a prediction or operation using the graph
@app.post("/graph/predict")
async def predict(graph_data: Dict[str, Any]):
    try:
        graph = deserialize_graph(graph_data.dict())
        # Here you would integrate your prediction logic using the graph
        # For demonstration, we'll just return the number of nodes and edges
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        return {"num_nodes": num_nodes, "num_edges": num_edges}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ai-api/find-destinations")
async def find_destinations(data: Dict[str, Any]):
    """
    Find destinations for agents and passengers.

    Args:
        data (Dict[str, Any]): A dictionary containing the following keys:
            - num_agents (int): The number of agents.
            - num_passengers (int): The number of passengers.
            - agent_positions (List[List[float]]): The positions of the agents.
            - passenger_positions (List[List[float]]): The positions of the passengers.
            - passenger_destinations (List[List[float]]): The destinations of the passengers.

    Returns:
        Example:
            [
                {
                    "position": {
                        "node_id": 11543660372,
                        "longitude": 0.15496014168933958,
                        "latitude": 0.9391560276780935
                    }
                },
                ...
            ]

    Raises:
        HTTPException: If an error occurs while processing the data.

    Example:
        data = {
            "num_agents": 2,
            "num_passengers": 2,
            "agent_positions": [[37.824454, -122.231589], [37.821592, -122.234797]],
            "passenger_positions": [[37.824454, -122.231589], [37.821592, -122.234797]],
            "passenger_destinations": [[37.824454, -122.231589], [37.821592, -122.234797]],
        }

    The function takes a dictionary `data` as input, which contains information about the number of agents, passengers, their positions, and destinations. It then processes the data and returns a list of actions for the agents. If an error occurs during the processing, an HTTPException with a status code of 400 and the error message is raised.
    """
    try:
        global G
        print("data", data)
        print("type of data", type(data))
        # numpy array with size of (10, 2)
        actions = trained_coordinator.inference(
            data
        )  # TODO replace with actual coordinator call
        print("actions", actions)
        # take first data.num_agents from array
        agents_actions = actions[: data["num_agents"]]
        agents_actions_list = agents_actions.tolist()
        # Convert list of actions to the specified format
        result = [
            {
                "position": {
                    "node_id": osm.find_closest_node(G, action[0], action[1]),
                    "longitude": action[0],
                    "latitude": action[1],
                }
            }
            for i, action in enumerate(agents_actions_list)
        ]
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ai-api/move-agent")
async def move_agent(agent: Dict[str, Any]):
    """
    Move the agent to the next node based on the provided agent information.

    Args:
        agent (Dict[str, Any]): The agent information containing the vehicle ID and the next node ID.

        eg. data = {
            "vehicle_id": 1,
            "next_node_id": 2,
        }

    Returns:
        Dict[str, Any]: The updated position of the agent, including the node ID, longitude, and latitude.

        eg. return {
            "vehicle_id": 1,
            "position": {
                "node_id": 2,
                "longitude": 37.824454,
                "latitude": -122.231589,
            },
        }

    Raises:
        HTTPException: If an error occurs during the movement of the agent.
    """
    try:
        global G
        print("agent", agent)
        longitude, latitude = osm.find_x_y_coordinates_of_node(G, agent["next_node_id"])
        return {
            "vehicle_id": agent["vehicle_id"],
            "position": {
                "node_id": agent["next_node_id"],
                "longitude": longitude,
                "latitude": latitude,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/map-api/find-closest-node")
async def find_closest_node(data: Dict[str, Any]):
    """
    Find the closest node in the map to the given vehicle's position.

    Args:
        data (Dict[str, Any]): A dictionary containing information about the vehicle.

        eg. data = {
            "vehicle_id": 1,
            "position": {
                "longitude": 23.824454,
                "latitude": -112.54332,
            },
        }

    Returns:
        Dict[str, Any]: A dictionary containing the vehicle's ID and the closest node's ID, longitude, and latitude.

        eg. return {
            "vehicle_id": 1,
            "position": {
                "node_id": 2,
                "longitude": 37.824454,
                "latitude": -122.231589,
            },
        }
    """
    try:
        global G
        print("vehicle", data)
        closest_node_id = osm.find_closest_node(
            G, data["position"]["longitude"], data["position"]["latitude"]
        )
        print("closest_node_id", closest_node_id)
        longitude, latitude = osm.find_x_y_coordinates_of_node(G, closest_node_id)
        return {
            "vehicle_id": data["vehicle_id"],
            "position": {
                "node_id": closest_node_id,
                "longitude": longitude,
                "latitude": latitude,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/map-api/find-distance")
async def find_distance(data: Dict[str, Any]):
    """
    Calculate the distance between two nodes in the map.

    Args:
        data (Dict[str, Any]): A dictionary containing the following keys:
            - source_node_id (Any): The ID of the source node.
            - target_node_id (Any): The ID of the target node.
            - passenger_id (Any): The ID of the passenger.

        eg. data = {
            "passenger_id": 1,
            "source_node_id": 2,
            "target_node_id": 3,
        }


    Returns:
        Dict[str, Any]: A dictionary containing the passenger ID and the calculated distance.

        eg. return {
            "passenger_id": 1,
            "distance": 1000,
        }

    Raises:
        HTTPException: If an error occurs during the distance calculation.
    """
    try:
        global G
        distance = osm.find_distance(G, data["source_node_id"], data["target_node_id"])
        return {"passenger_id": data["passenger_id"], "distance": distance}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run the app with Uvicorn on a custom port
if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--port":
        print("Usage: python endpoint.py --port <port_number>")
        sys.exit(1)

    port = int(sys.argv[2])
    print(f"Running API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
