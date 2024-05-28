# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Any, Dict

import networkx as nx
import numpy as np
import osmnx as ox
import uvicorn
from fastapi import FastAPI, HTTPException
from networkx.readwrite import json_graph
from pydantic import BaseModel

import openstreetsmap_api as osm
from envs import RideShareEnv

# Initialize FastAPI app
app = FastAPI()
lock = asyncio.Lock()

# Example graph for demonstration
G = osm.G_map_from_address("350 5th Ave, New York, New York")


# Function to serialize a graph to a dictionary
def serialize_graph(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    return nx.node_link_data(graph)


# Function to deserialize a dictionary to a graph
def deserialize_graph(data: Dict[str, Any]) -> nx.MultiDiGraph:
    return nx.node_link_graph(data)


# works


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Web3 Wheels AI API!"}


# Endpoint to get the serialized graph
# TODO check again
@app.get("/graph")
async def get_graph():
    async with lock:
        try:
            graph_data = serialize_graph(G)
            return graph_data
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# Endpoint to create or update the graph
@app.post("/graph")
async def update_graph(graph_data: Dict[str, Any]):
    async with lock:
        try:
            global G
            G = deserialize_graph(graph_data)
            return {"message": "Graph created/updated successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/ai-api/find-destinations")
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
    async with lock:
        try:
            global G
            print("data", data)
            print("type of data", type(data))
            # numpy array with size of (10, 2)
            actions = np.random.rand(10, 2)  # TODO replace with actual coordinator call
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


@app.post("/ai-api/mock/find-destinations")
async def find_mock_destinations(data: Dict[str, Any]):
    async with lock:
        try:
            global G
            matches = osm.find_best_matches(
                G, data["vehicle_node_ids"], data["passenger_node_ids"]
            )
            return [
                {"vehicle_node_id": agent, "destination_node_id": match}
                for agent, match in matches
            ]
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/ai-api/find-route")
async def find_route(data: Dict[str, Any]):
    """
    Find the path between two nodes in the map.

    Args:
        data (Dict[str, Any]): A dictionary containing the following
            keys:
            - source_node_id (Any): The ID of the source node.
            - target_node_id (Any): The ID of the target node.
            - passenger_id (Any): The ID of the passenger.
            eg.
            data = {
                "vehicle_id": 1,
                "source_node_id": 2,
                "target_node_id": 3,
                }
    """

    async with lock:
        try:
            shortest_path = osm.find_route(
                G, data["source_node_id"], data["target_node_id"]
            )
            return {"vehicle_id": data["vehicle_id"], "route": shortest_path}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# WORKS
@app.post("/ai-api/move-agent")
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
    async with lock:
        try:
            global G
            print("agent", agent)
            longitude, latitude = osm.find_x_y_coordinates_of_node(
                G, agent["next_node_id"]
            )
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


class VehicleData(BaseModel):
    vehicle_id: int
    position: Dict[str, float]


# WORKS
@app.post("/map-api/find-closest-node")
async def find_closest_node(data: VehicleData):
    """
    This endpoint finds the closest node in the graph to a given vehicle's position.

    It locks the global graph `G` to prevent concurrent modifications, then uses the `osm.find_closest_node`
    function to find the closest node to the vehicle's position. It then retrieves the longitude and latitude
    of the closest node using the `osm.find_x_y_coordinates_of_node` function.

    Args:
        data (VehicleData): A Pydantic model instance containing the vehicle's data.
            It should be in the following format:
            {
                "vehicle_id" : 0,
                "position" : {
                    "longitude" : 132.32131,
                    "latitude" : -32.321
                }
            }

    Returns:
        dict: A dictionary containing the vehicle's ID and the closest node's ID, longitude, and latitude.

    Raises:
        HTTPException: If an error occurs, an HTTPException is raised with status code 400 and the error message.
    """
    async with lock:
        try:
            global G
            print("vehicle", data)
            closest_node_id = osm.find_closest_node(
                G, data.position["longitude"], data.position["latitude"]
            )
            print("closest_node_id", closest_node_id)
            longitude, latitude = osm.find_x_y_coordinates_of_node(G, closest_node_id)
            return {
                "vehicle_id": data.vehicle_id,
                "position": {
                    "node_id": closest_node_id,
                    "longitude": longitude,
                    "latitude": latitude,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# WORKS
@app.post("/map-api/find-distance")
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
    async with lock:
        try:
            global G
            distance = osm.find_distance(
                G, data["source_node_id"], data["target_node_id"]
            )
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
