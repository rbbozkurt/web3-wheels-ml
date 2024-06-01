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
            print("Received data: ", data)
            print("Type of received data: ", type(data))

            actions = np.random.rand(10, 2)  # TODO replace with actual coordinator call
            print("Generated actions: ", actions)

            agents_actions = actions[: data["num_agents"]]
            print("Agent actions: ", agents_actions)

            agents_actions_list = agents_actions.tolist()
            print("Agent actions list: ", agents_actions_list)

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
            print("Result: ", result)
            return result
        except Exception as e:
            print("An error occurred: ", str(e))
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/ai-api/mock/find-destinations")
async def find_mock_destinations(data: Dict[str, Any]):
    """
    This function is an endpoint that finds the best matches for vehicles and passengers based on their node IDs.

    It accepts a POST request with a JSON body containing two arrays: 'vehicle_node_ids' and 'passenger_node_ids'.
    Each array should contain integers representing node IDs.

    Example request body:
    {
        "vehicle_node_ids": [42433644,1312312],
        "passenger_node_ids": [42459032342398,42433644, 42459098, 31231]
    }

    The function returns a list of dictionaries. Each dictionary contains a 'vehicle_node_id' and a 'destination_node_id'.
    The 'destination_node_id' is the best match for the corresponding 'vehicle_node_id'.

    Example response:
    [
        {
            "vehicle_node_id": 42433644,
            "destination_node_id": 42433644
        }
    ]

    If an error occurs during the process, it raises an HTTPException with status code 400 and the error message as detail.

    :param data: A dictionary containing two keys: 'vehicle_node_ids' and 'passenger_node_ids'.
                 Each key corresponds to an array of integers representing node IDs.
    :return: A list of dictionaries. Each dictionary contains a 'vehicle_node_id' and a 'destination_node_id'.
    """
    async with lock:
        try:
            print("Received data: ", data)
            print("Type of received data: ", type(data))

            global G
            matches = osm.find_best_matches(
                G, data["vehicle_node_ids"], data["passenger_node_ids"]
            )
            print("Matches: ", matches)

            result = [
                {"vehicle_node_id": agent, "destination_node_id": match}
                for agent, match in matches
            ]
            print("Result: ", result)
            return result
        except Exception as e:
            print("An error occurred: ", str(e))
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/ai-api/find-route")
async def find_route(data: Dict[str, Any]):
    """
    Find the shortest path between two nodes in the map for a given vehicle.

    Args:
        data (Dict[str, Any]): A dictionary containing the following keys:
            - vehicle_id (Any): The ID of the vehicle.
            - source_node_id (Any): The ID of the source node.
            - target_node_id (Any): The ID of the target node.

        eg. data = {
            "vehicle_id": 1,
            "source_node_id": 42433644,
            "target_node_id": 42433644,
        }

    Returns:
        Dict[str, Any]: A dictionary containing the vehicle ID and the shortest route from source node to target node.

        eg. return {
            "vehicle_id": 1,
            "route": [
                42433644
            ]
        }

    Raises:
        HTTPException: If an error occurs during the route finding process.
    """

    async with lock:
        try:
            print("Received data for find-route: ", data)
            print("Type of received data: ", type(data))

            shortest_path = osm.find_route(
                G, data["source_node_id"], data["target_node_id"]
            )
            print("Shortest path: ", shortest_path)

            result = {"vehicle_id": data["vehicle_id"], "route": shortest_path}
            print("Result: ", result)

            return result
        except Exception as e:
            print("An error occurred: ", str(e))
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
            print("Received agent data: ", agent)
            print("Type of received agent data: ", type(agent))

            longitude, latitude = osm.find_x_y_coordinates_of_node(
                G, agent["next_node_id"]
            )
            print("Longitude and Latitude: ", longitude, latitude)

            result = {
                "vehicle_id": agent["vehicle_id"],
                "position": {
                    "node_id": agent["next_node_id"],
                    "longitude": longitude,
                    "latitude": latitude,
                },
            }
            print("Result: ", result)

            return result
        except Exception as e:
            print("An error occurred: ", str(e))
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
            print("Received vehicle data: ", data)
            print("Type of received vehicle data: ", type(data))

            closest_node_id = osm.find_closest_node(
                G, data.position["longitude"], data.position["latitude"]
            )
            print("Closest node ID: ", closest_node_id)

            longitude, latitude = osm.find_x_y_coordinates_of_node(G, closest_node_id)
            print("Longitude and Latitude: ", longitude, latitude)

            result = {
                "vehicle_id": data.vehicle_id,
                "position": {
                    "node_id": closest_node_id,
                    "longitude": longitude,
                    "latitude": latitude,
                },
            }
            print("Result: ", result)

            return result
        except Exception as e:
            print("An error occurred: ", str(e))
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
            print("Received data for find-distance: ", data)
            print("Type of received data: ", type(data))

            distance = osm.find_distance(
                G, data["source_node_id"], data["target_node_id"]
            )
            print("Calculated distance: ", distance)

            result = {"passenger_id": data["passenger_id"], "distance": distance}
            print("Result: ", result)

            return result
        except Exception as e:
            print("An error occurred: ", str(e))
            raise HTTPException(status_code=400, detail=str(e))


# Run the app with Uvicorn on a custom port
if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--port":
        print("Usage: python endpoint.py --port <port_number>")
        sys.exit(1)

    port = int(sys.argv[2])
    print(f"Running API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
