# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import sys
from typing import Any, Dict

import networkx as nx
import uvicorn
from fastapi import FastAPI, HTTPException
from networkx.readwrite import json_graph
from pydantic import BaseModel

from envs import RideShareEnv

# Initialize FastAPI app
app = FastAPI()

# Example graph for demonstration
G = nx.MultiDiGraph()


# Function to serialize a graph to a dictionary
def serialize_graph(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    return nx.node_link_data(graph)


# Function to deserialize a dictionary to a graph
def deserialize_graph(data: Dict[str, Any]) -> nx.MultiDiGraph:
    return nx.node_link_graph(data)


# TODO: function to Initialize Agent and add to list

"""
usage: taxi = TaxiAgent(environment, carInfo)
where carInfo is a dictionary with the following keys:
        carInfo["name"]
        carInfo["description"]
        carInfo["vin"]
        carInfo["position"]["longitude"],
        carInfo["position"]["latitude"],

        ## Other properties are OPTIONAL

Then add to env: RideShareEnv.add_agent(taxi)
"""
# TODO: function to Initialize Passenger and add to list

"""
usage:
    passenger = Passenger(
        passenger_id=1,
        pickup_location={"latitude": 37.824454, "longitude": -122.231589},
        dropoff_location={"latitude": 37.821592, "longitude": -122.234797},
    )


Then add to env: RideShareEnv.add_passenger(passenger)
"""


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


# Run the app with Uvicorn on a custom port
if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--port":
        print("Usage: python endpoint.py --port <port_number>")
        sys.exit(1)

    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)

    env = RideShareEnv()

    # Continuously update the environment
    while True:
        env.step()
