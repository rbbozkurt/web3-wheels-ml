<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>
-->

# Web3 Wheels Machine Learning

Welcome to the ML component of our decentralized autonomous vehicles (DAV) project! This repository houses the machine learning algorithms, model training techniques, and implementations used to optimize vehicle actions, vehicle management, and passenger experience within our cutting-edge DAV ecosystem.

## Setup

To set up the development environment for this project, follow these steps:

### 1. Clone the repository

#### HTTPS

```bash
git clone https://github.com/rbbozkurt/web3-wheels-ml.git
cd web3-wheels-ml
```

#### SSH

```bash
git clone git@github.com:rbbozkurt/web3-wheels-ml.git
cd web3-wheels-ml
```

### 2. Run the Setup Script

Execute the setup.sh script to set up the necessary dependencies and environment.

```[bash]
./setup.sh
```

This script will perform the following tasks:

- Check if `pip3` is installed and install it if necessary.
- Install `pipenv` using pip3.
- Remove any existing Pipenv environment and associated files (Pipfile.lock).
- Insall project dependencies, including development dependencies.
- Setup pre-commit hooks from `.pre-commit-config.yaml` if it exists.

The following things are done by hooks automatically:

- formatting of python files using black and isort
- formatting of other files using prettier
- syntax check of JSON and yaml files
- adding new line at the end of files
- removing trailing whitespaces
- prevent commits to `dev` and `main` branch
- check adherence to REUSE licensing format

### 2. Activate Environment

To work within the environment you can now run:

```[bash]
# to activate the virtual environment
pipenv shell
# to run a single command
pipenv run <COMMAND>
```

To install new packages in the environment add them to the `Pipfile`. Always pin the exact package version to avoid package conflicts and unexpected side effects from package upgrades.

```[bash]
# to add a package to the development environment
[dev-packages]
<PACKAGE_NAME> = "==<VERSION_NUMBER>"
# to add a package to the production environment
[packages]
<PACKAGE_NAME> = "==<VERSION_NUMBER>"
```

Note that this project runs under an MIT license and we only permit the use of non-copyleft-licensed packages. Please be aware of this when installing new packages and inform yourself before blindly installing.

When you have any issues with the environment contact `rbbozkurt`.

### 3. Run the AI API

To run the application, execute the `./src/ai_api/endpoint.py` script with a port number:

```[bash]
python src/ai_api/endpoint.py --port <port_number>
```

#### 1. Welcome Message

- Endpoint: `GET /`
- Description : Returns a welcome message
- Success Response:

```json
{
  "message": "Welcome to the Web3 Wheels AI API!"
}
```

### 2. Find Destinations

- Endpoint: `POST /ai-api/find-destinations`

- Description: Finds destinations for agents and passengers.

- Request Example:

```json
{
  "num_agents": 2,
  "num_passengers": 2,
  "agent_positions": [
    [37.824454, -122.231589],
    [37.821592, -122.234797]
  ],
  "passenger_positions": [
    [37.824454, -122.231589],
    [37.821592, -122.234797]
  ],
  "passenger_destinations": [
    [37.824454, -122.231589],
    [37.821592, -122.234797]
  ]
}
```

- Sucess Response:

```json
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
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

### 3. Find Mock Destinations

- Endpoint: `POST /ai-api/mock/find-destinations`

- Description: Finds the best matches for vehicles and passengers based on their node IDs.

- Request Example:

```json
{
  "vehicle_node_ids": [42433644, 1312312],
  "passenger_node_ids": [42459032342398, 42433644, 42459098, 31231]
}
```

- Success Response:

```json
    [
        {
            "vehicle_node_id": 42433644,
            "destination_node_id": 42433644
        },
        ...
    ]
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

### 4. Find Route

- Endpoint: `POST /ai-api/find-route`

- Description: Finds the shortest path between two nodes in the map for a given vehicle.

- Request Example:

```json
{
  "vehicle_id": 1,
  "source_node_id": 42433644,
  "target_node_id": 3431231
}
```

- Succes Response :

```json
    {
        "vehicle_id": 1,
        "route": [42433644, ..., 3431231]
    }
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

### 5. Move Agent

- Endpoint: `POST /ai-api/move-agent`

- Description: Moves the agent to the next node based on the provided agent information.

- Request Example:

```json
{
  "vehicle_id": 1,
  "next_node_id": 2
}
```

- Success Response:

```json
{
  "vehicle_id": 1,
  "position": {
    "node_id": 2,
    "longitude": 37.824454,
    "latitude": -122.231589
  }
}
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

### 6. Find Closest Node:

- Endpoint: `POST /map-api/find-closest-node`

- Description: Finds the closest node in the graph to a given vehicle's position.

- Request Example:

```json
{
  "vehicle_id": 0,
  "position": {
    "longitude": 132.32131,
    "latitude": -32.321
  }
}
```

- Success Response:

```json
{
  "vehicle_id": 0,
  "position": {
    "node_id": 42433644,
    "longitude": 132.32131,
    "latitude": -32.321
  }
}
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

### 7. Find Distance

- Endpoint: `POST /map-api/find-distance`

- Description: Calculates the distance between two nodes in the map.

- Request Example:

```json
{
  "passenger_id": 1,
  "source_node_id": 2,
  "target_node_id": 3
}
```

- Success Response:

```json
{
  "passenger_id": 1,
  "distance": 1000
}
```

- Error Response:

```json
{
  "detail": "Error message"
}
```

## License

This project is operated under an MIT license. Every file must contain the REUSE-compliant license and copyright declaration:

[REUSE documentation](https://reuse.software/faq/)

```[bash]
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024
```
