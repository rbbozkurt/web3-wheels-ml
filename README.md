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

To run the application, execute the `start_ai_api.sh` script with an optional port number:

```[bash]
./start_ai_api.sh --port <port_number>
```

If no port number is provided, the application will listen on port `8080` by default.

You can also display usage information by running the script with the `--help` or `-h` option:

```[bash]
./start_ai_api.sh --help
```

## License

This project is operated under an MIT license. Every file must contain the REUSE-compliant license and copyright declaration:

[REUSE documentation](https://reuse.software/faq/)

```[bash]
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024
```
