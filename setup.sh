#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

# Stop execution if any command fails
set -e

# Default Python version
DEFAULT_PYTHON_VERSION="3.10"

# Function to display usage
usage() {
    echo "Usage: $0"
    echo "Python version is $DEFAULT_PYTHON_VERSION)"
    exit 1
}

# Function to check if a command is available
command_exists() {
    command -v "$1" &>/dev/null
}

# Function to remove existing Pipenv environment
remove_existing_pipenv() {
    if [ -f Pipfile ]; then
        echo "Removing existing Pipenv environment..."
        pipenv --rm || true
        rm -f Pipfile.lock || true
    fi
}

# Check if pip is installed
check_pip() {
    if ! command_exists pip3; then
        echo "pip3 is not installed. Installing pip..."
        python3 -m ensurepip --default-pip
    fi
    echo "pip3 check completed."

}

# Install Pipenv using pip
check_pipenv() {
    if ! command_exists pipenv; then
        echo "pipenv is not installed. Installing pipenv..."
        pip3 install pipenv
        export PATH="$HOME/Library/Python/3.9/bin:$PATH"
        source ~/.bashrc

    fi
    echo "pipenv check completed."

}

# Install dependencies using Pipenv
install_dependencies() {
    echo "Installing dependencies..."
    pipenv install --dev
}

activate_pipenv() {
    echo "Activating Pipenv shell..."
    pipenv shell
}

# Setup pre-commit hooks from .pre-commit-config.yaml if it exists
setup_pre_commit_hooks() {
    if [ -f .pre-commit-config.yaml ]; then
        echo "Setting up pre-commit hooks..."
        pipenv run pre-commit install
    else
        echo "No .pre-commit-config.yaml found. Skipping pre-commit hook setup."
    fi
}

# Main setup function
main_setup() {
    check_pip
    check_pipenv
    remove_existing_pipenv
    install_dependencies
    setup_pre_commit_hooks
    echo "Setup completed successfully."
    activate_pipenv
}
main_setup
