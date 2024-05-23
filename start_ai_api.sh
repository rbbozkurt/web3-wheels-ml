#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>


usage() {
    echo "Usage: $0 [--port <port_number>]"
    echo
    echo "Options:"
    echo "  --port <port_number>  Specify the port number to listen on. Default is 8080."
    exit 1
}

port_number=${1:-8080}

if [[ $1 == "--help" || $1 == "-h" ]]; then
    usage
fi

python src/ai_api/endpoint.py --port $port_number
