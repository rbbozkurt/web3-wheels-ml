# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

import networkx as nx

from openstreetsmap_api import G_map_from_address, G_map_from_place

graph_world_350_5th_Ave_New_York_New_York = G_map_from_address(
    "350 5th Ave, New York, New York"
)
graph_world_Berlin_Germany = G_map_from_place("Berlin, Germany")
