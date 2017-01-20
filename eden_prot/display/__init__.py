#!/usr/bin/env python
"""Provides drawing for proteins."""

from eden.display import draw_graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from toolz import curry, map


def draw_ligand(graph):
    """draw_ligand."""
    display_params = dict(size=10,
                          edge_label='label',
                          vertex_label='label',
                          vertex_color='chain_id',
                          font_size=12, edge_alpha=.1,
                          vertex_alpha=.2,
                          colormap='RdBu',
                          vertex_size=400,
                          ignore_for_layout='nesting')
    draw_graph(graph, **display_params)


def draw_protein(graph, vertex_color='chain_id'):
    """draw_protein."""
    display_params = dict(size=70,
                          edge_label=None,
                          vertex_label='label',
                          vertex_color=vertex_color,
                          font_size=18,
                          edge_alpha=.2,
                          vertex_alpha=.2,
                          colormap='RdBu',
                          vertex_size=1500,
                          ignore_for_layout='nesting')
    draw_graph(graph, **display_params)


def draw_ligand_protein(graph):
    """draw_ligand_protein."""
    display_params = dict(size=70,
                          edge_label=None,
                          vertex_label='label',
                          vertex_color='contact',
                          font_size=18,
                          edge_alpha=.2,
                          vertex_alpha=.5,
                          colormap='RdBu',
                          vertex_size=1500,
                          ignore_for_layout='nesting')
    draw_graph(graph, **display_params)


def _extract(pos, iterable):
    return list(map(lambda x: x[pos], iterable))
_extract0 = curry(_extract)(0)
_extract1 = curry(_extract)(1)
_extract2 = curry(_extract)(2)


def _coord_list(start, end):
    x_list = []
    for x_start, x_end in zip(start, end):
        x_list.append(x_start)
        x_list.append(x_end)
        x_list.append(None)
    return x_list


def _extract_colors(graph, attribute):
    attributes = [graph.node[u].get(attribute, None) for u in graph.nodes()]
    if isinstance(attributes[0], basestring) or \
            isinstance(attributes[-1], basestring):
        attributes = LabelEncoder().fit_transform(attributes)
    return attributes


def _extract_node_coords(graph):
    coords = [graph.node[u]['coords'] for u in graph.nodes()]
    x_coords = _extract0(coords)
    y_coords = _extract1(coords)
    z_coords = _extract2(coords)
    return x_coords, y_coords, z_coords


def _extract_edge_coords(graph):
    start_coords = []
    end_coords = []
    typeof = []
    for u, v in graph.edges():
        start_coords.append(graph.node[u]['coords'])
        end_coords.append(graph.node[v]['coords'])
        typeof.append(graph.edge[u][v]['typeof'])

    start_x_coords = _extract0(start_coords)
    start_y_coords = _extract1(start_coords)
    start_z_coords = _extract2(start_coords)
    end_x_coords = _extract0(end_coords)
    end_y_coords = _extract1(end_coords)
    end_z_coords = _extract2(end_coords)

    return typeof, \
        start_x_coords, \
        end_x_coords, \
        start_y_coords, \
        end_y_coords, \
        start_z_coords, \
        end_z_coords


def plot3d(graph, angle=0, vertex_color='label'):
    """plot3d."""
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    x_coords, y_coords, z_coords = _extract_node_coords(graph)
    edge_typeof, \
        start_x_coords, \
        end_x_coords, \
        start_y_coords, \
        end_y_coords, \
        start_z_coords, \
        end_z_coords = _extract_edge_coords(graph)
    for t, x0, x1, y0, y1, z0, z1 in zip(edge_typeof,
                                         start_x_coords,
                                         end_x_coords,
                                         start_y_coords,
                                         end_y_coords,
                                         start_z_coords,
                                         end_z_coords):
        if t == 'ligand':
            ax.plot([x0, x1], [y0, y1], [z0, z1], '-', color='cornflowerblue',
                    lw=3, alpha=0.9)
        if t == 'ligand_protein':
            ax.plot([x0, x1], [y0, y1], [z0, z1], '-', color='green',
                    alpha=0.5)
        if t == 'protein_conj':
            ax.plot([x0, x1], [y0, y1], [z0, z1], '-', color='cornflowerblue',
                    alpha=0.2)
        if t == 'protein_disj':
            ax.plot([x0, x1], [y0, y1], [z0, z1], '-', color='blue',
                    alpha=0.05)
    colors = _extract_colors(graph, vertex_color)
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=40, cmap='RdBu')

    ax.view_init(30, angle)
    plt.draw()
