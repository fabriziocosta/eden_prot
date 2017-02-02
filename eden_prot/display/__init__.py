#!/usr/bin/env python
"""Provides drawing for proteins."""

from eden.display import draw_graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from toolz import curry, map
from ipywidgets import interactive, fixed, FloatSlider, IntSlider
from ipywidgets import HBox, VBox
from eden_prot import make_trimmed_ligand_protein_graph


def _interactive_display(structure, ligand_marker,
                         min_conj=3.5,
                         max_conj=5.5,
                         max_disj=6,
                         depth=2,
                         inter_dist=4,
                         v_ang=20, h_ang=20, size=10):
    trimmed_ligand_protein_graph = make_trimmed_ligand_protein_graph(
        structure, ligand_marker,
        min_dist_conj=min_conj,
        max_dist_conj=max_conj,
        max_dist_disj=max_disj,
        depth=depth,
        interaction_distance_th=inter_dist)
    plot3d(trimmed_ligand_protein_graph, vangle=v_ang, hangle=h_ang, size=size)


def interactive_plot3d(structure, ligand_marker):
    """interactive_display."""
    w = interactive(
        _interactive_display,
        structure=fixed(structure), ligand_marker=fixed(ligand_marker),
        min_conj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=4.5, continuous_update=False),
        max_conj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=5.5, continuous_update=False),
        max_disj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=6.0, continuous_update=False),
        depth=IntSlider(
            min=0, max=5, step=1, value=2, continuous_update=False),
        inter_dist=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=3.1, continuous_update=False),
        v_ang=FloatSlider(
            min=0.0, max=180.0, step=5.0, value=30, continuous_update=False),
        h_ang=FloatSlider(
            min=0.0, max=180.0, step=5.0, value=30, continuous_update=False),
        size=IntSlider(
            min=9, max=20, step=1, value=15, continuous_update=False))

    int_w = VBox([HBox(w.children[3:5]),
                  HBox(w.children[:3]),
                  HBox(w.children[5:])])
    return int_w


def _interactive_draw_ligand_protein(structure, ligand_marker,
                                     min_conj=3.5,
                                     max_conj=5.5,
                                     max_disj=6,
                                     depth=2,
                                     inter_dist=4,
                                     size=10):
    trimmed_ligand_protein_graph = make_trimmed_ligand_protein_graph(
        structure, ligand_marker,
        min_dist_conj=min_conj,
        max_dist_conj=max_conj,
        max_dist_disj=max_disj,
        depth=depth,
        interaction_distance_th=inter_dist)
    draw_ligand_protein(trimmed_ligand_protein_graph, size=size)


def interactive_draw_ligand_protein(structure, ligand_marker):
    """interactive_draw_ligand_protein."""
    w = interactive(
        _interactive_draw_ligand_protein,
        structure=fixed(structure),
        ligand_marker=fixed(ligand_marker),
        min_conj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=4.5, continuous_update=False),
        max_conj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=5.5, continuous_update=False),
        max_disj=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=6.0, continuous_update=False),
        depth=IntSlider(
            min=0, max=5, step=1, value=1, continuous_update=False),
        inter_dist=FloatSlider(
            min=1.0, max=8.0, step=0.1, value=3.1, continuous_update=False),
        size=IntSlider(
            min=20, max=70, step=1, value=60, continuous_update=False))

    int_w = VBox([HBox(w.children[3:5]),
                  HBox(w.children[:3]),
                  HBox(w.children[5:])])
    return int_w


def draw_ligand(graph, size=70):
    """draw_ligand."""
    display_params = dict(size=size,
                          edge_label='label',
                          vertex_label='label',
                          vertex_color='chain_id',
                          font_size=12, edge_alpha=.1,
                          vertex_alpha=.2,
                          colormap='RdBu',
                          vertex_size=400,
                          ignore_for_layout='nesting')
    draw_graph(graph, **display_params)


def draw_protein(graph, vertex_color='chain_id', size=70):
    """draw_protein."""
    display_params = dict(size=size,
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


def draw_ligand_protein(graph, size=70):
    """draw_ligand_protein."""
    display_params = dict(size=size,
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


def plot3d(graph, hangle=30, vangle=30, size=20, vertex_color='label'):
    """plot3d."""
    fig = plt.figure(figsize=(size, size))
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

    ax.view_init(vangle, hangle)
    plt.draw()
