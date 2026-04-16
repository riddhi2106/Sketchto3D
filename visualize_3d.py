"""
Member 4 - 3D Visualization with Plotly
=======================================
Builds an interactive 3D figure from the extruded point cloud.
"""

import numpy as np
import plotly.graph_objects as go

from extrude import build_wall_segments


def build_mesh_figure(vertices, faces, color="lightsteelblue", opacity=0.85):
    """Render a solid-slab triangle mesh via Plotly Mesh3d."""
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(title="Mesh empty - try raising alpha or lowering stride")
        return fig

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            name="Solid slab",
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title="X (col)",
            yaxis_title="Y (row)",
            zaxis_title="Z (depth)",
            aspectmode="data",
            yaxis=dict(autorange="reversed"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Member 4 - 3D Solid Slab",
    )
    return fig


def build_figure(points_3d, edges, show_walls=True, point_size=2):
    """
    Build an interactive Plotly 3D figure of the extruded sketch.

    Parameters
    ----------
    points_3d : (2N, 3) array
        Bottom layer stacked on top layer from extrude_points.
    edges : (N, 2) int array
        Index pairs connecting bottom and top layers.
    show_walls : bool
        If True, draw vertical line segments between layers.
    point_size : int
        Marker size for the scatter points.
    """
    n = edges.shape[0]
    bottom = points_3d[:n]
    top    = points_3d[n:]

    traces = [
        go.Scatter3d(
            x=bottom[:, 0], y=bottom[:, 1], z=bottom[:, 2],
            mode="markers",
            marker=dict(size=point_size, color="steelblue"),
            name="Bottom (z=0)",
        ),
        go.Scatter3d(
            x=top[:, 0], y=top[:, 1], z=top[:, 2],
            mode="markers",
            marker=dict(size=point_size, color="crimson"),
            name="Top (z=depth)",
        ),
    ]

    if show_walls and n > 0:
        walls = build_wall_segments(points_3d, edges)
        traces.append(
            go.Scatter3d(
                x=walls[:, 0], y=walls[:, 1], z=walls[:, 2],
                mode="lines",
                line=dict(color="gray", width=2),
                name="Walls",
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (col)",
            yaxis_title="Y (row)",
            zaxis_title="Z (depth)",
            aspectmode="data",
            yaxis=dict(autorange="reversed"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        title="Member 4 - 3D Extruded Sketch",
    )
    return fig
