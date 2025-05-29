from typing import List, Optional, Union

import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Mesh

from src.tasks.domains.mesh_wrapper import MeshWrapper


def plot_mesh(
    mesh: Union[Mesh, MeshWrapper],
    scalars: np.ndarray = None,
    title: str = "Value per element",
    boundary: np.ndarray = None,
    logscale: bool = False,
    colorscale: str = "Jet",
    symmetric: bool = False,
    inner_wireframe: Union[bool, str] = "auto",
) -> go.Figure:
    """
    Plots the values of a scalar field defined over the elements or vertices of a mesh.

    Args:
        mesh (Union[Mesh, MeshWrapper]):
            The mesh object containing the elements over which the scalar field is defined.
        scalars (np.ndarray): Array of scalar values corresponding to the elements or vertices of the mesh.
        title (str): The title of the plot.
        boundary (np.ndarray, optional): Array defining the boundaries of the plot.
            If None, boundaries are determined from the mesh. Defaults to None.
        logscale (bool, optional): Whether to use a logarithmic scale for the scalar values. Defaults to False.
        colorscale (str, optional): The color scale to use for plotting the scalar values. Defaults to "Jet".
        symmetric (bool, optional): Whether to make the color scale symmetric around zero. Defaults to False.
        inner_wireframe (bool, optional): Whether to include the inner wireframe of the mesh in the plot. This aids
            3d visualization, but makes the plot slower and bulkier. Defaults to "auto", which plots the inner wireframe
            only for 3d meshes with less than 10000 elements.

    Returns:
        go.Figure: A Plotly figure object representing the plot of scalar values over the mesh elements or vertices.
    """
    if scalars is None:
        # Default to element size
        if isinstance(mesh, MeshWrapper):
            scalars = mesh.simplex_volumes
        else:
            scalars = MeshWrapper(mesh).simplex_volumes

    element_midpoint_trace = scalar_contour_trace(
        mesh=mesh,
        scalars=scalars.flatten(),
        trace_name="Scalar",
        logscale=logscale,
        colorscale=colorscale,
        symmetric=symmetric,
    )
    wireframe_trace_ = wireframe_trace(mesh, inner_wireframe=inner_wireframe)  # vertex and wireframe traces
    traces = [element_midpoint_trace, wireframe_trace_]
    layout = get_layout(boundary=boundary, title=title, mesh=mesh)
    value_per_element_plot = go.Figure(data=traces, layout=layout)
    return value_per_element_plot


def scalar_contour_trace(
    mesh: Mesh,
    scalars: np.array,
    trace_name: str = "Value (v)",
    logscale: bool = False,
    colorscale: str = "Jet",
    round_legend: bool = False,
    symmetric: bool = False,
    visible: str | bool = True,
) -> BaseTraceType:
    """
    Generates a Plotly contour trace representing scalar values on a mesh.

    Args:
        mesh (Mesh):
            A scikit-fem mesh object containing element and node connectivity.
        scalars (np.array):
            A NumPy array containing scalar values evaluated either per mesh element or per mesh node.
        trace_name (str, optional):
            The name of the trace to appear in the legend. Defaults to `"Value (v)"`.
        logscale (bool, optional):
            If `True`, the color scale is displayed in log scale. Defaults to `False`.
        colorscale (str, optional):
            The color scheme used for visualization. Defaults to `"Jet"`.
        round_legend (bool, optional):
        symmetric (bool, optional):
            If `True`, the color scale is symmetric around zero (useful for error maps). Defaults to `False`.
        visible (str, optional):
            The initial visibility setting. Options are:
            - `"true"` (default): The trace is visible when the plot loads.
            - `"legendonly"`: The trace is hidden initially but can be toggled via the legend.

    Returns:
        BaseTraceType:
            A Plotly trace representing the scalar field on the mesh.

    Notes:
        - If `symmetric=True`, the color scale is centered around zero for better visualization of signed errors.
        - The `visible` parameter allows controlling which traces are initially displayed.
    """
    scalars = scalars.flatten()
    if scalars.shape[0] == mesh.nelements:
        intensitymode = "cell"
    elif scalars.shape[0] == mesh.nvertices:
        intensitymode = "vertex"
    else:
        raise ValueError(f"Invalid shape for scalars. Must be ({mesh.nvertices},) or ({mesh.nelements},), given {scalars.shape}")
    vertices = mesh.p
    if mesh.dim() == 2:
        # append a row of zeros to the vertices to make it 3d
        vertices = np.concatenate((vertices, np.zeros((1, vertices.shape[1]))), axis=0)
        hovertemplate = "v: %{text:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<extra></extra>"
        faces = mesh.t
    else:
        hovertemplate = "v: %{text:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>"

        # Map each boundary facet to its corresponding element
        # (since on the boundary, each facet belongs to one element)
        boundary_facet_indices = mesh.boundary_facets()
        faces = mesh.facets[:, boundary_facet_indices]

        if intensitymode == "cell":
            boundary_elements = mesh.f2t[0, boundary_facet_indices]  # Assuming the first row links facets to elements
            scalars = scalars[boundary_elements]  # Assign element solution to its boundary facets

    colorbar = dict(yanchor="top", y=1, x=0, ticks="outside")  # put colorbar on the left
    if logscale:
        # Apply logarithm. Assumes scalars are all positive and non-zero.
        adjusted_scalars = np.log(scalars)
        cmin, cmax = np.nanmin(adjusted_scalars), np.nanmax(adjusted_scalars)

        # Generate tick values evenly distributed across the log-transformed scale
        colorbar["tickvals"] = np.linspace(cmin, cmax, num=5)
        # Map these tick values back to the original scale for the labels. Round for nicer display
        colorbar["ticktext"] = [f"{np.exp(tv):.2e}" for tv in colorbar["tickvals"]]
    else:
        adjusted_scalars = scalars
        cmin, cmax = np.nanmin(scalars), np.nanmax(scalars)

    if round_legend:
        from src.helpers.qol import round_up_to_next

        cmax = round_up_to_next(cmax)
        cmin = round_up_to_next(cmin)

    if symmetric:
        cmax = max(abs(cmin), abs(cmax))
        cmin = -cmax

    face_trace = go.Mesh3d(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        flatshading=True,  # Enable flat shading
        i=faces[0],
        j=faces[1],
        k=faces[2],
        text=scalars,
        intensity=adjusted_scalars,
        intensitymode=intensitymode,
        cmin=cmin,
        cmax=cmax,
        colorbar=colorbar,
        colorscale=colorscale,
        name=trace_name,
        hovertemplate=hovertemplate,
        showlegend=True,
        visible=visible,
    )
    return face_trace


def wireframe_trace(
    mesh: Mesh,
    color: str = "black",
    showlegend: bool = True,
    opacity: float = 1.0,
    name: str = "Wireframe",
    inner_wireframe: bool | str = "auto",
    visible: bool | str = True,
) -> BaseTraceType:
    """
    Generates a 3D wireframe plot trace for a scikit-fem simplex mesh in 2D or 3D.

    Args:
        mesh (Mesh):
            A scikit-fem mesh object containing vertex coordinates and connectivity information.
            The mesh must have the following attributes:
            - `mesh.facets`: A (2, num_edges) or (3, num_edges) array listing edge connections.
            - `mesh.p`: A (2, num_nodes) or (3, num_nodes) array containing vertex coordinates.
        color (str, optional):
            The color of the wireframe lines. Defaults to "black".
        showlegend (bool, optional):
            Whether to display the legend entry for this trace. Defaults to True.
        opacity (float, optional):
            The opacity of the wireframe plot, between 0 (transparent) and 1 (fully visible). Defaults to 1.0.
        name (str, optional):
            The name of the trace. Defaults to "Wireframe".
        inner_wireframe (bool | str, optional):
            Determines whether to include internal edges in 3D meshes.
            - If "auto", it is set to True for 3D meshes with fewer than 3000 elements.
            - If True, all edges are plotted.
            - If False, only boundary edges are plotted. Defaults to "auto".
        visible (str, optional): Initial visibility setting ("true" or "legendonly"). Defaults to "true".

    Returns:
        BaseTraceType:
            A `Scatter3d` trace representing the wireframe structure of the mesh.

    Raises:
        ValueError: If the mesh dimension is not 2 or 3.

    Notes:
        - For 2D meshes, an extra row of zeros is added to the vertex coordinates to enable 3D plotting.
        - In 3D, setting `inner_wireframe=False` will only show boundary edges.
    """
    if inner_wireframe == "auto":
        inner_wireframe = mesh.dim() == 3 and mesh.nelements < 3000
    vertices = mesh.p
    if mesh.dim() == 2:
        # append a row of zeros to the vertices to make it 3d
        vertices = np.concatenate((vertices, np.zeros((1, vertices.shape[1]))), axis=0)

    # edge/wireframe trace
    if mesh.dim() == 2:
        edges = mesh.facets
    elif mesh.dim() == 3 and inner_wireframe:
        edges = mesh.edges
    elif mesh.dim() == 3 and not inner_wireframe:
        edges = mesh.edges[:, mesh.boundary_edges()]
    else:
        raise ValueError(f"Invalid mesh dimension {mesh.dim()}. Must be 2 or 3.")
    num_edges = edges.shape[1]
    edge_x_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_z_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_x_positions[0::3] = vertices[0, edges[0]]
    edge_x_positions[1::3] = vertices[0, edges[1]]
    edge_y_positions[0::3] = vertices[1, edges[0]]
    edge_y_positions[1::3] = vertices[1, edges[1]]
    edge_z_positions[0::3] = vertices[2, edges[0]]
    edge_z_positions[1::3] = vertices[2, edges[1]]

    trace = go.Scatter3d(
        x=edge_x_positions,
        y=edge_y_positions,
        z=edge_z_positions,
        mode="lines",
        line=dict(color=color, width=1),
        name=name,
        showlegend=showlegend,
        opacity=opacity,
        visible=visible,
    )
    return trace


def vertex_trace(
    mesh: Mesh,
    color: str = "black",
    showlegend: bool = True,
    name: str = "Vertices",
    inner_wireframe: bool | str = "auto",
) -> BaseTraceType:
    """
    Generates a 3D scatter plot trace of the vertices of a given mesh.

    Args:
        mesh (Mesh):
            The mesh object containing the vertex coordinates and connectivity information.
        color (str, optional):
            The color of the vertex markers in the plot. Defaults to "black".
        showlegend (bool, optional):
            Whether to display the legend entry for this trace. Defaults to True.
        name (str, optional):
            The name of the trace. Defaults to "Vertices".
        inner_wireframe (bool | str, optional):
            Determines whether to plot all vertices or only boundary vertices for 3D meshes.
            If "auto", it is set to True for 3D meshes with fewer than 3000 elements.
            If True, all vertices are plotted.
            If False, only boundary vertices are plotted. Defaults to "auto".

    Returns:
        BaseTraceType:
            A `Scatter3d` trace representing the mesh vertices, to be used with Plotly.

    Notes:
        - For 2D meshes, an extra row of zeros is added to the vertex coordinates to enable 3D plotting.
        - If `inner_wireframe` is False and the mesh is 3D, only boundary nodes are included in the trace.
    """
    if inner_wireframe == "auto":
        inner_wireframe = mesh.dim() == 3 and mesh.nelements < 3000
    vertices = mesh.p
    if mesh.dim() == 2:
        # append a row of zeros to the vertices to make it 3d
        vertices = np.concatenate((vertices, np.zeros((1, vertices.shape[1]))), axis=0)
    if mesh.dim() == 3 and not inner_wireframe:
        plotted_vertices = vertices[:, mesh.boundary_nodes()]
    else:
        plotted_vertices = vertices
    # vertex trace
    vertex_trace = go.Scatter3d(
        x=plotted_vertices[0],
        y=plotted_vertices[1],
        z=plotted_vertices[2],
        mode="markers",
        marker={"size": 0.6, "color": color},
        name=name,
        showlegend=showlegend,
    )
    return vertex_trace


def get_3d_scatter_mesh_traces(
    mesh: Mesh,
    scalars,
    colorbar_title: str = "Scalars",
    opacity: float = 0.8,
    size: int = 1,
    showlegend: bool = True,
    colorscale: str = "Jet",
    symmetric: bool = False,
) -> List[BaseTraceType]:
    """
    Generates Plotly traces for a 3D scatter plot over a scikit-fem mesh.

    This function returns a list of Plotly traces representing the mesh vertices
    as point markers colored by a given scalar field and a wireframe overlay.

    Args:
        mesh (Mesh): A scikit-fem Mesh object. Should contain the attribute `mesh.p`
            with shape (3, num_vertices), holding coordinates of vertices.
        scalars (array-like): An array of shape (num_vertices,) or (num_elements containing
            the scalar field to be plotted. If num_elements, the element midpoints are used.
        colorbar_title (str, optional): Title for the colorbar. Defaults to "Potential".
        opacity (float, optional): The opacity level of the scatter plot. Defaults to 0.8.
        size (int, optional): The size of the scatter markers. Defaults to 1.
        showlegend (bool, optional): Determines if legend should be shown in the plot. Defaults to True.
        colorscale (str, optional): The colorscale to use for the plot. Defaults to "Jet".
        symmetric (bool, optional): If True, the color scale is symmetric around 0. Defaults to False.

    Returns:
        List[BaseTraceType]: A list of Plotly traces, including:
            - One trace representing the colored scatter plot.
            - Additional traces for the mesh wireframe.
    """
    if scalars.shape[0] == mesh.nelements:
        positions = mesh.p[:, mesh.t].mean(axis=1)
    elif scalars.shape[0] == mesh.nvertices:
        positions = mesh.p
    else:
        raise ValueError(f"Invalid shape for scalars. " f"Must be ({mesh.nelements},)  or ({mesh.nvertices},), given {scalars.shape}")

    scatter_trace = get_3d_scatter_trace(
        positions,
        scalars,
        colorbar_title,
        opacity,
        showlegend=showlegend,
        size=size,
        colorscale=colorscale,
        symmetric=symmetric,
    )
    wireframe_trace_ = wireframe_trace(mesh, showlegend=showlegend)
    traces = [scatter_trace, wireframe_trace_]
    # no node trace because this is contained in the scatter plot
    return traces


def get_3d_scatter_trace(
    positions,
    scalars,
    colorbar_title,
    opacity: float = 0.8,
    showlegend: bool = True,
    size: int = 1,
    colorscale: str = "Jet",
    symmetric: bool = False,
):
    if symmetric:
        cmax = max(abs(np.nanmin(scalars)), abs(np.nanmax(scalars)))
        cmin = -cmax
    else:
        cmin, cmax = np.nanmin(scalars), np.nanmax(scalars)

    scatter_trace = go.Scatter3d(
        x=positions[0],
        y=positions[1],
        z=positions[2],
        mode="markers",
        showlegend=showlegend,
        marker=dict(
            size=size,
            color=scalars,  # set color to an array/list of desired values
            colorscale=colorscale,  # choose a colorscale
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            colorbar=dict(
                title=colorbar_title,
                yanchor="top",
                y=1,
                x=0,
                ticks="outside",  # put colorbar on the left
            ),
        ),
        hovertemplate="v: %{marker.color:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>",
        name="Value (v)",
    )
    return scatter_trace


def get_layout(
    mesh: Mesh,
    boundary: np.array = None,
    title: Optional[str] = None,
) -> go.Layout:
    """
    Get a layout for a plotly figure.
    Args:
        boundary: The boundary of the plot. The format is [x_min, y_min, x_max, y_max]
        title: the title of the plot
        mesh: The mesh object

    Returns:

    """
    if boundary is None:
        boundary = np.concatenate((mesh.p.min(axis=1), mesh.p.max(axis=1)), axis=0)
    if mesh.dim() == 2:
        assert boundary is not None, "Boundary must be specified for mesh2d layout"
        layout = go.Layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=""),
                yaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=""),
                zaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=""),
                aspectmode="data",
                aspectratio=dict(x=1, y=1, z=1),  # Set aspect ratio to 1:1:1
                camera=dict(
                    eye=dict(x=0, y=0, z=1.5),  # Camera positioned above the plot
                    up=dict(x=0, y=1, z=0),  # Up direction along Y-axis
                ),
                dragmode="pan",  # Set default interaction mode to pan
            ),
            margin=dict(l=0, r=0, b=0, t=50),  # Reduce plot margins
            title=dict(text=title, x=0.5, y=0.95, xanchor="center", yanchor="top"),  # Center the title
            legend=dict(
                x=1,  # Horizontal position (1 is far right)
                y=0.3,  # Vertical position (1 is top)
                xanchor="right",  # Anchor the legend's right edge at x position
                yanchor="top",  # Anchor the legend's top edge at y position
            ),
        )
    elif mesh.dim() == 3:
        layout = go.Layout(
            scene=dict(
                aspectmode="data",
                aspectratio=dict(x=1, y=1, z=1),  # Set aspect ratio to 1:1:1
            ),
            title=dict(
                text=title,
                x=0.5,  # Center the title
                xanchor="center",
                yanchor="top",
                y=0.9,
            ),
        )

    else:
        raise ValueError(f"Unknown mesh dimension {mesh.dim()=}")
    return layout
