from typing import Optional

import numpy as np
from plotly import graph_objects as go

from src.algorithm.util.amber_util import interpolate_vertex_field
from src.helpers.custom_types import MeshGenerationStatus
from src.mesh_util.mesh_visualization import (
    get_layout,
    scalar_contour_trace,
    vertex_trace,
    wireframe_trace,
)
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.fem_problem import FEMProblem


def get_learner_plot(
    predicted_mesh: MeshWrapper,
    labels: Optional[np.ndarray] = None,
    solution: Optional[np.ndarray] = None,
    predicted_sizing_field: Optional[np.ndarray] = None,
    mesh_generation_status: MeshGenerationStatus = "success",
    plot_boundary: Optional[np.ndarray] = None,
):
    """
    Get plots for the learner's mesh, labels, predictions and the solution.
    Args:
        predicted_mesh: The predicted mesh. Used to plot the solution on.
        labels:
        predicted_sizing_field:
        solution:
        mesh_generation_status: Status of the refinement/mehs generation. If "failed", could not generate a new mesh.
        plot_boundary: If provided, the bounding box of the mesh. Used for nicer plot formatting
    Returns:

    """
    traces = []
    if solution is not None:
        solution_trace = scalar_contour_trace(
            mesh=predicted_mesh.mesh,
            scalars=solution,
            trace_name="Solution",
        )
        traces.append(solution_trace)

    if labels is not None:
        label_trace = scalar_contour_trace(
            mesh=predicted_mesh.mesh,
            scalars=labels,
            trace_name="True Sizing Field",
            logscale=True,
            visible="legendonly",
        )
        traces.append(label_trace)
    else:
        element_size_trace = scalar_contour_trace(
            mesh=predicted_mesh.mesh,
            scalars=predicted_mesh.simplex_volumes,
            trace_name="Element Size",
            logscale=True,
            visible="legendonly",
        )
        traces.append(element_size_trace)

    if predicted_sizing_field is not None:
        sizing_field_trace = scalar_contour_trace(
            mesh=predicted_mesh.mesh,
            scalars=predicted_sizing_field,
            trace_name="Prediction",
            logscale=True,
        )
        traces.append(sizing_field_trace)
        projected_mesh = predicted_mesh.mesh
        projected_predictions = predicted_sizing_field

        if labels is not None:
            if projected_mesh is not None:
                error_trace = scalar_contour_trace(
                    mesh=projected_mesh,
                    scalars=projected_predictions - labels,
                    trace_name="Error",
                    colorscale="RdBu",
                    symmetric=True,
                    round_legend=True,
                    visible="legendonly",
                )
                traces.append(error_trace)

    traces.append(wireframe_trace(mesh=predicted_mesh.mesh, name=f"Predicted Mesh"))  # Mesh wireframe
    traces.append(vertex_trace(mesh=predicted_mesh.mesh, name=f"Predicted Vertices"))  # Mesh vertices

    title = f"Prediction. Status: {mesh_generation_status}. " f"Elem: {predicted_mesh.mesh.t.shape[1]}, " f"Vert: {predicted_mesh.mesh.p.shape[1]}"

    layout = get_layout(boundary=plot_boundary, title=title, mesh=predicted_mesh.mesh)
    learner_plot = go.Figure(data=traces, layout=layout)
    return learner_plot


def get_reference_plot(reference_mesh: MeshWrapper, fem_problem: FEMProblem | None, reference_name: str = "") -> go.Figure:
    """
    Generates a visualization of the reference mesh, including the sizing field, element sizes,
    and optionally the FEM solution.

    Args:
        reference_mesh (MeshWrapper):
            The reference mesh used for visualization.
        fem_problem (Optional[AbstractFEMProblem], optional):
            The finite element method (FEM) problem to compute the reference solution. If None,
            no solution plot is generated. Defaults to None.
        reference_name (str, optional):
            A name for the reference mesh, used for plot titles. Defaults to an empty string.

    Returns:
        go.Figure:
            A Plotly figure containing multiple traces:
            - The **sizing field** (default visible)
            - The **element size** (hidden by default)
            - The **FEM solution** (if provided, hidden by default)

    Notes:
        - The **sizing field** is displayed by default.
        - The **element size** and **FEM solution** are available in the legend but hidden initially.
    """
    traces = []

    # Compute and plot the sizing field (default visible)
    expert_sizing_field = get_sizing_field(reference_mesh)
    sizing_field_trace = scalar_contour_trace(
        mesh=reference_mesh.mesh,
        scalars=expert_sizing_field,
        trace_name=f"{reference_name} Sizing Field",
        logscale=True,
        visible=True,  # Default trace
    )
    traces.append(sizing_field_trace)

    # Plot element sizes (hidden by default)
    element_size_trace = scalar_contour_trace(
        mesh=reference_mesh.mesh,
        scalars=reference_mesh.simplex_volumes,
        trace_name=f"{reference_name} Element Size",
        logscale=True,
        visible="legendonly",  # Initially hidden
    )
    traces.append(element_size_trace)

    # Compute and plot the FEM solution if provided (hidden by default)
    if fem_problem is not None:
        expert_solution = fem_problem.calculate_solution(reference_mesh)
        solution_trace = scalar_contour_trace(
            mesh=reference_mesh.mesh,
            scalars=expert_solution,
            trace_name=f"{reference_name} Solution",
            visible="legendonly",  # Initially hidden
        )
        traces.append(solution_trace)

    traces.append(wireframe_trace(mesh=reference_mesh.mesh, name=f"{reference_name.title()} Mesh"))  # Mesh wireframe
    traces.append(vertex_trace(mesh=reference_mesh.mesh, name=f"{reference_name.title()} Vertices"))  # Mesh vertices

    # Define plot title
    title = f"{reference_name} Reference Mesh. " f"Elem: {reference_mesh.mesh.t.shape[1]}, " f"Vert: {reference_mesh.mesh.p.shape[1]}"

    # Generate Plotly figure
    layout = get_layout(boundary=None, title=title, mesh=reference_mesh.mesh)
    reference_plot = go.Figure(data=traces, layout=layout)

    return reference_plot
