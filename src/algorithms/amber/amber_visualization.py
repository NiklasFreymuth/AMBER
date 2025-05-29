from typing import Dict, Optional

import numpy as np
from plotly import graph_objects as go

from src.algorithms.amber.amber_util import get_sizing_field
from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.environments.problems import AbstractFiniteElementProblem

from src.environments.util.mesh_visualization import plot_mesh


def get_learner_plots(
        learner_mesh: MeshWrapper,
        plot_boundary: np.ndarray,
        labels: Optional[np.ndarray] = None,
        solution: Optional[np.ndarray] = None,
        predicted_sizing_field: Optional[np.ndarray] = None,
        refinement_success: bool = True,
):
    """
    Get plots for the learner's mesh, labels, predictions and the solution.
    Args:
        learner_mesh:
        labels:
        predicted_sizing_field:
        solution:
        plot_boundary:
        refinement_success: If True, the learner's mesh can be refined. If False, the refined mesh would be too big

    Returns:

    """
    current_plots = {}

    # element_size_plot = plot_value_per_element(
    #     scalars=learner_mesh.get_simplex_volumes(),
    #     title=f"Element size. #Elements: {learner_mesh.t.shape[1]}",
    #     mesh=learner_mesh,
    #     boundary=plot_boundary,
    # )
    # current_plots |= {"element_size": element_size_plot}

    if solution is not None:
        solution_plot = plot_mesh(
            scalars=solution,
            title="Solution",
            mesh=learner_mesh,
            boundary=plot_boundary,
        )
        current_plots |= {f"solution": solution_plot}

    if labels is not None:
        label_plot = plot_mesh(
            scalars=labels,
            title=f"True sizing field. Total elements: {len(labels)}",
            mesh=learner_mesh,
            boundary=plot_boundary,
            logscale=True,
        )
        current_plots |= {f"label": label_plot}

    if predicted_sizing_field is not None:
        sizing_field_plot = plot_mesh(
            scalars=predicted_sizing_field,
            title=f"Pred. sizing field. Ref. success: {refinement_success}",
            mesh=learner_mesh,
            boundary=plot_boundary,
            logscale=True,
        )
        error_plot = plot_mesh(
            scalars=predicted_sizing_field - labels,
            title="Sizing field prediction error",
            mesh=learner_mesh,
            boundary=plot_boundary,
            colorscale="RdBu",
            symmetric=True,
        )
        sizing_field_diff_plot = plot_mesh(
            scalars=predicted_sizing_field - get_sizing_field(learner_mesh),
            title="Sizing field delta. Negative: Elements get smaller",
            mesh=learner_mesh,
            boundary=plot_boundary,
            colorscale="RdBu",
            symmetric=True,
        )
        current_plots |= {
            f"sizing_field": sizing_field_plot,
            f"error": error_plot,
            f"sizing_field_diff": sizing_field_diff_plot,
        }
    # fem_problem_plots = fem_problem.additional_plots()
    # current_plots = current_plots | prefix_keys(
    #     fem_problem_plots, prefix="fem_problem", separator="_"
    # )
    return current_plots


def get_reference_plots(
        reference_mesh: MeshWrapper, fem_problem: AbstractFiniteElementProblem, reference_name: str = ""
) -> Dict[str, go.Figure]:
    """
    Get plots for a reference mesh and solution.
    These plots are used to compare the reference solution to the learner's
    predictions in a visual way.
    Usually, the expert's solution is the ground truth, and only needs to be plotted once at the beginning of the
    training process.
    Args:
        reference_mesh:
        fem_problem:
        reference_name: Name of the reference mesh. Used for the title of the plot

    Returns:

    """
    expert_sizing_field = get_sizing_field(reference_mesh)
    expert_sizing_field_plot = plot_mesh(
        scalars=expert_sizing_field,
        title=f"{reference_name} sizing field",
        mesh=reference_mesh,
        boundary=fem_problem.plot_boundary,
        logscale=True,
    )

    expert_solution = fem_problem.calculate_solution(reference_mesh)
    expert_solution_plot = plot_mesh(
        scalars=expert_solution,
        title=f"{reference_name} solution",
        mesh=reference_mesh,
        boundary=fem_problem.plot_boundary,
    )
    element_size_plot = plot_mesh(
        scalars=reference_mesh.get_simplex_volumes(),
        title=f"{reference_name} element size. #Elements: {reference_mesh.t.shape[1]}",
        mesh=reference_mesh,
        boundary=fem_problem.plot_boundary,
    )
    expert_plots = {
        "sizing_field": expert_sizing_field_plot,
        "solution": expert_solution_plot,
        "element_size": element_size_plot,
    }
    return expert_plots
