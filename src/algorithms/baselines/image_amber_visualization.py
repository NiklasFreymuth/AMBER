from typing import List, Dict, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.algorithms.baselines.mesh_to_image import MeshImage
from src.environments.util.mesh_visualization import get_3d_scatter_trace
from util.torch_util.torch_util import detach


def get_feature_image_plots(feature_names: List[str], mesh_image: MeshImage, ) -> Dict[str, go.Figure]:
    """
    Plots the pixel features and labels as images. Plots a separate image for each feature and one for the labels
    Args:
        feature_names: Names of the features. Must have length num_features
        mesh_image: MeshImage object containing the pixel features and labels. Contains
            pixel_features: Pixel features. Has shape (1, num_features, {image_resolution}^dim)

    Returns:

    """
    pixel_features = detach(mesh_image.features)
    assert pixel_features.ndim in [4, 5], \
        f"Expected pixel_features to have shape (1, num_features, image_resolution^dim), but got {pixel_features.shape}"
    assert pixel_features.shape[0] == 1, \
        f"Expected pixel_features to have shape (1, num_features, image_resolution^dim), but got {pixel_features.shape}"
    assert len(feature_names) == pixel_features.shape[1], \
        f"Expected feature_names to have length num_features. Got {len(feature_names)} and {pixel_features.shape[1]}"

    if pixel_features.ndim == 4:  # 2d meshes
        figures = {
            feature_name: px.imshow(feature, color_continuous_scale='Jet', title=feature_name.title())
            for feature_name, feature in zip(feature_names, pixel_features[0])
        }
    else:  # 3d meshes
        pixel_features = pixel_features[0]
        pixel_features = pixel_features.reshape(pixel_features.shape[0], -1)

        figures = {
            feature_name: _get_3d_scatter(mesh_image, feature, feature_name)
            for feature_name, feature in zip(feature_names, pixel_features)
        }

    return figures


def get_prediction_image_plots(mesh_image: MeshImage,
                               prediction_grid: Optional[np.ndarray] = None,
                               ) -> Dict[str, go.Figure]:
    """
    Plots the labels, predictions and their differences as separate images.
    Args:
        mesh_image: MeshImage object containing the pixel features and labels. Contains
            is_mesh: Flattened boolean array indicating whether each pixel is part of the mesh.
            label_grid: Grid of labels. Has shape (1, image_resolution^dim). 0 outside the domain
        prediction_grid: Grid of predictions. Has shape (1, image_resolution^dim). 0 outside the domain

    Returns:

    """
    figures = {}
    is_mesh = mesh_image.is_mesh
    label_grid = detach(mesh_image.label_grid)

    pixel_grids = {"Labels": label_grid}
    symmetric = {"Labels": False}
    colorscale = {"Labels": "Jet"}
    if prediction_grid is not None:
        pixel_grids["Predictions"] = prediction_grid
        pixel_grids["Differences"] = label_grid - prediction_grid
        symmetric["Predictions"] = False
        symmetric["Differences"] = True
        colorscale["Predictions"] = "Jet"
        colorscale["Differences"] = "RdBu"

    for key, pixel_grid in pixel_grids.items():
        assert pixel_grid.ndim in [3, 4], \
            f"Expected {key} to have shape (1, image_resolution^dim), but got {label_grid.shape}"
        assert pixel_grid.shape[0] == 1, \
            f"Expected {key} to have shape (1, image_resolution^dim), but got {label_grid.shape}"

        if pixel_grid.ndim == 3:  # 2d image
            pixel_grid[~is_mesh.reshape(label_grid.shape)] = np.nan  # Set non-mesh elements to NaN for visualization

            if symmetric[key]:
                cmax = np.nanmax(np.abs(pixel_grid))
                fig = px.imshow(pixel_grid[0], color_continuous_scale="RdBu",
                                zmin=-cmax, zmax=cmax, title=key)
            else:
                pixel_grid, colorbar = _to_log_scale(pixel_grid)
                fig = px.imshow(pixel_grid[0], color_continuous_scale='Jet', title=key)
                fig.update_layout(coloraxis_colorbar=colorbar)
            figures[key] = fig

        else:  # 3d image
            pixel_grid = pixel_grid.reshape(-1)
            figures[key] = _get_3d_scatter(mesh_image=mesh_image, feature_map=pixel_grid,
                                           name=key, symmetric=symmetric[key], colorscale=colorscale[key])

    return figures


def _to_log_scale(scalars: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply logarithm. Assumes scalars are all positive. Scalars may be 0
    Args:
        scalars:

    Returns:

    """
    adjusted_scalars = np.where(np.isnan(scalars), np.nan, np.log(scalars + 1e-12))
    cmin, cmax = np.nanmin(adjusted_scalars), np.nanmax(adjusted_scalars)

    # Generate tick values evenly distributed across the log-transformed scale
    # Map these tick values back to the original scale for the labels. Round for nicer display
    tickvals = np.linspace(cmin, cmax, num=5)
    ticktext = np.round(np.exp(tickvals), 4)
    colorbar = {
        "tickvals": tickvals,
        "ticktext": ticktext
    }
    return adjusted_scalars, colorbar


def _get_3d_scatter(mesh_image: MeshImage, feature_map: np.ndarray, name: str,
                    colorscale: Optional[str] = "Jet",
                    symmetric: Optional[bool] = False) -> go.Figure:
    """
    Plots the provided feature and labels as a 3d scatter plot.
    Args:
        mesh_image:
        feature_map: Shape (num_pixels,)
        name: List of names to use for the plots
        symmetric: If True, the color scale is symmetric around 0

    Returns:

    """
    points = mesh_image.feature_coordinates
    is_mesh = mesh_image.is_mesh
    points = points[:, is_mesh]
    feature_map = feature_map[is_mesh]
    figure = go.Figure(get_3d_scatter_trace(positions=points,
                                            scalars=feature_map,
                                            size=3,
                                            colorscale=colorscale,
                                            colorbar_title=name.title(),
                                            symmetric=symmetric))
    return figure
