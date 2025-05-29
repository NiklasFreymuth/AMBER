from typing import Dict, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

MetricDict = Dict[str, float]  # Scalar metrics
PlotDict = Dict[str, go.Figure]  # Named plots
SizingFieldInterpolationType = Literal[
    "interpolated_vertex",
    "sampled_vertex",
    "element_weighted_sum",
    "pixel",
]
MeshNodeType = Literal["element", "vertex", "pixel"]
SizingField = NDArray[np.float64]
DatasetMode = Literal["train", "val", "test"]
MeshGenerationStatus = Literal["success", "failed", "scaled"]
