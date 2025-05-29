import os
from typing import Tuple

from src.algorithm.dataloader.source_data import SourceData
from src.mesh_util.load_mesh import load_expert_mesh
from src.tasks.dataset_preparator import DatasetPreparator
from src.tasks.domains import ExtendedMeshTet1, ExtendedMeshTri1
from src.tasks.domains.gmsh_util import geom_fn_from_file, get_bounding_box
from src.tasks.domains.mesh_extension_mixin import MeshExtensionMixin
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.inlet_feature_provider import InletFeatureProvider

DATASET_ROOT_PATH = "data"


def mesh_from_geometry_fn(geometry_fn: callable, max_initial_element_volume: float, dim: int = 3) -> ExtendedMeshTri1 | ExtendedMeshTet1:
    """
    Generate an initial mesh from a geometry-producing function.

    The geometry is defined by `geometry_fn`, which returns a pygmsh-compatible geometry object.
    Only 3D tetrahedral meshing is currently supported.

    Args:
        geometry_fn (callable): A function returning a geometry object (e.g., based on a STEP file).
        max_initial_element_volume (float): Maximum volume allowed for initial mesh elements.
        dim (int, optional): Spatial dimension of the mesh. Only dim=3 is supported. Defaults to 3.

    Returns:
        MeshExtensionMixin: Generated mesh object using MeshExtensionMixin.

    Raises:
        NotImplementedError: If `dim` is set to 2 (2D meshing is not supported).
    """
    if dim == 2:
        mesh_cls = ExtendedMeshTri1
    else:  # dim == 3
        mesh_cls = ExtendedMeshTet1
    mesh = mesh_cls.init_from_geom_fn(
        geom_fn=geometry_fn,
        max_element_volume=max_initial_element_volume,
    )
    assert isinstance(mesh, (ExtendedMeshTri1, ExtendedMeshTet1)), "Mesh is not of the expected type."
    return mesh


class ExpertGeometryDatasetPreparator(DatasetPreparator):
    """
    Loads a dataset from expert annotations on a geometry
    """

    @property
    def dataset_path(self):
        dataset_name = self.task_config.name
        return os.path.join(DATASET_ROOT_PATH, dataset_name)

    def _prepare_source_and_mesh(self, data_idx: int, dataset_mode: str) -> Tuple[SourceData, MeshWrapper]:
        data_point_path = os.path.join(str(self.dataset_path), dataset_mode, f"{data_idx + 1:03d}")

        # get geometry and information on how to mesh
        geometry_fn = self._load_geometry_fn(geometry_path=data_point_path)
        bounding_box = get_bounding_box(geometry_fn=geometry_fn)
        dimension = len(bounding_box) // 2  # either 2d or 3d geometries
        max_initial_element_volume = self.max_initial_element_volume(bounding_box=bounding_box, dimension=dimension)

        # get expert mesh
        expert_mesh = self._load_expert_mesh(expert_mesh_path=data_point_path)
        expert_mesh.geom_fn = geometry_fn
        expert_mesh = MeshWrapper(expert_mesh)

        # get initial (coarse, uniform) mesh
        initial_mesh = mesh_from_geometry_fn(geometry_fn=geometry_fn, max_initial_element_volume=max_initial_element_volume, dim=dimension)
        initial_mesh = MeshWrapper(initial_mesh)

        # todo: Refactor.
        if "mold" in self.task_config.name:
            from src.helpers.qol import filter_included_fields

            inlet_file = data_point_path + "_features.txt"
            observation_features = filter_included_fields(self.task_config.inlet_features)
            feature_provider = InletFeatureProvider(inlet_file=inlet_file, observation_features=observation_features)

        else:
            feature_provider = None

        source_data = SourceData(expert_mesh=expert_mesh, initial_mesh=initial_mesh, feature_provider=feature_provider)
        return source_data, initial_mesh

    def _load_expert_mesh(self, *, expert_mesh_path: str) -> ExtendedMeshTri1 | ExtendedMeshTet1:
        extension = self.task_config.extensions.mesh
        return load_expert_mesh(expert_mesh_path=expert_mesh_path, extension=extension)

    def _load_geometry_fn(self, *, geometry_path: str) -> callable:
        extension = self.task_config.extensions.geometry
        if extension is not None and not geometry_path.endswith(f".{extension}"):
            geometry_path = f"{geometry_path}.{extension}"
        geom_fn = geom_fn_from_file(file_path=geometry_path)
        return geom_fn
