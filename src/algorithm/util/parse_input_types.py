from src.helpers.custom_types import MeshNodeType, SizingFieldInterpolationType


def get_mesh_node_type(interpolation_type: SizingFieldInterpolationType) -> MeshNodeType:
    if "vertex" in interpolation_type:
        # do sizing field estimation on the vertices
        return "vertex"
    elif "pixel" in interpolation_type:
        return "pixel"
    elif "element" in interpolation_type:
        return "element"
    else:
        raise ValueError(f"Unknown interpolation type '{interpolation_type}'")
