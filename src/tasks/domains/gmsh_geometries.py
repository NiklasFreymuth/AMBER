import numpy as np
import pygmsh


def square_hole_geom(hole_position: np.array, hole_size: np.array, length: float = 1.0) -> pygmsh.occ.Geometry:
    # Manually create a geometry instance
    geom = pygmsh.occ.Geometry()

    # Add rectangle and disk (hole) to the geometry
    rectangle = geom.add_rectangle([0.0, 0.0, 0.0], a=length, b=length)

    # Perform boolean difference
    hole = geom.add_disk([hole_position[0], hole_position[1], 0.0], hole_size)
    geom.boolean_difference(rectangle, hole)
    return geom


def sloped_l_geom(
    hole_position: np.array,
    hole_size: float,
    length: float = 1.0,
    cutout_length: float = 0.5,
) -> pygmsh.occ.Geometry:
    geom = pygmsh.occ.Geometry()

    # Create the outer rectangle (large part of the L-shape)
    outer_rectangle = geom.add_rectangle([0.0, 0.0, 0.0], a=length, b=length)

    # Create the inner rectangle (cutout to form the L-shape)
    # Assuming the cutout is from the top right corner, adjust as needed
    cutout_rectangle = geom.add_rectangle([length - cutout_length, 0.0, 0.0], a=cutout_length, b=length - cutout_length)

    # Subtract the inner rectangle from the outer to get the L-shape
    l_shape = geom.boolean_difference([outer_rectangle], [cutout_rectangle], delete_first=True, delete_other=True)

    # Add the disk to the geometry
    disk = geom.add_disk([hole_position[0], hole_position[1], 0.0], hole_size)

    # blend the disk with the L-shape
    final_shape = geom.boolean_union([l_shape, disk])

    return geom


def polygon_geom(
    boundary_nodes: np.array,
) -> pygmsh.occ.Geometry:
    """
    Creates a mesh from a polygon defined by its boundary nodes
    Args:
        boundary_nodes: Boundary nodes of the polygon. Has shape (num_nodes, 2)

    Returns: A mesh of the given class

    """
    geom = pygmsh.occ.Geometry()
    geom.add_polygon(boundary_nodes)
    return geom


def lattice_geom(
    n_holes: int,
    hole_size: float,
    domain_size: float = 1.0,
) -> pygmsh.occ.Geometry:
    """
    Create a 2D square domain with a regular lattice of square cutouts.

    The geometry consists of:
      - One large outer square domain.
      - A centered grid of smaller square holes, evenly spaced.

    Spacing is automatically calculated so that holes and margins are uniform:
        spacing = (domain_size - (n_holes * hole_size)) / (n_holes + 1)

    ASCII illustration (top view):

          +---------------------------------+
          |   □    □    □    □    □    □    |
          |                                 |
          |   □    □    □    □    □    □    |
          |                                 |
          |   □    □    □    □    □    □    |
          |                                 |
          |   □    □    □    □    □    □    |
          |                                 |
          |   □    □    □    □    □    □    |
          |                                 |
          |   □    □    □    □    □    □    |
          +---------------------------------+

    Each '□' represents a square cutout, arranged in a regular lattice.

    Parameters:
        domain_size (float): Length of the outer square domain side.
        n_holes (int): Number of cutouts along one axis (total holes = n_holes²).
        hole_size (float): Side length of each square cutout.

    Returns:
        pygmsh.occ.Geometry: The geometry object, ready for meshing.

    Notes:
        - Holes are uniformly spaced and centered.
        - spacing = (domain_size - n_holes * hole_size) / (n_holes + 1)
        - Ensure domain_size > n_holes * hole_size for feasible geometry.
    """
    geom = pygmsh.occ.Geometry()

    # Outer domain square
    outer = geom.add_rectangle([0.0, 0.0, 0.0], a=domain_size, b=domain_size)

    holes = []
    spacing = (domain_size - (n_holes * hole_size)) / (n_holes + 1)
    assert spacing > 0, f"Cutouts may not overlap, given {spacing=} from {hole_size=} and {n_holes=}"

    # Add hole squares
    for i in range(n_holes):
        for j in range(n_holes):
            x0 = spacing + i * (hole_size + spacing)
            y0 = spacing + j * (hole_size + spacing)
            hole = geom.add_rectangle([x0, y0, 0.0], a=hole_size, b=hole_size)
            holes.append(hole)

    # Subtract holes from outer domain
    geom.boolean_difference([outer], holes)

    return geom
