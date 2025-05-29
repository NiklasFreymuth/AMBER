import numpy as np
from skfem import Basis, Functional, InteriorFacetBasis
from skfem.helpers import grad

from src.helpers.qol import wrapped_partial
from src.tasks.features.fem.elliptic_problem.load_function.gmm_load import GMMDensity


def get_edge_residual(basis: Basis, solution: np.ndarray):
    """
    Calculates the error/residual for the edge of the mesh. This error is given as the square of the jump of the gradient
    of the solution over the edge.
    This corresponds to h||[[\delta u \cdot \mathbf{n}]]||^2, where \delta u is the jump of the gradient of the solution
    and \mathbf{n} is the normal vector of the edge.
    Args:
        basis:
        solution:

    Returns:

    """
    # facet jump
    mesh = basis.mesh
    element = basis.elem
    fbasis = [InteriorFacetBasis(mesh, element, side=i) for i in [0, 1]]
    # the interior facet basis computes the integral over each element
    w = {f"u{i + 1}": fbasis[i].interpolate(solution) for i in [0, 1]}
    eta_E = edge_jump.elemental(fbasis[0], **w)  # calculates error on the boundary based on w.n, w.u1 and w.u2
    tmp = np.zeros(mesh.facets.shape[1])

    # add all indices without buffering, i.e., for a[0,0,1], add something to the element 0 twice
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(0.5 * tmp[mesh.t2f], axis=0)
    return eta_E


def get_interior_residual(basis: Basis, load: GMMDensity) -> np.ndarray:
    """
    Calculates the residual/error for the interior of the mesh for a Poisson problem. This interior residual is given by
    the square of the load function times the square of the element size, i.e., h^2 * ||f^2||.
    Args:
        basis: A scikit-FEM Basis instance that consists of a mesh and an element.
          Must use a triangle mesh and linear triangular elements.
        load: A AbstractTargetFunction instance that defines the GMM model that specifies the load

    Returns:

    """
    interior_residual_ = wrapped_partial(_interior_residual, load=load)

    eta_K = Functional(interior_residual_).elemental(basis)
    # .elemental evaluates element-wise
    return eta_K


def _interior_residual(w, load: GMMDensity):
    """
    Provides a closed-form solution for the interior residual
    May be arbitrarily wrong for small meshes and large coefficients :(
    Args:
        w:
        load: A AbstractTargetFunction instance that defines the GMM model that specifies the load
    Returns:

    """
    h = w.h
    x, y = w.x
    positions = np.stack((x, y), axis=-1)
    return (h**2) * (load.evaluate(positions) ** 2)


@Functional
def edge_jump(w):
    h = w.h
    n = w.n
    dw1 = grad(w["u1"])  # short for w['u1'].grad
    dw2 = grad(w["u2"])
    return h * ((dw1[0] - dw2[0]) * n[0] + (dw1[1] - dw2[1]) * n[1]) ** 2
