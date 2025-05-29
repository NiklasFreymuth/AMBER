from typing import Union

import numpy as np
from skfem import adaptive_theta


def compute_gradient_at_centroid(element, node_gradients):
    # Compute the centroid of the triangle
    centroid = np.mean(node_gradients[:, element], axis=1)
    return centroid


def compute_element_gradients(mesh, u):
    from skfem import Basis, ElementTetP1, ElementTriP1

    if mesh.t.shape[0] == 4:  # 3d mesh, consisting of tetrahedra
        basis = Basis(mesh, ElementTetP1())
    else:  # 2d mesh, consisting of triangles
        basis = Basis(mesh, ElementTriP1())
    interpolated_solution = basis.interpolate(u)
    gradients = interpolated_solution.grad
    gradients = gradients.mean(-1).T / 2
    return gradients


def compute_node_gradients(mesh, original_gradients):
    node_gradients = np.zeros((*mesh.p.shape,))
    num_elems_at_node = np.zeros(mesh.p.shape[1])

    for pos, element in enumerate(mesh.t.T):
        grad = original_gradients[pos]
        for node in element:
            node_gradients[:, node] += grad
            num_elems_at_node[node] += 1

    node_gradients /= num_elems_at_node

    return node_gradients


class ZienkiewiczZhuErrorHeuristic:
    def __init__(self, theta: Union[float, int], refinement_strategy: str = "percentage"):
        """
        Performs mesh refinement based on the Zienkiewicz-Zhu error estimator. This is a heuristic
        that is based on the error estimator of the Zienkiewicz-Zhu error estimator. The error estimator
        is computed for each element and the elements with the largest error are refined.
        Currently implemented for linear triangular elements in Scikit FEM only.
        Args:
            theta: If the error is within theta*max_error, the element is not refined.
                If int: The number of elements to refine.
                If float: The fraction of elements to refine.
        """
        self._theta = theta
        self._refinement_strategy = refinement_strategy  # either "percentage" or "absolute"

    def _get_error_per_element(self, mesh, solution: np.ndarray) -> np.ndarray:
        original_gradients = compute_element_gradients(mesh, solution)
        node_gradients = compute_node_gradients(mesh, original_gradients)
        centroid_gradients = np.array([compute_gradient_at_centroid(e, node_gradients) for e in mesh.t.T])
        error_per_element = np.linalg.norm(original_gradients - centroid_gradients, axis=1) ** 2
        return error_per_element

    def __call__(self, mesh, solution: np.ndarray) -> np.ndarray:
        """

        Args:
            error_per_element: error per face. The error is the integrated
            L1 norm of the difference between the solution and the
            reference solution.

        Returns: actions to take. 1 means refine, -1 means do nothing

        """
        error_per_element = self._get_error_per_element(mesh=mesh, solution=solution)
        if isinstance(self._theta, float):
            assert 0 <= self._theta <= 1
            if self._refinement_strategy == "percentage":
                elements_to_refine = np.argsort(error_per_element)[::-1][
                    : int(len(error_per_element) * (1 - self._theta))
                ]
            elif self._refinement_strategy == "absolute":
                elements_to_refine = adaptive_theta(error_per_element, theta=self._theta)

        elif isinstance(self._theta, int):
            elements_to_refine = np.argsort(error_per_element)[::-1][: self._theta]
        else:
            raise ValueError(f"Theta must be float or int, but is {type(self._theta)}")
        actions = np.zeros_like(error_per_element)
        actions[elements_to_refine] = 1
        return actions

    def get_actions(self, mesh, solution: np.ndarray) -> np.ndarray:
        return self.__call__(mesh=mesh, solution=solution)


class ErrorRemeshingHeuristic:
    def __init__(self, theta: Union[float, int], area_scaling: bool = False):
        """

        Args:
            theta: If the error is within theta*max_error, the element is not refined.
                If int: The number of elements to refine.
                If float: The fraction of elements to refine.
            area_scaling: If true, the error is scaled by the area of the element.
        """
        self._theta = theta
        self._area_scaling = area_scaling

    def __call__(self, error_per_element: np.ndarray, element_volumes: np.ndarray) -> np.ndarray:
        """

        Args:
            error_per_element: error per face. The error is the integrated
            L1 norm of the difference between the solution and the
            reference solution.

        Returns: actions to take. 1 means refine, -1 means do nothing

        """
        if self._area_scaling:
            error_per_element = error_per_element / element_volumes
        if isinstance(self._theta, float):
            assert 0 <= self._theta <= 1
            elements_to_refine = adaptive_theta(error_per_element, theta=self._theta)
        elif isinstance(self._theta, int):
            error_per_element[element_volumes < 1.0e-6] = 0
            elements_to_refine = np.argsort(error_per_element)[::-1][: self._theta]
        else:
            raise ValueError(f"Theta must be float or int, but is {type(self._theta)}")
        actions = np.zeros_like(error_per_element)
        actions[elements_to_refine] = 1
        return actions

    def get_actions(self, error_per_element: np.ndarray, element_volumes) -> np.ndarray:
        return self.__call__(error_per_element=error_per_element, element_volumes=element_volumes)


if __name__ == "__main__":
    # set up a simple poisson problem
    import matplotlib.pyplot as plt
    from skfem import Basis, ElementTriP1, MeshTri, asm, enforce, solve
    from skfem.models import laplace, unit_load
    from skfem.visuals.matplotlib import draw, plot

    mesh = MeshTri.init_lshaped().refined(2)
    elem = ElementTriP1()
    zz_error = ZienkiewiczZhuErrorHeuristic(theta=0.8)

    x = None
    num_refinements = 5
    for i in range(num_refinements):
        print(i, mesh)
        basis = Basis(mesh, elem)
        A = asm(laplace, basis)
        b = asm(unit_load, basis)
        A, b = enforce(A, b, D=mesh.boundary_nodes())
        x = solve(A, b)
        action = zz_error(mesh, x)
        if i < num_refinements - 1:
            # do not refine in the last step to see the solution
            mesh = mesh.refined(np.argwhere(action > 0.0).flatten())
    ax = draw(mesh)
    plot(mesh, x, ax=ax, shading="gouraud", colorbar=True)
    plt.show()
