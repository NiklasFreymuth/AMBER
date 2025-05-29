from typing import Dict, Optional

import gmsh


class gmsh_session:
    """
    A context manager for a Gmsh session. This ensures that the Gmsh session is properly initialized and finalized.
    """

    def __init__(self, gmsh_kwargs: Optional[Dict] = None, verbose: bool = False):
        if gmsh_kwargs is None:
            gmsh_kwargs = {}
        self.gmsh_kwargs = gmsh_kwargs
        self._verbose = verbose

    def __enter__(self):
        gmsh.initialize()

        # Set default options
        gmsh.option.setNumber("General.Terminal", int(self._verbose))  # turn off info messages/prints
        gmsh.option.setNumber("General.Verbosity", int(self._verbose))  # turn off info messages/prints
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_kwargs.get("min_sizing_field", 1e-6))
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.gmsh_kwargs.get("max_sizing_field", 10))

        # Set up a default model. We only ever use one model at a time, so the name "model" is fine.
        gmsh.model.add("model")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finalize the Gmsh session
        gmsh.model.remove()
        gmsh.finalize()

        # Handle exceptions if necessary
        if exc_type:
            print(f"An exception occurred: {exc_val}")
        # Return False to propagate exceptions, True to suppress them
        return False
