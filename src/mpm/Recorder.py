import os


class WriteFile:
    def __init__(self, sims, visualize=False):
        """
        Robust Recorder:
        - Accepts Simulation object
        - Extracts save_path safely
        - Works even if attributes change
        """

        self.sims = sims
        self.visualize = visualize

        # --- SAFE SAVE PATH RESOLUTION ---
        if hasattr(sims, "save_path"):
            save_path = sims.save_path
        elif hasattr(sims, "output_path"):
            save_path = sims.output_path
        else:
            raise RuntimeError(
                "Recorder error: Simulation object has no save_path attribute"
            )

        if not isinstance(save_path, (str, bytes, os.PathLike)):
            raise TypeError(
                f"Recorder error: save_path must be a string, got {type(save_path)}"
            )

        self.save_path = save_path

        # --- OUTPUT DIRECTORIES ---
        self.particle_path = os.path.join(self.save_path, "Particle")
        self.grid_path = os.path.join(self.save_path, "Grid")
        self.vtk_path = os.path.join(self.save_path, "VTK")

        os.makedirs(self.particle_path, exist_ok=True)
        os.makedirs(self.grid_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)

    # ------------------------------------------------------------------
    # PLACEHOLDERS (avoid runtime crashes even if not used)
    # ------------------------------------------------------------------
    def write_particle(self, *args, **kwargs):
        pass

    def write_grid(self, *args, **kwargs):
        pass

    def write_vtk(self, *args, **kwargs):
        pass

    def close(self):
        pass
