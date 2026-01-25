import os

class WriteFile:
    def __init__(self, sims, visualize=False):
        """
        Robust Recorder:
        - Accepts Simulation object or save_path (str)
        - Fully backward compatible
        """

        self.visualize = visualize

        # --- CASE 1: Simulation object (CORRECT for GeoTaichi) ---
        if hasattr(sims, "save_path"):
            save_path = sims.save_path

        # --- CASE 2: string path (legacy / standalone use) ---
        elif isinstance(sims, (str, bytes, os.PathLike)):
            save_path = sims

        else:
            raise TypeError(
                f"WriteFile expected Simulation or path-like, got {type(sims)}"
            )

        # --- Normalize path ---
        self.save_path = str(save_path)

        # --- Output directories ---
        self.particle_path = os.path.join(self.save_path, "Particle")
        self.grid_path     = os.path.join(self.save_path, "Grid")
        self.vtk_path      = os.path.join(self.save_path, "VTK")

        # --- Create folders safely ---
        for p in [self.save_path, self.particle_path, self.grid_path, self.vtk_path]:
            os.makedirs(p, exist_ok=True)

    # ----------------------------------------------------------
    # Dummy-safe methods (avoid crash even if not used)
    # ----------------------------------------------------------
    def write_particle(self, *args, **kwargs):
        pass

    def write_grid(self, *args, **kwargs):
        pass

    def write_vtk(self, *args, **kwargs):
        pass

    def finalize(self):
        pass
