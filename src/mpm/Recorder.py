import os
import numpy as np
from third_party.pyevtk.hl import pointsToVTK
from third_party.pyevtk.hl import gridToVTK


class Recorder:
    def __init__(self, save_path, visualize=True):
        self.save_path = save_path
        self.visualize = visualize

        self.particle_path = os.path.join(save_path, 'Particle')
        self.grid_path = os.path.join(save_path, 'Grid')
        self.vtk_path = os.path.join(save_path, 'VTK')

        os.makedirs(self.particle_path, exist_ok=True)
        os.makedirs(self.grid_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)

    # ================================================================
    # Main interface
    # ================================================================
    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)

    # ================================================================
    # Particle
    # ================================================================
    def save_particle(self, sims, scene):
        self.MonitorParticle(sims, scene)

    def MonitorParticle(self, sims, scene):
        particle = scene.particle
        npart = scene.particleNum

        pos = particle.position.to_numpy()
        vel = particle.velocity.to_numpy()
        vol = particle.volume.to_numpy()

        stress = particle.stress.to_numpy()
        vel_grad = particle.velocity_gradient.to_numpy()
        state_vars = particle.state_vars.to_numpy()

        output = {
            "position": pos,
            "velocity": vel,
            "volume": vol,
            "stress": stress,
            "velocity_gradient": vel_grad,
            "state_vars": state_vars,
        }

        if self.visualize:
            self.VisualizeParticle2D(
                sims,
                pos,
                vel,
                vol,
                state_vars
            )

        np.savez(
            self.particle_path + f'/MPMParticle{sims.current_print:06d}',
            **output
        )

    # ================================================================
    # 🔧 FIX REAL AQUI
    # ================================================================
    def VisualizeParticle2D(self, sims, position, velocity, volume, state_vars):
        posx = position[:, 0].astype(np.float64)
        posy = position[:, 1].astype(np.float64)

        # 🔥 CORREÇÃO CRÍTICA
        posz = np.zeros_like(posx, dtype=np.float64)

        velx = velocity[:, 0].astype(np.float64)
        vely = velocity[:, 1].astype(np.float64)
        velz = np.zeros_like(velx, dtype=np.float64)

        data = {
            "velocity": (velx, vely, velz),
            "volume": volume.astype(np.float64)
        }

        for i in range(state_vars.shape[1]):
            data[f"state_{i}"] = state_vars[:, i].astype(np.float64)

        pointsToVTK(
            self.vtk_path + f'/GraphicMPMParticle{sims.current_print:06d}',
            posx, posy, posz,
            data=data
        )

    # ================================================================
    # Grid
    # ================================================================
    def save_grid(self, sims, scene):
        grid = scene.grid
        coords = grid.coords.to_numpy()

        nx, ny = grid.node_num
        x = coords[:, 0].reshape(nx, ny).astype(np.float64)
        y = coords[:, 1].reshape(nx, ny).astype(np.float64)
        z = np.zeros_like(x, dtype=np.float64)

        gridToVTK(
            self.grid_path + f'/MPMGrid{sims.current_print:06d}',
            x, y, z
        )
