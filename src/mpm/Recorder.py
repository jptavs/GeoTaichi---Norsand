import os
import numpy as np
from third_party.pyevtk.hl import pointsToVTK, gridToVTK


class WriteFile:

    def __init__(self, save_path, visualize=True):
        self.save_path = save_path
        self.visualize = visualize

        self.particle_path = os.path.join(save_path, 'Particle')
        self.grid_path = os.path.join(save_path, 'Grid')
        self.vtk_path = os.path.join(save_path, 'VTK')

        os.makedirs(self.particle_path, exist_ok=True)
        os.makedirs(self.grid_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)

    # ==========================================================
    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)

    # ==========================================================
    def save_particle(self, sims, scene):
        self.MonitorParticle(sims, scene)

    def MonitorParticle(self, sims, scene):
        particle = scene.particle

        position = particle.position.to_numpy()
        velocity = particle.velocity.to_numpy()
        volume = particle.volume.to_numpy()

        stress = particle.stress.to_numpy()
        velocity_gradient = particle.velocity_gradient.to_numpy()
        state_vars = particle.state_vars.to_numpy()

        if self.visualize:
            self.VisualizeParticle2D(
                sims, position, velocity, volume, state_vars
            )

        np.savez(
            self.particle_path + f'/MPMParticle{sims.current_print:06d}',
            position=position,
            velocity=velocity,
            volume=volume,
            stress=stress,
            velocity_gradient=velocity_gradient,
            state_vars=state_vars
        )

    # ==========================================================
    def VisualizeParticle2D(self, sims, position, velocity, volume, state_vars):

        posx = position[:, 0].astype(np.float64)
        posy = position[:, 1].astype(np.float64)
        posz = np.zeros_like(posx, dtype=np.float64)

        velx = velocity[:, 0].astype(np.float64)
        vely = velocity[:, 1].astype(np.float64)
        velz = np.zeros_like(velx, dtype=np.float64)

        data = {
            "velocity": (velx, vely, velz),
            "volume": volume.astype(np.float64)
        }

        # Estado interno do NorSand: escrever como campos escalares
        for i in range(state_vars.shape[1]):
            data[f"state_{i}"] = state_vars[:, i].astype(np.float64)

        pointsToVTK(
            self.vtk_path + f'/GraphicMPMParticle{sims.current_print:06d}',
            posx, posy, posz,
            data=data
        )

    # ==========================================================
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
