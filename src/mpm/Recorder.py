import os
import numpy as np
from src.mpm.Simulation import Simulation
from src.mpm.utils.Type import vec2f, vec3f
from pyevtk.hl import pointsToVTK


class WriteFile:
    def __init__(self, sims: Simulation, visualize=False):
        self.sims = sims
        self.visualize = visualize

        # Caminho correto vem do Simulation
        save_path = sims.save_path

        self.particle_path = os.path.join(save_path, 'Particle')
        self.grid_path = os.path.join(save_path, 'Grid')
        self.vtk_path = os.path.join(save_path, 'VTK')

        os.makedirs(self.particle_path, exist_ok=True)
        os.makedirs(self.grid_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)

    # ------------------------------------------------------------------
    def output(self, sims, scene):
        self.save_particle(sims, scene)

    # ------------------------------------------------------------------
    def save_particle(self, sims, scene):
        position = scene.particle.position.to_numpy()
        velocity = scene.particle.velocity.to_numpy()
        volume = scene.particle.volume.to_numpy()

        stress = scene.particle.stress.to_numpy()
        velocity_gradient = scene.particle.velocity_gradient.to_numpy()

        state_vars = {}
        if hasattr(scene.particle, 'state_vars'):
            for k, v in scene.particle.state_vars.items():
                state_vars[k] = v.to_numpy()

        output = {
            'position': position,
            'velocity': velocity,
            'volume': volume,
            'stress': stress,
            'velocity_gradient': velocity_gradient
        }
        output.update(state_vars)

        # ---- VTK ----
        self.visualizeParticle2D(
            sims,
            position,
            velocity,
            volume,
            state_vars
        )

        np.savez(
            self.particle_path + f'/MPMParticle{sims.current_print:06d}',
            **output
        )

    # ------------------------------------------------------------------
    def visualizeParticle2D(self, sims, position, velocity, volume, state_vars):
        posx = position[:, 0].astype(np.float64)
        posy = position[:, 1].astype(np.float64)
        posz = np.zeros_like(posx)

        velx = velocity[:, 0].astype(np.float64)
        vely = velocity[:, 1].astype(np.float64)
        velz = np.zeros_like(velx)

        data = {
            "velocity": (velx, vely, velz),
            "volume": volume.astype(np.float64)
        }

        for k, v in state_vars.items():
            data[k] = v.astype(np.float64)

        pointsToVTK(
            self.vtk_path + f'/GraphicMPMParticle{sims.current_print:06d}',
            posx, posy, posz,
            data=data
        )


# ---- símbolo exigido pelo loader ----
WriteFile = WriteFile
