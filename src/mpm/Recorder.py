import os
import numpy as np

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.utils.linalg import no_operation
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK


class Recorder:
    def __init__(self, sims):
        self.vtk_path = None
        self.particle_path = None
        self.grid_path = None

        self.save_particle = no_operation
        self.save_grid = no_operation

        self.mkdir(sims)
        self.manage_function(sims)

    # -------------------------------------------------------------
    def manage_function(self, sims: Simulation):
        if 'particle' in sims.monitor_type:
            self.visualizeParticle = no_operation
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeParticle = self.VisualizeParticle2D
                else:
                    self.visualizeParticle = self.VisualizeParticle

            self.save_particle = self.MonitorParticle

        if 'grid' in sims.monitor_type:
            self.visualizeGrid = no_operation
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeGrid = self.VisualizeGrid2D
                else:
                    self.visualizeGrid = self.VisualizeGrid
            self.save_grid = self.MonitorGrid

        self.visualizedObject = no_operation
        if 'object' in sims.monitor_type:
            self.visualizedObject = self.VisualizeObject2D

    # -------------------------------------------------------------
    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)
        self.visualizedObject(sims, scene)

    # -------------------------------------------------------------
    def mkdir(self, sims: Simulation):
        os.makedirs(sims.path, exist_ok=True)

        self.particle_path = sims.path + '/particles'
        self.vtk_path = sims.path + '/vtks'
        self.grid_path = sims.path + '/grids'

        os.makedirs(self.particle_path, exist_ok=True)
        os.makedirs(self.vtk_path, exist_ok=True)
        os.makedirs(self.grid_path, exist_ok=True)

    # =============================================================
    # ======================= VISUALIZAÇÃO ========================
    # =============================================================

    def VisualizeObject2D(self, sims: Simulation, scene: myScene):
        polygon = scene.contact.polygon_vertices.to_numpy()
        polygon = np.ascontiguousarray(polygon)

        dtype = polygon.dtype

        posx = polygon[:, 0]
        posy = polygon[:, 1]
        posz = np.zeros(posx.shape[0], dtype=dtype)

        connectivity = np.arange(polygon.shape[0], dtype=np.int32)
        offsets = np.array([polygon.shape[0]], dtype=np.int32)
        cell_types = np.array([7], dtype=np.uint8)

        unstructuredGridToVTK(
            self.vtk_path + f'/GraphicObject{sims.current_print:06d}',
            posx, posy, posz,
            connectivity, offsets, cell_types
        )

    # -------------------------------------------------------------
    def VisualizeParticle(self, sims, position, velocity, volume, state_vars):
        position = np.ascontiguousarray(position)
        velocity = np.ascontiguousarray(velocity)
        volume = np.ascontiguousarray(volume)

        posx = position[:, 0]
        posy = position[:, 1]
        posz = position[:, 2]

        velx = velocity[:, 0]
        vely = velocity[:, 1]
        velz = velocity[:, 2]

        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)

        pointsToVTK(
            self.vtk_path + f'/GraphicMPMParticle{sims.current_print:06d}',
            posx, posy, posz,
            data=data
        )

    # -------------------------------------------------------------
    def VisualizeParticle2D(self, sims, position, velocity, volume, state_vars):
        position = np.ascontiguousarray(position)
        velocity = np.ascontiguousarray(velocity)
        volume = np.ascontiguousarray(volume)

        dtype = position.dtype

        posx = position[:, 0]
        posy = position[:, 1]
        posz = np.zeros(posx.shape[0], dtype=dtype)

        velx = velocity[:, 0]
        vely = velocity[:, 1]
        velz = np.zeros(velx.shape[0], dtype=dtype)

        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)

        pointsToVTK(
            self.vtk_path + f'/GraphicMPMParticle{sims.current_print:06d}',
            posx, posy, posz,
            data=data
        )

    # -------------------------------------------------------------
    def VisualizeGrid(self, sims, coords, point_data={}, cell_data={}):
        coords = np.ascontiguousarray(coords)

        coordx = np.unique(coords[:, 0])
        coordy = np.unique(coords[:, 1])
        coordz = np.unique(coords[:, 2])

        gridToVTK(
            self.vtk_path + f'/GraphicMPMGrid{sims.current_print:06d}',
            coordx, coordy, coordz,
            pointData=point_data,
            cellData=cell_data
        )

    # -------------------------------------------------------------
    def VisualizeGrid2D(self, sims, coords, point_data={}, cell_data={}):
        coords = np.ascontiguousarray(coords)
        dtype = coords.dtype

        coordx = np.unique(coords[:, 0])
        coordy = np.unique(coords[:, 1])
        coordz = np.zeros(1, dtype=dtype)

        gridToVTK(
            self.vtk_path + f'/GraphicMPMGrid{sims.current_print:06d}',
            coordx, coordy, coordz,
            pointData=point_data,
            cellData=cell_data
        )

    # =============================================================
    # ======================== MONITOR ============================
    # =============================================================

    def MonitorParticle(self, sims: Simulation, scene: myScene):
        n = scene.particleNum[0]

        position = scene.particle.x.to_numpy()[:n]
        velocity = scene.particle.v.to_numpy()[:n]
        volume = scene.particle.vol.to_numpy()[:n]

        state_vars = scene.material.get_state_vars_dict(0, n)

        self.visualizeParticle(sims, position, velocity, volume, state_vars)

        np.savez(
            self.particle_path + f'/MPMParticle{sims.current_print:06d}',
            position=position,
            velocity=velocity,
            volume=volume,
            state_vars=state_vars,
            t_current=sims.current_time
        )

    def MonitorGrid(self, sims: Simulation, scene: myScene):
        coords = scene.element.mesh.nodal_coords
        self.visualizeGrid(sims, coords)

        np.savez(
            self.grid_path + f'/MPMGrid{sims.current_print:06d}',
            coords=coords,
            t_current=sims.current_time
        )
