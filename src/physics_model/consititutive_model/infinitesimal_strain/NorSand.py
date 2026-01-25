import taichi as ti
import math

from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import (
    InfinitesimalStrainModel
)
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import Sigrot
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class NorSandModel(InfinitesimalStrainModel):

    def __init__(self, material_type, configuration,
                 solver_type="Explicit",
                 stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)

        # Elasticidade
        self.G = 0.0
        self.K = 0.0
        self.nu = 0.0

        # Parâmetros NorSand
        self.Gamma = 0.0
        self.lambda_csl = 0.0
        self.M = 0.0
        self.chi = 0.0
        self.H = 0.0
        self.e0 = 0.0

        self.stress_integration = stress_integration

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------
    def model_initialize(self, parameter):

        self.density = DictIO.GetEssential(parameter, 'Density')
        self.young = DictIO.GetEssential(parameter, 'YoungModulus')
        self.nu = DictIO.GetEssential(parameter, 'PoissonRatio')

        self.Gamma = DictIO.GetAlternative(parameter, 'Gamma', 1.10)
        self.lambda_csl = DictIO.GetAlternative(parameter, 'LambdaCSL', 0.05)
        self.M = DictIO.GetAlternative(parameter, 'M', 1.25)
        self.chi = DictIO.GetAlternative(parameter, 'Chi', 3.5)
        self.H = DictIO.GetAlternative(parameter, 'HardeningH', 100.0)
        self.e0 = DictIO.GetAlternative(parameter, 'InitialVoidRatio', 0.65)

        self.G = self.young / (2.0 * (1.0 + self.nu))
        self.K = self.young / (3.0 * (1.0 - 2.0 * self.nu))

    # ------------------------------------------------------------------
    # Variáveis de estado (⚠️ TUDO float32)
    # ------------------------------------------------------------------
    def define_state_vars(self):
        return {
            "stress": ti.types.vector(6, ti.f32),
            "void_ratio": ti.f32,
            "p": ti.f32,
            "q": ti.f32
        }

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        stateVars[np].void_ratio = ti.cast(self.e0, ti.f32)
        stateVars[np].p = ti.cast(1.0, ti.f32)
        stateVars[np].q = ti.cast(0.0, ti.f32)

    # ------------------------------------------------------------------
    # Stress update 2D
    # ------------------------------------------------------------------
    @ti.func
    def ComputeStress2D(self, np, previous_stress, dt, particle, stateVars):

        L = particle[np].L

        de_xx = L[0, 0] * dt
        de_yy = L[1, 1] * dt
        de_xy = 0.5 * (L[0, 1] + L[1, 0]) * dt
        dw_xy = 0.5 * (L[0, 1] - L[1, 0]) * dt

        strain_increment = ti.Vector([
            de_xx,
            de_yy,
            0.0,
            2.0 * de_xy,
            0.0,
            0.0
        ], ti.f32)

        vorticity_increment = ti.Vector([0.0, 0.0, dw_xy], ti.f32)

        return self.core(
            np,
            previous_stress,
            strain_increment,
            vorticity_increment,
            stateVars
        )

    # ------------------------------------------------------------------
    # Stress update 3D
    # ------------------------------------------------------------------
    @ti.func
    def ComputeStress3D(self, np, previous_stress, dt, particle, stateVars):

        L = particle[np].L

        de_xx = L[0, 0] * dt
        de_yy = L[1, 1] * dt
        de_zz = L[2, 2] * dt

        de_xy = 0.5 * (L[0, 1] + L[1, 0]) * dt
        de_yz = 0.5 * (L[1, 2] + L[2, 1]) * dt
        de_xz = 0.5 * (L[0, 2] + L[2, 0]) * dt

        dw_yz = 0.5 * (L[1, 2] - L[2, 1]) * dt
        dw_zx = 0.5 * (L[2, 0] - L[0, 2]) * dt
        dw_xy = 0.5 * (L[0, 1] - L[1, 0]) * dt

        strain_increment = ti.Vector([
            de_xx,
            de_yy,
            de_zz,
            2.0 * de_xy,
            2.0 * de_yz,
            2.0 * de_xz
        ], ti.f32)

        vorticity_increment = ti.Vector([dw_yz, dw_zx, dw_xy], ti.f32)

        return self.core(
            np,
            previous_stress,
            strain_increment,
            vorticity_increment,
            stateVars
        )

    # ------------------------------------------------------------------
    # Núcleo constitutivo
    # ------------------------------------------------------------------
    @ti.func
    def core(self, np, previous_stress, strain_increment,
             vorticity_increment, stateVars):

        de = strain_increment

        d_vol = de[0] + de[1] + de[2]
        d_dev = de - (d_vol / 3.0) * ti.Vector(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], ti.f32
        )

        d_sigma_vol = self.K * d_vol
        d_sigma_dev = 2.0 * self.G * d_dev

        stress_rot = Sigrot(previous_stress, vorticity_increment)

        stress_new = (
            previous_stress
            + stress_rot
            + d_sigma_dev
            + d_sigma_vol * ti.Vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], ti.f32)
        )

        p_current = -(stress_new[0] + stress_new[1] + stress_new[2]) / 3.0
        stateVars[np].p = p_current
        stateVars[np].void_ratio += (1.0 + stateVars[np].void_ratio) * d_vol

        return stress_new
