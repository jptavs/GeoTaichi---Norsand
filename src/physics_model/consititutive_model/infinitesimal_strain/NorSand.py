import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import PI, FTOL, Threshold
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class NorSandModel(PlasticMaterial):
    """
    NorSand constitutive model for GeoTaichi
    Compatible with ULExplicitEngine + Return Mapping
    """

    def __init__(self, material_type="Solid", configuration="UL",
                 solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)

        # Defaults (overwritten in model_initialize)
        self.lambda_csl = 0.05
        self.Gamma = 1.10
        self.M_tc = 1.25
        self.N = 0.3
        self.chi = 3.5
        self.H_mod = 100.0
        self.e0 = 0.6

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650.0)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.25)

        self.lambda_csl = DictIO.GetAlternative(material, 'LambdaCSL', 0.05)
        self.Gamma = DictIO.GetAlternative(material, 'Gamma', 1.10)
        self.M_tc = DictIO.GetAlternative(material, 'M', 1.25)
        self.N = DictIO.GetAlternative(material, 'N', 0.3)
        self.chi = DictIO.GetAlternative(material, 'Chi', 3.5)
        self.H_mod = DictIO.GetAlternative(material, 'HardeningH', 100.0)
        self.e0 = DictIO.GetAlternative(material, 'InitialVoidRatio', 0.65)

        self.density = density
        self.young = young
        self.poisson = poisson

        self.shear, self.bulk = self.calculate_lame_parameter(young, poisson)
        self.max_sound_speed = self.get_sound_speed(density, young, poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print("Constitutive model: NorSand (Plastic, Return Mapping)")
        print("Model ID:", materialID)
        if not GlobalVariable.RANDOMFIELD:
            print(f"M      = {self.M_tc}")
            print(f"Gamma  = {self.Gamma}")
            print(f"Lambda = {self.lambda_csl}")
            print(f"Chi    = {self.chi}")
            print(f"H      = {self.H_mod}")

    # ------------------------------------------------------------------
    # STATE VARIABLES
    # ------------------------------------------------------------------

    def define_state_vars(self):
        state_vars = {
            "epdstrain": float,     # accumulated plastic deviatoric strain
            "p_image": float,       # image pressure (hardening variable)
            "void_ratio": float
        }
        return state_vars

    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        return ti.Vector([
            self.bulk,
            self.shear,
            self.M_tc,
            self.lambda_csl,
            self.Gamma,
            self.chi,
            self.H_mod,
            self.N
        ])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([
            state_vars.epdstrain,
            state_vars.p_image,
            state_vars.void_ratio
        ])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].epdstrain = internal_vars[0]
        stateVars[np].p_image = internal_vars[1]
        stateVars[np].void_ratio = internal_vars[2]

    # ------------------------------------------------------------------
    # INVARIANTS
    # ------------------------------------------------------------------

    @ti.func
    def ComputeStressInvariant(self, stress):
        p = -(stress[0, 0] + stress[1, 1] + stress[2, 2]) / 3.0
        J2 = ComputeStressInvariantJ2(stress)
        q = ti.sqrt(3.0 * J2) + Threshold
        return p, q

    # ------------------------------------------------------------------
    # YIELD FUNCTION
    # ------------------------------------------------------------------

    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        M = material_params[2]
        p_image = internal_vars[1]

        p, q = self.ComputeStressInvariant(stress)

        f = 0.0
        if p > Threshold:
            f = q - M * p * (1.0 + ti.log(p_image / p))
        else:
            f = q - M * p

        return f, 0.0

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f, _ = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return ti.cast(f > FTOL, ti.i32), f

    # ------------------------------------------------------------------
    # DERIVATIVES
    # ------------------------------------------------------------------

    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        M = material_params[2]
        p_image = internal_vars[1]

        p, q = self.ComputeStressInvariant(stress)

        df_dq = 1.0
        df_dp = 0.0
        if p > Threshold:
            df_dp = -M * ti.log(p_image / p)

        dp_dsigma = DpDsigma()
        dq_dsigma = DqDsigma(stress)

        return df_dp * dp_dsigma + df_dq * dq_dsigma

    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        # Associado para robustez numérica (MPM explícito)
        return self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)

    # ------------------------------------------------------------------
    # HARDENING MODULUS
    # ------------------------------------------------------------------

    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma,
                              stress, internal_vars, state_vars, material_params):

        M = material_params[2]
        lambda_csl = material_params[3]
        Gamma = material_params[4]
        H = material_params[6]

        p_image = internal_vars[1]
        e = internal_vars[2]

        p, q = self.ComputeStressInvariant(stress)

        psi = e - (Gamma - lambda_csl * ti.log(ti.max(p, Threshold)))

        # NorSand-inspired hardening/softening
        Kp = H * (M - q / ti.max(p, Threshold)) * (1.0 - psi)

        return ti.max(Kp, 1e-6)
