import taichi as ti
import numpy as np
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import PI, Threshold
from src.utils.ObjectIO import DictIO

@ti.data_oriented
class NorSandModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        # Parâmetros de literatura (Defaults para Rejeito)
        self.lambda_csl = 0.05
        self.Gamma = 1.10
        self.M = 1.25
        self.N = 0.3
        self.chi = 3.5
        self.H = 100.0
        self.e0 = 0.7

    def model_initialize(self, material):
        # Permite alteração no Colab via dicionário 'material'
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.2)
        
        # NorSand Specific - Carrega do input ou mantém default de literatura
        lambda_csl = DictIO.GetAlternative(material, 'LambdaCSL', 0.05)
        Gamma = DictIO.GetAlternative(material, 'Gamma', 1.10)
        M = DictIO.GetAlternative(material, 'M', 1.25)
        N = DictIO.GetAlternative(material, 'N', 0.3)
        chi = DictIO.GetAlternative(material, 'Chi', 3.5)
        H = DictIO.GetAlternative(material, 'HardeningH', 100.0)
        e0 = DictIO.GetAlternative(material, 'InitialVoidRatio', 0.7)

        self.add_material(density, young, poisson, lambda_csl, Gamma, M, N, chi, H, e0)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, lambda_csl, Gamma, M, N, chi, H, e0):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.lambda_csl = lambda_csl
        self.Gamma = Gamma
        self.M = M
        self.N = N
        self.chi = chi
        self.H = H
        self.e0 = e0
        self.shear, self.bulk = self.calculate_lame_parameter(self.young, self.poisson)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def define_state_vars(self):
        # Variáveis de estado críticas para a evolução do modelo
        return {
            'epdstrain': float,
            'p_image': float,      # Pressão imagem (tamanho da superfície)
            'void_ratio': float    # Índice de vazios atual
        }

    @ti.func
    def ComputeStressInvariant(self, stress):
        # p' e q (von Mises 3D)
        p = -(stress[0,0] + stress[1,1] + stress[2,2]) / 3.0
        q = ti.sqrt(ComputeStressInvariantJ2(stress) * 3.0) + Threshold
        return p, q

    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        p, q = self.ComputeStressInvariant(stress)
        p_image = internal_vars[1]
        M = material_params[2]
        # f = q - M*p*(1 - ln(p/p_image))
        return q - M * p * (1.0 - ti.log(p / p_image))

    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        # Vetor para uso nos kernels internos
        return ti.Vector([self.bulk, self.shear, self.M, self.lambda_csl, self.Gamma, self.chi, self.H, self.N])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.epdstrain, state_vars.p_image, state_vars.void_ratio])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].epdstrain = internal_vars[0]
        stateVars[np].p_image = internal_vars[1]
        stateVars[np].void_ratio = internal_vars[2]
