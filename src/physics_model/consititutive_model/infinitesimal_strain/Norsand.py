import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import PI, FTOL, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable

@ti.data_oriented
class NorSandModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        # Parâmetros padrão (serão sobrescritos pelo model_initialize ou random field)
        self.lambda_csl = 0.05
        self.Gamma = 1.10
        self.M_tc = 1.25 # Critical State Ratio (compression)
        self.N = 0.3     # Volumetric coupling
        self.chi = 3.5   # State dilatancy parameter
        self.H_mod = 100.0 # Hardening modulus parameter
        self.e0 = 0.6    # Initial void ratio placeholder

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.2)
        
        # NorSand Parameters
        lambda_csl = DictIO.GetAlternative(material, 'LambdaCSL', 0.05)
        Gamma = DictIO.GetAlternative(material, 'Gamma', 1.10)
        M = DictIO.GetAlternative(material, 'M', 1.25)
        N = DictIO.GetAlternative(material, 'N', 0.3)
        chi = DictIO.GetAlternative(material, 'Chi', 3.5)
        H_mod = DictIO.GetAlternative(material, 'HardeningH', 100.0)
        e0 = DictIO.GetAlternative(material, 'InitialVoidRatio', 0.6)

        self.add_material(density, young, poisson, lambda_csl, Gamma, M, N, chi, H_mod, e0)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, lambda_csl, Gamma, M, N, chi, H_mod, e0):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.lambda_csl = lambda_csl
        self.Gamma = Gamma
        self.M_tc = M
        self.N = N
        self.chi = chi
        self.H_mod = H_mod
        self.e0 = e0
        
        self.shear, self.bulk = self.calculate_lame_parameter(self.young, self.poisson)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: NorSand Model (Fully Coupled)')
        print("Model ID: ", materialID)
        if GlobalVariable.RANDOMFIELD is False:
            print(f'M (Critical State): {self.M_tc}')
            print(f'Gamma (CSL intercept): {self.Gamma}')
            print(f'Lambda (CSL slope): {self.lambda_csl}')
            print(f'Chi (Dilatancy): {self.chi}')
            print(f'H (Hardening): {self.H_mod}')

    def define_state_vars(self):
        state_vars = {}
        # NorSand precisa de p_image (tamanho da superfície) e void_ratio
        state_vars.update({'epdstrain': float, 'p_image': float, 'void_ratio': float})
        
        if GlobalVariable.RANDOMFIELD:
            # Mapeamento para campos aleatórios
            state_vars.update({'density': float, 'shear': float, 'bulk': float, 
                             'Gamma': float, 'lambda_csl': float, 'M_tc': float, 
                             'chi': float, 'H_mod': float})
        return state_vars

    # ========================== RANDOM FIELD INFRASTRUCTURE ==========================
    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        # Adaptado do DP: Defaults caso o arquivo random field não cubra tudo
        self.M_tc = DictIO.GetAlternative(parameter, 'M', 1.25)
        self.chi = DictIO.GetAlternative(parameter, 'Chi', 3.5)
        # ... outros defaults se necessário

    def read_random_field(self, start_particle, end_particle, stateVars):
        # AVISO: O arquivo .txt deve seguir a ordem:
        # 0:Density, 1:Young, 2:Poisson, 3:Gamma, 4:Lambda, 5:M, 6:Chi, 7:H
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments='#').transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise RuntimeError("Shape error for the random field file")
            
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        
        # Mapeamento das colunas do arquivo para NorSand
        gamma_arr = np.ascontiguousarray(random_field[0:, 3])
        lambda_arr = np.ascontiguousarray(random_field[0:, 4])
        M_arr = np.ascontiguousarray(random_field[0:, 5])
        chi_arr = np.ascontiguousarray(random_field[0:, 6])
        H_arr = np.ascontiguousarray(random_field[0:, 7]) # Opcional, ou fixo

        shear, bulk = self.calculate_lame_parameter(young, poisson)
        
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, 
                                      gamma_arr, lambda_arr, M_arr, chi_arr, H_arr, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    @ti.kernel
    def kernel_add_random_material(self, start_particle: int, end_particle: int, 
                                 density: ti.types.ndarray(), shear: ti.types.ndarray(), bulk: ti.types.ndarray(),
                                 gamma_arr: ti.types.ndarray(), lambda_arr: ti.types.ndarray(), 
                                 M_arr: ti.types.ndarray(), chi_arr: ti.types.ndarray(), H_arr: ti.types.ndarray(),
                                 stateVars: ti.template()):
        for np_idx in range(start_particle, end_particle):
            idx = np_idx - start_particle
            stateVars[np_idx].density = density[idx]
            stateVars[np_idx].shear = shear[idx]
            stateVars[np_idx].bulk = bulk[idx]
            # NorSand Vars
            stateVars[np_idx].Gamma = gamma_arr[idx]
            stateVars[np_idx].lambda_csl = lambda_arr[idx]
            stateVars[np_idx].M_tc = M_arr[idx]
            stateVars[np_idx].chi = chi_arr[idx]
            stateVars[np_idx].H_mod = H_arr[idx]
            
            # Inicialização segura do p_image para evitar log(0) no início
            stateVars[np_idx].p_image = 1000.0 
            stateVars[np_idx].void_ratio = self.e0 # Idealmente, ler e0 do campo também

    # ========================== INTERNAL VARS UTILS ==========================
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return ti.Vector([state_vars.bulk, state_vars.shear, state_vars.M_tc, state_vars.lambda_csl, 
                            state_vars.Gamma, state_vars.chi, state_vars.H_mod, self.N])
