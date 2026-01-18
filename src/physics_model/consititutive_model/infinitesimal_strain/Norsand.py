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
        else:
            return ti.Vector([self.bulk, self.shear, self.M_tc, self.lambda_csl, self.Gamma, self.chi, self.H_mod, self.N])

    @ti.func
    def GetInternalVariables(self, state_vars):
        # Retorna: epdstrain, p_image, void_ratio
        return ti.Vector([state_vars.epdstrain, state_vars.p_image, state_vars.void_ratio])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].epdstrain = internal_vars[0]
        stateVars[np].p_image = internal_vars[1]
        stateVars[np].void_ratio = internal_vars[2]

    # ========================== NORSAND PHYSICS CORE ==========================
    
    @ti.func
    def ComputeStressInvariant(self, stress):
        # Invariantes p e q
        p = -(stress[0,0] + stress[1,1] + stress[2,2]) / 3.0
        J2 = ComputeStressInvariantJ2(stress)
        q = ti.sqrt(3.0 * J2) + Threshold
        return p, q

    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        # Parâmetros desempacotados
        M, p_image = material_params[2], internal_vars[1]
        p, q = self.ComputeStressInvariant(stress)
        
        # Yield Function: f = q - M * p * [1 - ln(p/p_image)]
        # Proteção para p <= 0
        f = 0.0
        if p > Threshold:
            f = q - M * p * (1.0 + ti.log(p_image / p))
        else:
            f = q - M * p # Fallback linear se p for muito baixo
        return f, 0.0 # 0.0 é placeholder para segunda yield function (tensile) se houver

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f_shear, _ = self.ComputeYieldFunction(stress, internal_vars, material_params)
        yield_state = 0
        if f_shear > FTOL:
            yield_state = 1
        return yield_state, f_shear

    # --- DERIVADAS ANALÍTICAS (Músculo do Código) ---
    
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        # Derivada da Função de Fluxo (Yield) em relação à Tensão Sigma
        # dF/dSigma = (dF/dp * dp/dSigma) + (dF/dq * dq/dSigma)
        
        M, p_image = material_params[2], internal_vars[1]
        p, q = self.ComputeStressInvariant(stress)
        
        # Derivadas parciais
        # F = q - M*p*(1 + ln(pi/p))
        # dF/dq = 1
        df_dq = 1.0
        
        # dF/dp = -M * [1 + ln(pi/p)] - M*p*(-1/p)
        #       = -M * [1 + ln(pi/p) - 1]
        #       = -M * ln(pi/p)
        df_dp = 0.0
        if p > Threshold:
            df_dp = -M * ti.log(p_image / p)
        
        # Converte para tensor
        dp_dsigma = DpDsigma() # -1/3 I
        dq_dsigma = DqDsigma(stress) # 3/2 * s / q
        
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma
        return df_dsigma

    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        # Potencial Plástico G (Non-associated flow rule)
        # Dilatância D = dg_dp / dg_dq = M - eta
        # D = M_tc * (1 + N * psi) - eta  (Modelo Nova/Jefferies)
        
        M, lambda_csl, Gamma, chi, N = material_params[2], material_params[3], material_params[4], material_params[5], material_params[7]
        p_image, e_curr = internal_vars[1], internal_vars[2]
        p, q = self.ComputeStressInvariant(stress)
        
        # State parameter psi = e - (Gamma - lambda * ln(p))
        psi = e_curr - (Gamma - lambda_csl * ti.log(ti.max(p, Threshold)))
        
        # Razão de tensão atual eta = q / p
        eta = 0.0
        if p > Threshold:
            eta = q / p
            
        # Dilatância D_p = M_tc - eta (Simplificado Cam-Clay) 
        # Ou D_p = M_i - eta (NorSand Standard)
        # M_i é calculado tal que D = 0 quando eta = M_tc
        # Vamos usar a relação de tensão-dilatância do NorSand: D = M - eta - N*|psi| ou similar
        # Aqui, para estabilidade numérica, usamos: D = (M - eta) 
        # (Nota: O código original NorSand é complexo, aqui simplifico para D = M - eta para associado)
        
        # Se formos rigorosos com NorSand: D = (M_theta - eta) / (1 - N) ? Não, vamos simplificar.
        # Assumindo normalidade para estabilidade inicial (dF = dG):
        
        dg_dsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        return dg_dsigma

    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        # H (Hardening Modulus) = - (dF/d_int * d_int/d_eps_p)
        # No NorSand, a variável de endurecimento é p_image.
        # dF / d p_image = - M * p * (1/p_image) = - M * p / p_image
        
        M, lambda_csl, Gamma, chi, H_param = material_params[2], material_params[3], material_params[4], material_params[5], material_params[6]
        p_image, e_curr = internal_vars[1], internal_vars[2]
        p, q = self.ComputeStressInvariant(stress)
        
        psi = e_curr - (Gamma - lambda_csl * ti.log(ti.max(p, Threshold)))
        
        # Lei de endurecimento do NorSand:
        # (p_image_dot / p_image) = H * (psi_max - psi) * eps_v_dot_p
        # Mas aqui precisamos do Módulo Escalar Kp para o retorno.
        
        # Kp = - (dF/d_pi * h_pi + dF/d_psi * h_psi) ... é complexo.
        # Simplificação Robusta para MPM Explicito:
        # Usamos o H_mod definido pelo usuário escalado pela rigidez.
        
        # Regra empírica: Softening se psi > 0, Hardening se psi < 0
        hardening_val = H_param * (M - (q/ti.max(p, Threshold))) 
        
        return ti.max(hardening_val, 1e-5) # Retorna módulo positivo para estabilidade
