from pyoptsparse import OPT, Optimization, History
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, ClassVar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

@dataclass(frozen=True)
class AircraftConfig:
    
    # Wing 
    MAC: float = 0.6276          # mac (m)
    S: float = 3.151             # planform area (m^2)
    b: float = 5.021             # span (m)
    AR: float = 8.000            # aspect ratio
    e_wing: float = 0.8904       # efficiency

    # Flight
    TOGW: float = 27.221            # takeoff gross weight (kg)
    m_fixed: float = 25.868         # fixed aircraft weight (kg)
    V_cruise: float = 11.22         # Cruise velocity (m/s)
    V_stall: float = 8.84           # Stall velocity (m/s)
    rho: float = 1.211              # Air density (kg/m^3)

    # Stability 
    SM_target: float = 0.15      # Target static margin (fraction of MAC)
    h_cg: float = 0.40           # CG location (fraction of MAC)
    h_np: float = 0.25           # Neutral Point Location
    C_maf: float = -0.27         # airfoil pitching moment coefficient
    C_law: float = 5.0649        # Wing lift curve slope
    
    @property
    def h_ac_wing(self):  # Center of Pressure (fraction of MAC)
        return self.h_cg - self.C_mowf / self.C_law;

    @property
    def C_mowf(self) -> float:
        return self.C_maf * self.AR / (self.AR + 2)

    @property
    def C_L_cruise(self) -> float:
        return 2 * self.TOGW * 9.81 / (self.rho * self.V_cruise**2 * self.S)

@dataclass(frozen=True)
class TailConfig:
    
    # Tail
    AR_h: float = 8 * (2/3)      # Horizontal tail aspect ratio
    AR_v: float = 2.0            # Vertical tail aspect ratio
    swp_v: float = 25.0          # Vertical tail sweep angle (deg)

    alpha_stall: float = np.deg2rad(13)  # Tail stall angle (rad)

    C_l_alpha: float = 0.106 * 180 / np.pi  # 2D lift curve slope (rad)

    # Elevator
    elevator_ratio: float = 0.40          # chord ratio
    elevator_eff: float = 0.50            # effectiveness
    delta_e_max: float = 25               # Max deflection (deg)

    # Rudder
    rudder_ratio: float = 0.40            # chord ratio
    rudder_eff: float = 0.50              # effectiveness
    delta_r_max: float = 25               # Max deflection (deg)

    eta_h: float = 0.95          # Horizontal tail efficiency
    eta_v: float = 0.90          # Vertical tail efficiency

    @property
    def C_Lah(self) -> float:
        return self.C_l_alpha / (1 + self.C_l_alpha / (np.pi * self.AR_h))

    @property
    def C_Lav(self) -> float:
        return self.C_l_alpha / (1 + self.C_l_alpha / (np.pi * self.AR_v))

@dataclass(frozen=True)
class StructuralConfig:
    
    # Material
    #E: float = 135e9             # Young's modulus
    E: float = 216e9             # Ultra High modulus
    G: float = 8.96e9            # Shear modulus
    rho: float = 1660            # Material density
    sigma_ult: float = 1250e6    # Ultimate tensile strength
    tau_ult: float = 80e6
    FOS: float = 1.5          

    # Structural Constaints
    n_max = 2.0                  # Max load factor (g)

    # Spar
    h_spar: float = 10e-3        # height (m)
    d_spar: float = 8e-3         # diameter/hole (m)

    @property
    def sigma_allow(self) -> float:
        return self.sigma_ult / self.FOS
    
    @property
    def tau_allow(self) -> float:
        return self.tau_ult / self.FOS

    @property
    def I_spar(self) -> float:
        return self.h_spar**4 / 12 - np.pi * self.d_spar**4 / 64

    # carbon fiber tubes
    TUBE_DATABASE: ClassVar[List[Tuple[int, int, float, str]]] = [
        #https://www.rockwestcomposites.com/45009-uhm.html 
        #final selected tail boom
        (51.181, 47.625, 1.778, "51x47.6x1.8mm"),
    ]
   
    """
        # Round - Ultra High Modulus (UHM)
        (34.98, 30.00, 2.49, "35.0mm Round (HM Spread Tow)"),         # P/N 46657
        (39.90, 34.93, 2.49, "39.9mm Round (HM Spread Tow)"),         # P/N 46658
        # Circular
        (31.50, 28.58, 1.47, "31.5x28.6x1.5mm"),# 45562 (Ferrule)
        (34.98, 30.00, 2.49, "35x30x2.5mm"),    # 46657 (Spread Tow)
        (39.90, 34.93, 2.49, "39.9x34.9x2.5mm"),# 46658 (Spread Tow)
        (45.0, 41.6, 1.7, "45x41.6x1.7mm"),     # PBRT-41645 (Pullbraided)
        (47.75, 44.45, 1.65, "47.8x44.5x1.7mm"),# 45123 (1-7/8" OD)
        (50.0, 46.0, 2.0, "50x46x2mm"),         # PBRT-4650 (Pullbraided)
        (50.80, 44.45, 3.18, "50.8x44.5x3.2mm"),# 45111 (Fabric)
        # Square
        (34.29, 31.75, 1.27, "34.3x31.8x1.3mm"),  # 25540 (Fabric)
        (41.15, 38.10, 1.52, "41.2x38.1x1.5mm"),  # 25492 (Size 1.50" ID)
        (50.80, 44.45, 3.18, "50.8x44.5x3.2mm"),  # 25518 (Thick Wall)
        (54.10, 50.80, 1.65, "54.1x50.8x1.7mm"),  # 25504-IM (Intermediate Modulus)
        (54.61, 50.80, 1.91, "54.6x50.8x1.9mm"),  # 25513 (Size 2.00" ID)
        (57.91, 50.80, 3.56, "57.9x50.8x3.6mm"),  # 25523 (Fabric)
    """
    

AC = AircraftConfig()
TAIL = TailConfig()
STRUCT = StructuralConfig()

@dataclass
class PhysicsEquations:
    def tail_geometry(self, S: float, AR: float, tpr: float) -> Dict[str, float]:
    
        b = np.sqrt(S * AR)
        c_root = 2 * S / (b * (1 + tpr))
        c_tip = tpr * c_root
        MAC = (2 / 3) * c_root * (1 + tpr + tpr**2) / (1 + tpr)
        return {'b': b, 'MAC': MAC, 'c_root': c_root, 'c_tip': c_tip}

    def tail_weight(self, S: float, AR: float, tpr: float) -> float:
        geom = self.tail_geometry(S, AR, tpr)
    
        rib_spacing = 0.0635 
        avg_chord = (geom['c_root'] + geom['c_tip']) / 2.0
            
        refLength = 0.081 
        refVol = 1036.73963495e-9 
        density = 160 
            
        avg_rib_vol = ((avg_chord / refLength) ** 3) * refVol
        num_ribs_continuous = geom['b'] / rib_spacing
        TotalRibWeight = num_ribs_continuous * avg_rib_vol * density

        SparWeight = (geom['b'] / 1.22) * 0.07438 
        SkinWeight = (S * 2) * 0.09  
            
        WingWeight = TotalRibWeight + SparWeight + SkinWeight

        return WingWeight * 9.81

    def get_moment_arms(self, l_boom: float,S_h: float, S_v: float, tpr_h: float, tpr_v: float, tail: TailConfig) -> Tuple[float, float, float]:

        v = PHYS.tail_geometry(S_v, tail.AR_v, tpr_v)
        h = PHYS.tail_geometry(S_h, tail.AR_h, tpr_h)

        tan_le_v = np.tan(np.deg2rad(tail.swp_v)) + (1 - tpr_v) / (tail.AR_v * (1 + tpr_v))

        y_mac_v = (v["b"] / 3) * (1 + 2 * tpr_v) / (1 + tpr_v)
        le_mac_v = l_boom + y_mac_v * tan_le_v
        l_v = le_mac_v + 0.25 * v["MAC"]

        le_vt_tip = l_boom + v["b"] * tan_le_v

        l_h = le_vt_tip + 0.25 * h["MAC"]

        h_vt = v["b"]

        return l_h, l_v, h_vt

    def get_boom_properties(self, GJ_req: float, struct: StructuralConfig,
                            l_boom: float = None, S_h: float = None) -> Dict:

        EI_req = 0.0
        if l_boom is not None and S_h is not None:
            q = 0.5 * 1.225 * 11.22**2 * struct.n_max  # Max Load Factor
            L_h_max = q * S_h * 1.1 # Cl_max
            delta_allow = l_boom / 100
            EI_req = (L_h_max * l_boom**3) / (3 * delta_allow)

        for OD_mm, ID_mm, width, name in STRUCT.TUBE_DATABASE:
            h = OD_mm * 1e-3
            d = ID_mm * 1e-3

            I = np.pi/64 * (h**4 - d**4) # Circular Moment of Inertia
            J = np.pi/32 * (h**4 - d**4) # Circular Polar Moment of Inertia
            A = np.pi/4 * (h**2 - d**2) # Circular Area
            #I = (h**4 - d**4) / 12 # Square Moment of Inertia
            #J = ((h + d)**3 * (h - d)) / 8 # Square Polar Moment of Inertia
            #A = (h**2 - d**2) # Square Area
            GJ = struct.G * J
            EI = struct.E * I
            #weight_per_m = A * struct.rho * 9.81
            weight_per_m = 0.458 * 9.81

            if GJ >= GJ_req and EI >= EI_req:
                return {
                    'h': h, 'd': d, 'I': I, 'J': J,
                    'GJ': GJ, 'weight_per_m': weight_per_m,
                    'name': name, 'is_dual': False
                }
            
            # dual-boom
            I_dual = 2 * I + 2 * A * (h/2)**2
            J_dual = 2 * J
            GJ_dual = struct.G * J_dual
            EI_dual = struct.E * I_dual

            if GJ_dual >= GJ_req and EI_dual >= EI_req:
                return {
                    'h': h, 'd': d, 'I': I_dual, 'J': J_dual,
                    'GJ': GJ_dual, 'weight_per_m': 2 * weight_per_m,
                    'name': f"DUAL {name}", 'is_dual': True
                }

        h, d = 50e-3, 48e-3
        I = np.pi/64 * (h**4 - d**4)
        J = np.pi/32 * (h**4 - d**4)
        A = np.pi/4 * (h**2 - d**2)
        return {
            'h': h, 'd': d,
            'I': 2*I + 2*A*(h/2)**2,
            'J': 2*J,
            'GJ': struct.G * 2*J,
            'weight_per_m': 2 * A * struct.rho * 9.81,
            'name': "DUAL 50x48x1mm (Max)",
            'is_dual': True
        }

PHYS = PhysicsEquations()

@dataclass
class Constraints:
    def evaluate_all_constraints(x: np.ndarray, AC: AircraftConfig,
                                TAIL: TailConfig, STRUCT: StructuralConfig) -> Dict[str, float]:
    
        l_boom, S_h, S_v, tpr_h, tpr_v = x

        funcs = {}
        q = 0.5 * AC.rho * AC.V_cruise**2

        h_geom = PHYS.tail_geometry(S_h, TAIL.AR_h, tpr_h)
        v_geom = PHYS.tail_geometry(S_v, TAIL.AR_v, tpr_v)
        l_h, l_v, h_vt = PHYS.get_moment_arms(l_boom, S_h, S_v, tpr_h, tpr_v, TAIL)

        V_H = l_h * S_h / (AC.MAC * AC.S)
        V_V = l_v * S_v / (AC.S * AC.b)

        W_h = PHYS.tail_weight(S_h, TAIL.AR_h, tpr_h)
        W_v = PHYS.tail_weight(S_v, TAIL.AR_v, tpr_v)
        m_tail = (W_h + W_v) / 9.81

        # Boom 
        omega = 2 * np.pi * 7.0
        I_theta = m_tail * (h_vt / 2 * np.sin(np.deg2rad(TAIL.swp_v))) ** 2
        GJ_freq = I_theta * l_boom * omega ** 2
        Y_v = q * STRUCT.n_max * S_v * 0.7
        T_boom = h_vt / 2 * Y_v
        GJ_twist = T_boom * l_boom / np.deg2rad(1)
        GJ_req = max(GJ_freq, GJ_twist)
        boom = PHYS.get_boom_properties(GJ_req, STRUCT, l_boom, S_h)
        W_boom = boom['weight_per_m'] * l_boom

        W_tail = W_h + W_v + W_boom
        m_tail = W_tail / 9.81  
        TOGW = AC.m_fixed + m_tail  

        C_L = (TOGW * 9.81) / (q * AC.S)

        # trim
        C_m_wing_cg = AC.C_mowf + C_L * (AC.h_cg - AC.h_ac_wing)

        # Deep stall
        AoA_ds = np.deg2rad(10)
        h_vt_min = (AC.MAC * np.sin(AoA_ds) + 0.1 * l_h)

        # tail vol ref
        V_H_ref = 0.40
        V_H_max = 0.60
        V_V_ref = 0.02
        V_V_max = 0.05

        # Boom deflection
        delta_weight = ((W_boom * l_boom**3) / (3 * STRUCT.E * boom['I']) 
                        + ((W_h + W_v) * l_boom**3) / (3 * STRUCT.E * boom['I']))

        C_Lh_max = 1.1
        L_h_max = q * STRUCT.n_max * S_h * C_Lh_max
        delta_aero = (L_h_max * l_boom**3) / (3 * STRUCT.E * boom['I'])

        delta_total = delta_weight + delta_aero
        delta_allow = l_boom / 100.0

        funcs['boom_deflection'] = delta_allow - delta_total

        # Boom stress
        M_aero = L_h_max * l_boom
        M_total = m_tail * 9.81 + M_aero
        sigma_boom = M_total * (boom['h'] / 2) / boom['I']

        funcs['boom_stress'] = STRUCT.sigma_allow - sigma_boom

        # V-tail stress
        M_bend_h = L_h_max * (v_geom['b'] / 2) + W_h * (v_geom['b'] / 4)
        sigma_bend = M_bend_h * (STRUCT.h_spar / 2) / STRUCT.I_spar

        funcs['vtail_bending'] = STRUCT.sigma_allow - sigma_bend 

        T_root = L_h_max / 2 * h_vt / 2
        tau_torsion = T_root / (2 * np.pi * STRUCT.h_spar * STRUCT.d_spar) 

        V_root = L_h_max / 2
        tau_shear = V_root / (STRUCT.h_spar * STRUCT.d_spar)

        v_tau = (tau_torsion**2 + tau_shear**2)**0.5

        funcs['vtail_torsion'] = STRUCT.tau_allow - v_tau

        sigma_vm = (sigma_bend**2 + 3 * v_tau ** 2) ** 0.5

        funcs['vtail_vm'] = STRUCT.sigma_allow - sigma_vm

        # Torsional
        funcs['torsion'] = boom['GJ'] - GJ_req
        
        # trim pitch capability
        C_L_tail_required = (C_m_wing_cg / (V_H * TAIL.eta_h)) if V_H > 0.01 else 999
        alpha_usable = TAIL.alpha_stall * 0.9
        C_L_tail_max = TAIL.C_Lah * alpha_usable

        funcs['pitch_trim'] = C_L_tail_max - C_L_tail_required

        # Tail no stall 
        C_L_tail_trim = (C_m_wing_cg / (V_H * TAIL.eta_h)) if V_H > 0.01 else 0
        alpha_trim = C_L_tail_trim / TAIL.C_Lah
        delta_e_rad = np.deg2rad(TAIL.delta_e_max)
        alpha_eff_max = alpha_trim - TAIL.elevator_eff * delta_e_rad

        funcs['tail_no_stall'] = abs(alpha_eff_max) - TAIL.alpha_stall

        funcs['incidence_max'] = np.deg2rad(5) + alpha_trim;

        # Deep stall
        funcs['deep_stall'] = h_vt - h_vt_min

        # Static margin
        C_Lalpha_wing = (2 * np.pi * AC.AR) / (2 + np.sqrt(4 + AC.AR**2))
        C_Lalpha_tail = (2 * np.pi * TAIL.AR_h) / (2 + np.sqrt(4 + TAIL.AR_h**2))

        de_da = 2.0 * C_Lalpha_wing / (np.pi * AC.AR)

        x_np_tail = TAIL.eta_h * (S_h / AC.S) * (l_h / AC.MAC) * (C_Lalpha_tail / C_Lalpha_wing) * (1 - de_da)
        h_np = AC.h_np + x_np_tail

        SM_actual = h_np - AC.h_cg

        funcs['static_margin_lower'] = SM_actual - 0.100
        funcs['static_margin_upper'] = 0.150 - SM_actual

        # Pitch control
        C_m_required = 0.20
        C_m_elevator = V_H * TAIL.C_Lah * TAIL.elevator_eff * delta_e_rad

        funcs['pitch_control'] = C_m_elevator - C_m_required

        # Yaw control
        C_n_required = 0.02
        delta_r_rad = np.deg2rad(TAIL.delta_r_max)
        C_n_rudder = V_V * TAIL.C_Lav * TAIL.rudder_eff * delta_r_rad

        funcs['yaw_control'] = (C_n_rudder - C_n_required) * 100

        # V_H lower bound
        funcs['v_h_lower'] = V_H - V_H_ref

        # V_H lower bound
        funcs['v_h_upper'] = V_H_max - V_H

        # V_V lower bound
        funcs['v_v_lower'] = V_V - V_V_ref

        # V_V upper bound 
        funcs['v_v_upper'] = V_V_max - V_V

        return funcs

    def constraint_function(xdict: dict) -> Tuple[Dict[str, float], bool]:

        x = xdict['xvars']
        funcs = Constraints.evaluate_all_constraints(np.array(x), AC, TAIL, STRUCT)
        return funcs, False

    def compute_derived_quantities(x: np.ndarray) -> Dict:
        
        l_boom, S_h, S_v, tpr_h, tpr_v = x

        q = 0.5 * AC.rho * AC.V_cruise**2

        h_geom = PHYS.tail_geometry(S_h, TAIL.AR_h, tpr_h)
        v_geom = PHYS.tail_geometry(S_v, TAIL.AR_v, tpr_v)
        l_h, l_v, h_vt = PHYS.get_moment_arms(l_boom, S_h, S_v, tpr_h, tpr_v, TAIL)

        V_H = l_h * S_h / (AC.MAC * AC.S)
        V_V = l_v * S_v / (AC.S * AC.b)

        W_h = PHYS.tail_weight(S_h, TAIL.AR_h, tpr_h)
        W_v = PHYS.tail_weight(S_v, TAIL.AR_v, tpr_v)
        m_tail = (W_h + W_v) / 9.81
        
        # Boom 
        omega = 2 * np.pi * 7.0
        I_theta = m_tail * (h_vt / 2 * np.sin(np.deg2rad(TAIL.swp_v))) ** 2
        GJ_freq = I_theta * l_boom * omega ** 2
        Y_v = q * STRUCT.n_max * S_v * 0.7
        T_boom = h_vt / 2 * Y_v
        GJ_twist = T_boom * l_boom / np.deg2rad(1)
        GJ_req = max(GJ_freq, GJ_twist)
        boom = PHYS.get_boom_properties(GJ_req, STRUCT, l_boom, S_h)
        W_boom = boom['weight_per_m'] * l_boom
        
        W_tail = W_h + W_v + W_boom
        m_tail = W_tail / 9.81  
        TOGW = AC.m_fixed + m_tail  

        q = 0.5 * AC.rho * AC.V_cruise**2

        if boom['is_dual']:
            S_wet_boom = 2 * np.pi * boom['h'] * l_boom
        else:
            S_wet_boom = np.pi * boom['h'] * l_boom
        S_wet_h = 2.1 * S_h
        S_wet_v = 2.1 * S_v
        S_wet_total = S_wet_h + S_wet_v + S_wet_boom

        C_D0 = 0.006
        D_parasite = C_D0 * q * S_wet_total

        C_L = (TOGW * 9.81) / (q * AC.S)
        C_m_wing = AC.C_mowf + C_L * (AC.h_cg - AC.h_ac_wing)
        C_L_tail = (C_m_wing / (V_H * TAIL.eta_h)) if V_H > 0.01 else 0
        alpha_trim = np.rad2deg(C_L_tail / TAIL.C_Lah)

        e_h = 0.90
        D_induced = (C_L_tail**2 / (np.pi * e_h * TAIL.AR_h)) * q * S_h
        D_total = D_parasite + D_induced

        return {
            'l_boom': l_boom, 'S_h': S_h, 'S_v': S_v, 'tpr_h': tpr_h, 'tpr_v': tpr_v, 'swp_v': 25,
            'l_h': l_h, 'l_v': l_v, 'h_vt': h_vt,
            'V_H': V_H, 'V_V': V_V,
            'AoA': alpha_trim,
            'b_h': h_geom['b'], 'c_root_h': h_geom['c_root'], 'c_tip_h': h_geom['c_tip'], 'MAC_h': h_geom['MAC'],
            'b_v': v_geom['b'], 'c_root_v': v_geom['c_root'], 'c_tip_v': v_geom['c_tip'], 'MAC_v': v_geom['MAC'],
            'W_h': W_h, 'W_v': W_v, 'W_boom': W_boom, 'TOGW': TOGW,
            'boom_name': boom['name'], 'boom_is_dual': boom['is_dual'],
            'D_parasite': D_parasite, 'D_induced': D_induced, 'D_total': D_total,
            'S_wet_h': S_wet_h, 'S_wet_v': S_wet_v, 'S_wet_boom': S_wet_boom, 'S_wet_total': S_wet_total,
            'C_L_tail': C_L_tail
        }

@dataclass
class Optimizer:
    def objective_function(xdict):

        x = xdict['xvars']
        l_boom, S_h, S_v, tpr_h, tpr_v = x

        q = 0.5 * AC.rho * AC.V_cruise**2

        h_geom = PHYS.tail_geometry(S_h, TAIL.AR_h, tpr_h)
        v_geom = PHYS.tail_geometry(S_v, TAIL.AR_v, tpr_v)
        l_h, l_v, h_vt = PHYS.get_moment_arms(l_boom, S_h, S_v, tpr_h, tpr_v, TAIL)

        V_H = l_h * S_h / (AC.MAC * AC.S)
        V_V = l_v * S_v / (AC.S * AC.b)

        W_h = PHYS.tail_weight(S_h, TAIL.AR_h, tpr_h)
        W_v = PHYS.tail_weight(S_v, TAIL.AR_v, tpr_v)
        m_tail = (W_h + W_v) / 9.81

        # Boom 
        omega = 2 * np.pi * 7.0
        I_theta = m_tail * (h_vt / 2 * np.sin(np.deg2rad(TAIL.swp_v))) ** 2
        GJ_freq = I_theta * l_boom * omega ** 2
        Y_v = q * STRUCT.n_max * S_v * 0.7
        T_boom = h_vt / 2 * Y_v
        GJ_twist = T_boom * l_boom / np.deg2rad(1)
        GJ_req = max(GJ_freq, GJ_twist)
        boom = PHYS.get_boom_properties(GJ_req, STRUCT, l_boom, S_h)
        W_boom = boom['weight_per_m'] * l_boom

        W_tail = W_h + W_v + W_boom
        m_tail = W_tail / 9.81 
        TOGW = AC.m_fixed + m_tail 

        C_L = (TOGW * 9.81) / (q * AC.S)

        # Wetted areas
        if boom['is_dual']:
            S_wet_boom = 2 * np.pi * boom['h'] * l_boom
        else:
            S_wet_boom = np.pi * boom['h'] * l_boom
        S_wet_h = 2.1 * S_h
        S_wet_v = 2.1 * S_v
        S_wet_total = S_wet_h + S_wet_v + S_wet_boom

        # Parasitic drag
        C_D0 = 0.006
        D_parasite = C_D0 * q * S_wet_total

        # Induced drag
        l_h, _, _ = PHYS.get_moment_arms(l_boom, S_h, S_v, tpr_h, tpr_v, TAIL)
        V_H = l_h * S_h / (AC.MAC * AC.S)

        C_m_wing = AC.C_mowf + C_L * (AC.h_cg - AC.h_ac_wing)
        C_L_tail = (C_m_wing / (V_H * TAIL.eta_h)) if V_H > 0.01 else 0

        e_h = 0.90
        D_induced = (C_L_tail**2 / (np.pi * e_h * TAIL.AR_h)) * q * S_h

        D_total = D_parasite + D_induced
        
        # References
        W_ref = 30.0
        D_ref = 2.0
        l_ref = 1.60

        # Weights
        w_drag   = 0.46
        w_weight = 0.46
        w_boom   = 0.08

        J = (w_drag * (D_total / D_ref) +
            w_weight * (W_tail / W_ref) +
            w_boom * (l_boom / l_ref) ** 2)
        
        funcs = {'obj': J}
        fail = False
        return funcs, fail

    def user_function(xdict):
        
        # objective
        obj_funcs, fail1 = Optimizer.objective_function(xdict)
 
        # constraints
        con_funcs, fail2 = Constraints.constraint_function(xdict)

        # Combine
        funcs = {**obj_funcs, **con_funcs}
        fail = fail1 or fail2
        return funcs, fail
    

    def run_optimization():

        # variables
        x0 = np.array([1.0, 0.30, 0.20, 0.70, 0.70])

        lower = np.array([1.0, 0.30, 0.18, 0.60, 0.60])
        upper = np.array([2.0, 1.20, 0.60, 0.70, 0.70])

        var_names = ['l_boom', 'S_h', 'S_v', 'tpr_h', 'tpr_v']

        # Create optimization problem
        optProb = Optimization('T-Tail Optimization', Optimizer.user_function)

        # Add design variables
        optProb.addVarGroup('xvars', 5, 'c', value=x0, lower=lower, upper=upper)

        # Add objective
        optProb.addObj('obj')

        # Add constraints
        constraint_names = [
            'boom_deflection', 'boom_stress', 'vtail_bending', 'vtail_torsion', 
            'vtail_vm', 'torsion', 'pitch_trim', 'tail_no_stall', 'deep_stall', 
            'static_margin_lower', 'static_margin_upper', 
            'pitch_control', 'yaw_control',
            #'incidence_max',
            'v_h_lower', 'v_v_lower', 'v_v_upper', 'v_h_upper'
        ]

        for name in constraint_names:
            optProb.addCon(name, lower=0.0)

        # Setup
        history_file = os.path.join(OUTPUT_DIR, 'optimization_history.hst')

        opt = OPT('slsqp', options={'ACC': 1e-5, 'MAXIT': 200})

        # Solve
        sol = opt(optProb, sens="CS", storeHistory=history_file)
        print(sol)
        
        return sol, optProb, history_file

    def post_process(sol, optProb, history_file: str):

        try:
            x_opt = sol.xStar['xvars']
        except (AttributeError, KeyError):
            print("Warning: Could not extract optimal solution")
            exit

        xdict = {'xvars': x_opt}
        funcs, _ = Optimizer.user_function(xdict)
        obj_val = funcs.pop('obj')

        print("\nExporting Summary Report\n")

        Output.export_summary_report(x_opt, obj_val, funcs)

        print("\nGenerating plots...\n")

        Output.plot_design(x_opt)
        
        return x_opt

    def main():
        sol, optProb, history_file = Optimizer.run_optimization()

        x_opt = Optimizer.post_process(sol, optProb, history_file)

        # summary
        derived = Constraints.compute_derived_quantities(x_opt)
        mass_total = derived['TOGW'] / 9.81
        print("\n FINAL DESIGN SUMMARY")
        print(f"\nHorizontal Tail Volume Coefficient: V_H = {derived['V_H']:.4f}")
        print(f"Vertical Tail Volume Coefficient:   V_V = {derived['V_V']:.4f}")
        print(f"Horizontal Tail Incidence Angle:   V_V = {derived['AoA']:.4f}")
        print(f"Total Tail Weight: {derived['TOGW']:.2f} N ({mass_total:.3f} kg)")
        print(f"Total Drag: {derived['D_total']:.3f} N")
        print(f"\nBoom Configuration: {derived['boom_name']}")

@dataclass
class Output:
    def export_summary_report(x_opt: np.ndarray, objective: float, constraints: Dict):

        derived = Constraints.compute_derived_quantities(x_opt)

        report_file = os.path.join(OUTPUT_DIR, 'optimization_summary.txt')

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" T-TAIL SIZING OPTIMIZATION - FINAL REPORT\n")
            f.write("="*70 + "\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" AIRCRAFT CONFIGURATION\n")
            f.write("-"*70 + "\n")
            f.write(f"  TOGW:           {AC.TOGW:.1f} kg\n")
            f.write(f"  V_cruise:       {AC.V_cruise:.1f} m/s\n")
            f.write(f"  Wing area:      {AC.S:.3f} m^2\n")
            f.write(f"  Wing span:      {AC.b:.3f} m\n")
            f.write(f"  Wing MAC:       {AC.MAC:.4f} m\n")
            f.write(f"  Static margin:  {AC.SM_target*100:.1f}%\n")
            f.write(f"  Cruise C_L:     {AC.C_L_cruise:.3f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" OPTIMAL DESIGN VARIABLES\n")
            f.write("-"*70 + "\n")
            f.write(f"  Boom length (l_boom):      {x_opt[0]:.4f} m\n")
            f.write(f"  H-tail area (S_h):         {x_opt[1]:.4f} m^2\n")
            f.write(f"  V-tail area (S_v):         {x_opt[2]:.4f} m^2\n")
            f.write(f"  H-tail taper (tpr_h):      {x_opt[3]:.4f}\n")
            f.write(f"  V-tail taper (tpr_v):      {x_opt[4]:.4f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" DERIVED PARAMETERS\n")
            f.write("-"*70 + "\n")
            f.write(f"  boom length (l_boom):      {derived['l_boom']:.4f} m\n")
            f.write(f"  H-tail moment arm (l_h):   {derived['l_h']:.4f} m\n")
            f.write(f"  V-tail moment arm (l_v):   {derived['l_v']:.4f} m\n")
            f.write(f"  T-tail height (h_vt):      {derived['h_vt']:.4f} m\n")
            f.write(f"  H-tail volume (V_H):       {derived['V_H']:.4f}\n")
            f.write(f"  V-tail volume (V_V):       {derived['V_V']:.5f}\n")
            f.write(f"\n  H-tail geometry:\n")
            f.write(f"    Span:        {derived['b_h']:.4f} m\n")
            f.write(f"    Half Span:   {derived['b_h'] / 2:.4f} m\n")
            f.write(f"    Root chord:  {derived['c_root_h']:.4f} m\n")
            f.write(f"    Tip chord:   {derived['c_tip_h']:.4f} m\n")
            f.write(f"    MAC:         {derived['MAC_h']:.4f} m\n")
            f.write(f"    AoA:         {-5.00:.2f}  deg\n")
            f.write(f"    Taper:       {derived['tpr_h']:.2f}\n")
            f.write(f"\n  V-tail geometry:\n")
            f.write(f"    Span:          {derived['b_v']:.4f} m\n")
            f.write(f"    Root chord:    {derived['c_root_v']:.4f} m\n")
            f.write(f"    Tip chord:     {derived['c_tip_v']:.4f} m\n")
            f.write(f"    MAC:           {derived['MAC_v']:.4f} m\n")
            f.write(f"    Sweep:         {derived['swp_v']:.2f}  deg\n")
            f.write(f"    Taper:         {derived['tpr_v']:.2f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" WEIGHT BREAKDOWN\n")
            f.write("-"*70 + "\n")
            f.write(f"  H-tail:         {derived['W_h']/9.81:.3f} kg ({derived['W_h']:.2f} N)\n")
            f.write(f"  V-tail:         {derived['W_v']/9.81:.3f} kg ({derived['W_v']:.2f} N)\n")
            f.write(f"  Boom:           {derived['W_boom']/9.81:.3f} kg ({derived['W_boom']:.2f} N)\n")
            f.write(f"  Boom config:    {derived['boom_name']}\n")
            f.write(f"  Total tail:     {derived['TOGW']/9.81:.3f} kg ({derived['TOGW']:.2f} N)\n")
            f.write(f"  % of TOGW:      {derived['TOGW']/(AC.TOGW*9.81)*100:.2f}%\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" DRAG BREAKDOWN\n")
            f.write("-"*70 + "\n")
            f.write(f"  Parasitic:      {derived['D_parasite']:.3f} N\n")
            f.write(f"  Induced:        {derived['D_induced']:.3f} N\n")
            f.write(f"  Total:          {derived['D_total']:.3f} N\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" CONSTRAINT STATUS\n")
            f.write("-"*70 + "\n")
            violated = {k: v for k, v in constraints.items() if v < 0}
            satisfied = {k: v for k, v in constraints.items() if v >= 0}
            f.write(f"  Total: {len(constraints)}\n")
            f.write(f"  Satisfied: {len(satisfied)}\n")
            f.write(f"  Violated: {len(violated)}\n")

            if violated:
                f.write("\n  VIOLATED:\n")
                for name, val in sorted(violated.items(), key=lambda x: x[1]):
                    f.write(f"    {name:20s}: {val:+.6f}\n")

            f.write("\n  SATISFIED (sorted by margin):\n")
            for name, val in sorted(satisfied.items(), key=lambda x: x[1]):
                f.write(f"    {name:20s}: {val:+.6f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(" OBJECTIVE\n")
            f.write("-"*70 + "\n")
            f.write(f"  Final value: {objective:.6f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write(" END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"  Exported: {report_file}")

    def plot_design(x_opt: np.ndarray):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('T-Tail Design Space Analysis', fontsize=14, fontweight='bold')

        n_grid = 50

        # l_boom vs S_h
        ax = axes[0, 0]
        l_boom_grid = np.linspace(1.0, 2.5, n_grid)
        S_h_grid = np.linspace(0.35, 1.2, n_grid)
        L, SH = np.meshgrid(l_boom_grid, S_h_grid)

        obj_vals = np.zeros_like(L)
        feasible = np.zeros_like(L, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([L[i, j], SH[i, j], x_opt[2], x_opt[3], x_opt[4]])
                xdict = {'xvars': x}
                funcs, _ = Optimizer.user_function(xdict)
                obj_vals[i, j] = funcs['obj']
                cons = {k: v for k, v in funcs.items() if k != 'obj' and k != 'incidence_max' and k != 'yaw_control'}
                feasible[i, j] = all(v >= -1e-6 for v in cons.values())

        contour = ax.contourf(L, SH, obj_vals, levels=60, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='Objective')
        ax.contourf(L, SH, ~feasible, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
        ax.contour(L, SH, feasible.astype(float), levels=[0.5], colors=['red'], linewidths=2)
        ax.plot(x_opt[0], x_opt[1], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
        ax.set_xlabel('Boom Length [m]')
        ax.set_ylabel('H-tail Area [m$^2$]')
        ax.set_title('l_boom vs S_h (S_v, tpr fixed)')
        ax.legend()

        # l_boom vs S_v
        ax = axes[0, 1]
        S_v_grid = np.linspace(0.18, 0.60, n_grid)
        L2, SV = np.meshgrid(l_boom_grid, S_v_grid)

        obj_vals2 = np.zeros_like(L2)
        feasible2 = np.zeros_like(L2, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([L2[i, j], x_opt[1], SV[i, j], x_opt[3], x_opt[4]])
                xdict = {'xvars': x}
                funcs, _ = Optimizer.user_function(xdict)
                obj_vals2[i, j] = funcs['obj']
                cons = {k: v for k, v in funcs.items() if k != 'obj' and k != 'incidence_max' and k != 'yaw_control'}
                feasible2[i, j] = all(v >= -1e-6 for v in cons.values())

        contour2 = ax.contourf(L2, SV, obj_vals2, levels=60, cmap='viridis', alpha=0.8)
        plt.colorbar(contour2, ax=ax, label='Objective')
        ax.contourf(L2, SV, ~feasible2, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
        ax.contour(L2, SV, feasible2.astype(float), levels=[0.5], colors=['red'], linewidths=2)
        ax.plot(x_opt[0], x_opt[2], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
        ax.set_xlabel('Boom Length [m]')
        ax.set_ylabel('V-tail Area [m$^2$]')
        ax.set_title('l_boom vs S_v (S_h, tpr fixed)')
        ax.legend()

        # S_h vs S_v
        ax = axes[1, 0]
        SH2, SV2 = np.meshgrid(S_h_grid, S_v_grid)

        obj_vals3 = np.zeros_like(SH2)
        feasible3 = np.zeros_like(SH2, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([x_opt[0], SH2[i, j], SV2[i, j], x_opt[3], x_opt[4]])
                xdict = {'xvars': x}
                funcs, _ = Optimizer.user_function(xdict)
                obj_vals3[i, j] = funcs['obj']
                cons = {k: v for k, v in funcs.items() if k != 'obj' and k != 'incidence_max' and k != 'yaw_control'}
                feasible3[i, j] = all(v >= -1e-6 for v in cons.values())

        contour3 = ax.contourf(SH2, SV2, obj_vals3, levels=60, cmap='viridis', alpha=0.8)
        plt.colorbar(contour3, ax=ax, label='Objective')
        ax.contourf(SH2, SV2, ~feasible3, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
        ax.contour(SH2, SV2, feasible3.astype(float), levels=[0.5], colors=['red'], linewidths=2)
        ax.plot(x_opt[1], x_opt[2], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
        ax.set_xlabel('H-tail Area [m$^2$]')
        ax.set_ylabel('V-tail Area [m$^2$]')
        ax.set_title('S_h vs S_v (l_boom, tpr fixed)')
        ax.legend()

        # Slice 4: tpr_h vs tpr_v
        ax = axes[1, 1]
        tpr_h_grid = np.linspace(0.40, 0.90, n_grid)
        tpr_v_grid = np.linspace(0.40, 0.95, n_grid)
        TPH, TPV = np.meshgrid(tpr_h_grid, tpr_v_grid)

        obj_vals4 = np.zeros_like(TPH)
        feasible4 = np.zeros_like(TPH, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([x_opt[0], x_opt[1], x_opt[2], TPH[i, j], TPV[i, j]])
                xdict = {'xvars': x}
                funcs, _ = Optimizer.user_function(xdict)
                obj_vals4[i, j] = funcs['obj']
                cons = {k: v for k, v in funcs.items() if k != 'obj' and k != 'incidence_max' and k != 'yaw_control'}
                feasible4[i, j] = all(v >= -1e-6 for v in cons.values())

        contour4 = ax.contourf(TPH, TPV, obj_vals4, levels=60, cmap='viridis', alpha=0.8)
        plt.colorbar(contour4, ax=ax, label='Objective')
        ax.contourf(TPH, TPV, ~feasible4, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
        ax.contour(TPH, TPV, feasible4.astype(float), levels=[0.5], colors=['red'], linewidths=2)
        ax.plot(x_opt[3], x_opt[4], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
        ax.set_xlabel('H-tail Taper Ratio')
        ax.set_ylabel('V-tail Taper Ratio')
        ax.set_title('tpr_h vs tpr_v (l_boom, S_h, S_v fixed)')
        ax.legend()

        plt.tight_layout()
        plot_file = os.path.join(PLOTS_DIR, 'design_space.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        n_grid = 40
        l_boom_grid = np.linspace(1.0, 2.5, n_grid)
        S_h_grid = np.linspace(0.35, 1.2, n_grid)
        L, SH = np.meshgrid(l_boom_grid, S_h_grid)

        obj_vals = np.zeros_like(L)
        feasible = np.zeros_like(L, dtype=bool)

        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([L[i, j], SH[i, j], x_opt[2], x_opt[3], x_opt[4]])
                xdict = {'xvars': x}
                funcs, _ = Optimizer.user_function(xdict)
                obj_vals[i, j] = funcs['obj']
                cons = {k: v for k, v in funcs.items() if k != 'obj'}
                feasible[i, j] = all(v >= -1e-6 for v in cons.values())

        surf = ax.plot_surface(L, SH, obj_vals, facecolors=plt.cm.viridis((obj_vals - obj_vals.min()) / (obj_vals.max() - obj_vals.min())),
                            alpha=0.8, linewidth=0.5, edgecolor='gray')

        infeasible_obj = np.where(~feasible, obj_vals, np.nan)
        ax.plot_wireframe(L, SH, infeasible_obj, color='red', alpha=0.3, linewidth=0.5)

        xdict_opt = {'xvars': x_opt}
        funcs_opt, _ = Optimizer.user_function(xdict)
        ax.scatter([x_opt[0]], [x_opt[1]], [funcs_opt['obj']], color='white', s=200, marker='*',
                edgecolor='black', linewidth=2, label='Optimal', zorder=10)

        ax.set_xlabel('Boom Length [m]')
        ax.set_ylabel('H-tail Area [m$^2$]')
        ax.set_zlabel('Objective')
        ax.set_title('T-Tail v5.0 Objective Surface\n(S_v, tpr_h, tpr_v fixed at optimal)')
        ax.legend()

        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(obj_vals)
        plt.colorbar(mappable, ax=ax, shrink=0.5, label='Objective')

        plt.tight_layout()
        plot_file = os.path.join(PLOTS_DIR, 'design_space_3d.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")
        
Optimizer.main()