# rst Imports
from pyoptsparse import OPT, Optimization, History
import numpy as np
import argparse
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")  # headless plotting

# Aerodynamic Constraints

def turn(V, CD0, WS, phi, k, rho):
    q = 0.5 * rho * V**2
    n = 1.0 / np.cos(np.deg2rad(phi))
    return q * CD0 / WS + WS * (n**2 * k / q)

def cruise(V, WS, CD0, k, rho):
    q = 0.5 * rho * V**2
    return q * CD0 / WS + WS * (k / q)

def takeoff(WS, CLmax):
    CL_TO = CLmax / 1.21
    return WS / (CL_TO * 169.339)

def climb(CL, CD0, k, grad):
    grad = np.deg2rad(grad)
    return CD0 / CL + CL * k + np.sin(grad)/np.cos(grad)

def landing(CLmax):
    return (CLmax / 80) * 88 * 47.880258888889

def landing_constraint_TS(S, CLmax):
   
    WS_land_limit = landing(CLmax)

    W0_guess = 30.0
    tol = 1e-3
    maxIter = 20
    T_guess = 100

    for _ in range(maxIter):
        
        TW_guess = T_guess / (W0_guess * g)
        
        
        W0, _, _, _, _, _, _, _, _, _ = calculate_weight(
            S, TW_guess, WS_land_limit, V_cruise_init, CLcruise / (CD0_init + k_init * CLcruise**2)
        )
        
        T_new = WS_land_limit * S
        if abs(T_new - T_guess) / (T_guess + 1e-12) < tol:
            T_guess = T_new
            break
        
        W0_guess = 0.5 * W0 + 0.5 * W0_guess
        T_guess = 0.5 * T_new + 0.5 * T_guess

    return T_guess

def updateCD0_and_weights(S_ref, V, rho, mu, AR_w, lambda_w, AR_ht, lambda_ht,
                          AR_vt, lambda_vt, L_f, D_f, CHT, CVT, L_HT, L_VT, Lambda_tmax):
    """Calculate CD0, e, k, tail areas, and structural weights."""
    
    a = 340
    mach = V / a
    
    # Wing geometry
    b_w = np.sqrt(AR_w * S_ref)
    c_r_w = 2 * S_ref / (b_w * (1 + lambda_w))
    MAC_w = (2/3) * c_r_w * ((1 + lambda_w + lambda_w**2) / (1 + lambda_w))
    tau_w = 1
    S_exp_w = S_ref
    S_wet_w = 2 * S_exp_w * (1 + 0.25 * t_c_w * (1 + tau_w * lambda_w) / (1 + lambda_w))
    
    # Tail areas
    S_ht = CHT * MAC_w * S_ref / L_HT
    S_vt = CVT * b_w * S_ref / L_VT
    
    # Horizontal tail
    b_ht = np.sqrt(AR_ht * S_ht)
    c_r_ht = 2 * S_ht / (b_ht * (1 + lambda_ht))
    S_exp_ht = S_ht
    S_wet_ht = 2 * S_exp_ht * (1 + 0.25 * t_c_t * (1 + lambda_ht) / (1 + lambda_ht))
    
    # Vertical tail
    b_vt = np.sqrt(AR_vt * S_vt)
    c_r_vt = 2 * S_vt / (b_vt * (1 + lambda_vt))
    S_exp_vt = S_vt
    S_wet_vt = 2 * S_exp_vt * (1 + 0.25 * t_c_t * (1 + lambda_vt) / (1 + lambda_vt))
    
    # Fuselage
    lambda_F = L_f / D_f
    S_wet_f = np.pi * D_f * L_f * ((1 - 2/lambda_F)**(2/3)) * np.sqrt(1 + 1/(lambda_F**2))
    
    # Skin friction
    Cf = lambda Re: 0.455 / (max(np.log10(max(Re, 1.0)), 1e-9)**2.58)
    Cf_w = Cf(rho * V * MAC_w / mu)
    Cf_ht = Cf(rho * V * c_r_ht / mu)
    Cf_vt = Cf(rho * V * c_r_vt / mu)
    Cf_f = Cf(rho * V * L_f / mu)
    
    # Form factors
    FF = lambda t_c, x_t, M: (1 + 0.6/x_t * t_c + 100 * t_c**4) * (1.34 * M**0.18)
    FF_w = FF(t_c_w, x_t_w, mach)
    FF_t = FF(t_c_t, x_t_t, mach)
    FF_f = 1 + 60 / (lambda_F**3) + lambda_F / 400
    
    # Parasite drag
    CD0 = (Cf_w * FF_w * S_wet_w / S_ref +
           Cf_ht * FF_t * S_wet_ht / S_ref +
           Cf_vt * FF_t * S_wet_vt / S_ref +
           Cf_f * FF_f * S_wet_f / S_ref + 0.035)
    
    # Oswald efficiency
    e = 2 / (2 - AR_w + np.sqrt(4 + AR_w**2 * (1 + np.tan(Lambda_tmax)**2)))
    k = 1 / (np.pi * AR_w * e)
    
    # Reference weights
    S_w_ref = 1.542
    S_ht_ref = 0.204
    S_vt_ref = 0.102
    W_wing_ref = 2.993
    W_ht_ref = 0.454
    W_vt_ref = 0.227
    
    # Scaled weights
    W_wing = W_wing_ref * (S_ref / S_w_ref)
    W_ht = W_ht_ref * (S_ht / S_ht_ref)
    W_vt = W_vt_ref * (S_vt / S_vt_ref)
    W_emp = W_ht + W_vt
    
    return CD0, e, k, S_ht, S_vt, W_wing, W_emp

def calculate_weight(S, TW_opt, WS_opt, V_cruise, LD):
    """
    Iteratively calculate aircraft weight given S, optimal TW, and WS.
    Returns: W0, T, CD0, e, k, W_wing, W_emp, W_engine, W_tank, W_f
    """
    
    W0 = 26.75  # Initial guess
    tol = 1e-3
    maxIter = 50
    alpha = 0.3
    
    CD0 = CD0_init
    e = e_init
    k = k_init
    
    for iteration in range(maxIter):
        # Fuel fraction
        W5_W4_range = 1 / np.exp((Range_req * g * phi_eq) / 
                                  (w_net * AFR_st * LD * eta_prop_mechanical))
        W5_W4_end = 1 / np.exp((Endurance_req * g * phi_eq) / 
                               (w_net * AFR_st * LD * V_cruise * eta_prop_mechanical))
        W5_W4 = min(W5_W4_range, W5_W4_end)
        W7_W0 = W5_W4 * 0.99
        Wf_W0 = (1 - W7_W0) * fuel_trap_factor
        W_f = Wf_W0 * W0 * 1.5
        
        # Tank weight
        V_F = W_f / rho_methane
        W_tank = 2 * t_tank * rho_tank * ((V_F / r_tank) - ((4/3) * np.pi * (r_tank**2)))
        
        # Wing loading and thrust
        W0_force = W0 * g
        WS_design = WS_opt
        TW_design = TW_opt
        T = TW_design * W0_force
        
        # Update aerodynamics and weights
        CD0, e, k, S_ht, S_vt, W_wing, W_emp = updateCD0_and_weights(
            S, V_cruise, rho, mu, AR_w, lambda_w, AR_ht, lambda_ht,
            AR_vt, lambda_vt, L_f, D_f, CHT, CVT, L_HT, L_VT, Lambda_tmax)
        
        # Engine weight
        P_cruise_W = T * V_cruise
        P_cruise_kW = P_cruise_W / 1000
        W_engine = 0.3654 * (P_cruise_kW)**0.9564
        
        # Total weight
        W0_actual = W_f + W_shell + W_components + W_engine + W_tank + W_wing + W_emp + W_payload
        
        # Check convergence
        err = abs(W0_actual - W0) / W0
        if err < tol:
            W0 = (1 - alpha) * W0 + alpha * W0_actual
            break
        
        W0 = (1 - alpha) * W0 + alpha * W0_actual
    
    return W0, T, CD0, e, k, W_wing, W_emp, W_engine, W_tank, W_f


# Aircraft Constraints

rho = 1.225
g = 9.81
mu = 1.789e-5

AR_w = 7.328
lambda_w = 0.75
t_c_w = 0.128
x_t_w = 0.306

AR_ht = 7.000
lambda_ht = 1
AR_vt = 12.158
lambda_vt = 1
t_c_t = 0.12
x_t_t = 0.3

CLmax = 1.76
CLclimb = 1.30
CLcruise = 0.80
bank_angle = 40
climb_grad = 5

L_f = 1.75
D_f = 0.5
t_tank = 0.0015875
rho_tank = 8000
rho_methane = 422
r_tank = 0.0508

CHT = 0.50
CVT = 0.03
L_HT = 1.8
L_VT = 1.8

Lambda_tmax = np.deg2rad(0)

Range_req = 7927
Endurance_req = 725

W1_W0 = 0.990
W2_W1 = 0.995
W3_W2 = 0.970
W4_W3 = 0.985
W6_W5 = 0.990
W7_W6 = 0.995
fuel_trap_factor = 1.06

W_shell = 5.490
W_components = 7.66
W_payload = 5

phi_eq = 0.26
AFR_st = 17.25
w_net = 129827
eta_prop_mechanical = 0.5

CD0_init = 0.0531
e_init = 0.881
k_init = 1 / (np.pi * AR_w * e_init)

V_stall_init = 8.46
V_cruise_init = 11.0

WS_land_limit = landing(CLmax)

# Disable VSPAERO Viewer
try:
    from openvsp import VSPViewer
    VSPViewecdr.disable()
except ImportError:
    pass

# rst Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default="slsqp")
# parser.add_argument("--opt", type=str, default="ipopt")
args = parser.parse_args()

# TW-WS Optimization

# rst Callback function
def userfunc_TW_WS(xdict):
    WS, TW = xdict["base_TW_WS"]  # Extract array

    funcs = {}
    funcs["obj_TW_WS"]         = -1000 * WS - (1.0 / TW)
    funcs["con_turn_TW_WS"]    = turn(V_cruise_init, CD0_init, WS, bank_angle, k_init, rho) - TW
    funcs["con_cruise_TW_WS"]  = cruise(V_cruise_init, WS, CD0_init, k_init, rho) - TW
    funcs["con_takeoff_TW_WS"] = takeoff(WS, CLmax) - TW
    funcs["con_climb_TW_WS"]   = climb(CLclimb, CD0_init, k_init, climb_grad) - TW
    funcs["con_landing_TW_WS"] = (WS - WS_land_limit) / WS_land_limit
    return funcs


# rst Sensitivity function
def userfuncsens_TW_WS(xdict, funcs):
    WS, TW = xdict["base_TW_WS"]  # Extract array

    funcsSens = {}
    funcsSens["obj_TW_WS"] = {"base_TW_WS": np.array([-1000.0, 1.0 / TW**2])}

    # Derivatives

    q_turn   = 0.5 * rho * V_cruise_init**2
    n        = 1.0 / np.cos(np.deg2rad(bank_angle))
    q_cruise = 0.5 * rho * V_cruise_init**2

    dturn    = -q_turn * CD0_init / WS**2 + (n**2 * k_init) / q_turn
    dcruise  = -q_cruise * CD0_init / WS**2 + k_init / q_cruise
    dtakeoff = 1.0 / (CLmax / 1.21 * 169.339)
    dclimb   = 0.0
    dland    = 1.0 / WS_land_limit

    # Constraints

    funcsSens["con_turn_TW_WS"]    = {"base_TW_WS": np.array([dturn,   -1.0])}
    funcsSens["con_cruise_TW_WS"]  = {"base_TW_WS": np.array([dcruise, -1.0])}
    funcsSens["con_takeoff_TW_WS"] = {"base_TW_WS": np.array([dtakeoff,-1.0])}
    funcsSens["con_climb_TW_WS"]   = {"base_TW_WS": np.array([dclimb,  -1.0])}
    funcsSens["con_landing_TW_WS"] = {"base_TW_WS": np.array([dland, 0.0])}

    return funcsSens


# rst Optimization problem
optProb_TW_WS = Optimization("TW - WS Optimization", userfunc_TW_WS)

# rst Add objective
optProb_TW_WS.addObj("obj_TW_WS")

# rst Add design variables
optProb_TW_WS.addVarGroup(name="base_TW_WS", nVars=2, varType="c", value=[50, 0.4], lower=[1e-9, 1e-9], upper=[WS_land_limit * 1.1, 1.0], scale=1.0)

# rst Add constraints
optProb_TW_WS.addCon("con_turn_TW_WS",    upper=0.0)
optProb_TW_WS.addCon("con_cruise_TW_WS",  upper=0.0)
optProb_TW_WS.addCon("con_takeoff_TW_WS", upper=0.0)
optProb_TW_WS.addCon("con_climb_TW_WS",   upper=0.0)
optProb_TW_WS.addCon("con_landing_TW_WS", upper=0.0)


# rst Instantiate optimizer
optOptions = {}
opt_TW_WS = OPT(args.opt, options=optOptions)

# rst Solve
sol_TW_WS = opt_TW_WS(optProb_TW_WS, sens=userfuncsens_TW_WS, storeHistory="opt_TW_WS.hst")
print(sol_TW_WS)

# Optimized Values
WS_opt = sol_TW_WS.xStar["base_TW_WS"][0]
TW_opt = sol_TW_WS.xStar["base_TW_WS"][1]

# T-S Optimization

weights = {}
iteration = {'count': 0}

# rst Callback function
def userfunc_T_S(xdict):
    S, T = xdict["base_T_S"]

    CDcruise = CD0_init + k_init * CLcruise**2
    LD = CLcruise / CDcruise

    W0, _, CD0, e, k, W_wing, W_emp, W_engine, W_tank, W_f = calculate_weight(
        S, TW_opt, WS_opt, V_cruise_init, LD)
    
    weights['W0'] = W0
    weights['CD0'] = CD0
    weights['e'] = e
    weights['k'] = k
    weights['W_wing'] = W_wing
    weights['W_emp'] = W_emp
    weights['W_engine'] = W_engine
    weights['W_tank'] = W_tank
    weights['W_f'] = W_f
    
    W0_force = W0 * g
    WS_actual = W0_force / S
    TW_actual = T / W0_force

    funcs = {}
    funcs["obj_T_S"] = S + 1e-6 * T

    funcs["con_turn_T_S"] = turn(V_cruise_init, CD0, WS_actual, bank_angle, k, rho) - TW_actual
    funcs["con_cruise_T_S"] = cruise(V_cruise_init, WS_actual, CD0, k, rho) - TW_actual
    funcs["con_takeoff_T_S"] = takeoff(WS_actual, CLmax) - TW_actual
    funcs["con_climb_T_S"] = climb(CLclimb, CD0, k, climb_grad) - TW_actual
    funcs["con_landing_T_S"] = (WS_actual - WS_land_limit) / WS_land_limit
    funcs["con_WS_T_S"] = (WS_actual - WS_opt) / WS_opt
    funcs["con_TW_T_S"] = (TW_actual - TW_opt) / TW_opt
    return funcs


# rst Sensitivity function for second problem
def userfuncsens_T_S(xdict, funcs):
    S, T = xdict["base_T_S"]

    CDcruise = CD0_init + k_init * CLcruise**2
    LD = CLcruise / CDcruise
    
    step = 1e-5 * S

    W0_temp, T_temp, CD0_temp, e_temp, k_temp, _, _, _, _, _ = calculate_weight(
        S + step, TW_opt, WS_opt, V_cruise_init, LD)
    
    W0 = weights['W0']
    CD0 = weights['CD0']
    k = weights['k']
    
    dW0 = (W0_temp - W0) / step
    dT = (T_temp - T) / step
    dCD0 = (CD0_temp - CD0) / step
    dk = (k_temp - k) / step
    
    W0_force = W0 * g
    WS_actual = W0_force / S
    TW_actual = T / W0_force
    
    dWS = g / S**2 * (S * dW0 - W0)
    dTW = (-T * dW0) / (W0_force ** 2)
    
    q_turn = 0.5 * rho * V_cruise_init**2
    n = 1.0 / np.cos(np.deg2rad(bank_angle))
    q_cruise = 0.5 * rho * V_cruise_init**2
    
    dturn = (-q_turn * CD0 / WS_actual**2 + (n**2 * k) / q_turn) * dWS + \
               q_turn / WS_actual * dCD0 + WS_actual * n**2 / q_turn * dk
    
    dcruise = (-q_cruise * CD0 / WS_actual**2 + k / q_cruise) * dWS + \
                 q_cruise / WS_actual * dCD0 + WS_actual / q_cruise * dk
    
    CL_TO = CLmax / 1.21
    dtakeoff = (1 / (CL_TO * 169.339)) * dWS
    
    dclimb = 0.0

    dland = (g / (WS_land_limit * S**2)) * (S * dW0 - W0)

    funcsSens = {}
    funcsSens["obj_T_S"] = {"base_T_S": np.array([1.0, 1e-6])}

    funcsSens["con_turn_T_S"]    = {"base_T_S": np.array([dturn, -1.0 / W0])}
    funcsSens["con_cruise_T_S"]  = {"base_T_S": np.array([dcruise, -1.0 / W0])}
    funcsSens["con_takeoff_T_S"] = {"base_T_S": np.array([dtakeoff, -1.0 / W0])}
    funcsSens["con_climb_T_S"]   = {"base_T_S": np.array([dclimb, -1.0 / W0])}
    funcsSens["con_landing_T_S"] = {"base_T_S": np.array([dland, 0.0])}
    funcsSens["con_WS_T_S"]      = {"base_T_S": np.array([dWS / WS_opt, 0.0])}
    funcsSens["con_TW_T_S"]      = {"base_T_S": np.array([dTW / TW_opt, 1.0 / (TW_opt * W0_force)])}

    return funcsSens


optProb_T_S = Optimization("T - S Optimization", userfunc_T_S)
optProb_T_S.addObj("obj_T_S")

W_guess = 26.75
S_init = (W_guess * g) / WS_opt
T_init = TW_opt * W_guess * g

optProb_T_S.addVarGroup(name="base_T_S", nVars=2, varType="c", value=[5.0, 100], 
                     lower=[1e-9, 1e-9], upper=[5.0, 200.0], scale=1.0)

optProb_T_S.addCon("con_turn_T_S",    upper=0.0)
optProb_T_S.addCon("con_cruise_T_S",  upper=0.0)
optProb_T_S.addCon("con_takeoff_T_S", upper=0.0)
optProb_T_S.addCon("con_climb_T_S",   upper=0.0)
optProb_T_S.addCon("con_landing_T_S", upper=0.0)
optProb_T_S.addCon("con_WS_T_S", upper=1e-4, lower=-1e-4)
optProb_T_S.addCon("con_TW_T_S", upper=1e-4, lower=-1e-4)

# rst Instantiate optimizer
opt_T_S = OPT(args.opt, options={})

# rst Solve
sol_T_S = opt_T_S(optProb_T_S, sens=userfuncsens_T_S, storeHistory="opt_T_S.hst")
print(sol_T_S)


# Load the history file
optHist_TW_WS = History("opt_TW_WS.hst")
values_TW_WS = optHist_TW_WS.getValues()

# Plot contours of the objective and the constraint boundary
WS_path = values_TW_WS["base_TW_WS"][:, 0]
TW_path = values_TW_WS["base_TW_WS"][:, 1]

WS = np.linspace(1e-9, 120, 1200)
TW = np.linspace(1e-9, 1.0, 1000)
WSp, TWp = np.meshgrid(WS, TW)

objFunc_TW_WS     = (-WSp + 100 * TWp) / 100
con_turn_TW_WS    = turn(V_cruise_init, CD0_init, WSp, bank_angle, k_init, rho) - TWp
con_cruise_TW_WS  = cruise(V_cruise_init, WSp, CD0_init, k_init, rho) - TWp
con_takeoff_TW_WS = takeoff(WSp, CLmax) - TWp
con_climb_TW_WS   = climb(CLclimb, CD0_init, k_init, climb_grad) - TWp
con_landing_TW_WS = WSp - WS_land_limit

fig_TW_WS, ax_TW_WS = plt.subplots()
ax_TW_WS.contour(WSp, TWp, objFunc_TW_WS, cmap="plasma", levels=40)
heatmap = ax_TW_WS.pcolormesh(WSp, TWp, objFunc_TW_WS, cmap="plasma", shading='auto', alpha=0.8,
                                vmin=-1.0, vmax=1.0)
ax_TW_WS.contour(WSp, TWp, con_turn_TW_WS,    levels=[0.0], colors="r")
ax_TW_WS.contour(WSp, TWp, con_cruise_TW_WS,  levels=[0.0], colors="g")
ax_TW_WS.contour(WSp, TWp, con_takeoff_TW_WS, levels=[0.0], colors="b")
ax_TW_WS.contour(WSp, TWp, con_climb_TW_WS,   levels=[0.0], colors="m")
ax_TW_WS.contour(WSp, TWp, con_landing_TW_WS, levels=[0.0], colors="k")
ax_TW_WS.contourf(WSp, TWp, np.maximum.reduce([
    con_turn_TW_WS,
    con_cruise_TW_WS,
    con_takeoff_TW_WS,
    con_climb_TW_WS,
    con_landing_TW_WS
]), levels=[0.0, 1e9], colors="red", alpha=0.8)

# Plot the path of the optimizer
ax_TW_WS.plot(values_TW_WS["base_TW_WS"][:, 0], values_TW_WS["base_TW_WS"][:, 1], "-o", markersize=6, clip_on=False,
        label="Optimization Path")
ax_TW_WS.plot(values_TW_WS["base_TW_WS"][0, 0], values_TW_WS["base_TW_WS"][0, 1], "s", markersize=8, clip_on=False,
        label="Initial Value")
ax_TW_WS.plot(values_TW_WS["base_TW_WS"][-1, 0], values_TW_WS["base_TW_WS"][-1, 1], "^", markersize=8, clip_on=False,
        label="Optimal Value")
ax_TW_WS.set_xlabel("Wing Loading (W/S) [N/m²]", fontsize=10)
ax_TW_WS.set_ylabel("Thrust-to-Weight (T/W) [-]", fontsize=10)
ax_TW_WS.set_title("TW-WS Optimization", fontsize=12)
ax_TW_WS.set_ylim([0, 0.6])
ax_TW_WS.legend(loc="upper right")
legend = ax_TW_WS.legend(loc='upper center',
                        bbox_to_anchor=(0.5, -0.25),
                        ncol=3,
                        fontsize=9,
                        frameon=True,
                        framealpha=0.9,
                        title_fontsize=9)
plt.tight_layout(pad=0.5)
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
cbar = fig_TW_WS.colorbar(heatmap, ax=ax_TW_WS,
                         orientation='horizontal',
                         fraction=0.04,
                         pad=0.30)
cbar.set_label('Objective Value', fontsize=10, labelpad=2.5)
fig_TW_WS.savefig("OptimizerPath (TW - WS).png")


# Load the history file
optHist_T_S = History("opt_T_S.hst")
values_T_S = optHist_T_S.getValues()

# Plot contours of the objective and the constraint boundary
S_path = values_T_S["base_T_S"][:, 0]
T_path = values_T_S["base_T_S"][:, 1]

Sp = np.linspace(1e-9, 6.0, 600)
Tp = np.linspace(1e-9, 200, 2000)

T_con_turn_T_S = []
T_con_cruise_T_S = []
T_con_takeoff_T_S = []
T_con_climb_T_S = []
T_con_landing_T_S = []

for S_val in Sp:
    CDcruise = CD0_init + k_init * CLcruise**2
    LD = CLcruise / CDcruise
    W0_temp, T_temp, CD0_temp, e_temp, k_temp, _, _, _, _, _ = calculate_weight(
        S_val, TW_opt, WS_opt, V_cruise_init, LD)
    W0_force = W0_temp * g
    WS_val = W0_force / S_val
    W0_landing, T_landing, _, _, _, _, _, _, _, _ = calculate_weight(
        S_val, TW_opt, WS_land_limit, V_cruise_init, LD)
    
    T_con_turn_T_S.append(turn(V_cruise_init, CD0_temp, WS_val, bank_angle, k_temp, rho) * W0_force)
    T_con_cruise_T_S.append(cruise(V_cruise_init, WS_val, CD0_temp, k_temp, rho) * W0_force)
    T_con_takeoff_T_S.append(takeoff(WS_val, CLmax) * W0_force)
    T_con_climb_T_S.append(climb(CLclimb, CD0_temp, k_temp, climb_grad) * W0_force)   
    T_con_landing_T_S.append(T_landing)

T_con_turn_T_S    = np.array(T_con_turn_T_S)
T_con_cruise_T_S  = np.array(T_con_cruise_T_S)
T_con_takeoff_T_S = np.array(T_con_takeoff_T_S)
T_con_climb_T_S   = np.array(T_con_climb_T_S)
T_con_landing_T_S = np.array(T_con_landing_T_S)

Smesh, Tmesh = np.meshgrid(Sp, Tp)
objFunc_T_S  = (Smesh + 1e-6 * Tmesh) / 6

fig_T_S, ax_T_S = plt.subplots()
ax_T_S.contour(Smesh, Tmesh, objFunc_T_S, cmap="plasma", levels=40)
heatmap = ax_T_S.pcolormesh(Sp, Tp, objFunc_T_S, cmap="plasma", shading='auto', alpha=0.8,
                                vmin=0.0, vmax=1.0)
ax_T_S.plot(Sp, T_con_turn_T_S, 'r-')
ax_T_S.plot(Sp, T_con_cruise_T_S, 'g-')
ax_T_S.plot(Sp, T_con_takeoff_T_S, 'b-')
ax_T_S.plot(Sp, T_con_climb_T_S, 'm-')
ax_T_S.plot(Sp, T_con_landing_T_S, 'k-')


T_con_min_T_S = np.maximum.reduce([
    T_con_turn_T_S,
    T_con_cruise_T_S,
    T_con_takeoff_T_S,
    T_con_climb_T_S,
])
ax_T_S.fill_between(Sp, 0, T_con_min_T_S, color='red', alpha=0.8)
T_con_max_T_S = np.maximum.reduce([
    T_con_turn_T_S,
    T_con_cruise_T_S,
    T_con_takeoff_T_S,
    T_con_climb_T_S,
    T_con_landing_T_S
])
T_con_max_T_S = np.minimum(T_con_max_T_S, 200)
ax_T_S.fill_between(Sp, T_con_max_T_S, 200, color='red', alpha=0.8)


# Plot the path of the optimizer
ax_T_S.plot(values_T_S["base_T_S"][:, 0], values_T_S["base_T_S"][:, 1], "-o", markersize=6, clip_on=False,
        label="Optimization Path")
ax_T_S.plot(values_T_S["base_T_S"][0, 0], values_T_S["base_T_S"][0, 1], "s", markersize=8, clip_on=False,
        label="Initial Value")
ax_T_S.plot(values_T_S["base_T_S"][-1, 0], values_T_S["base_T_S"][-1, 1], "^", markersize=8, clip_on=False,
        label="Optimal Value")
ax_T_S.set_xlabel("Planform Area (S) [m²]", fontsize=10)
ax_T_S.set_ylabel("Thrust (T) [N]", fontsize=10)
ax_T_S.set_title("T-S Optimization", fontsize=12)
ax_T_S.legend(loc="upper right")
legend = ax_T_S.legend(loc='upper center',
                        bbox_to_anchor=(0.5, -0.25),
                        ncol=3,
                        fontsize=9,
                        frameon=True,
                        framealpha=0.9,
                        title_fontsize=9)
plt.tight_layout(pad=0.5)
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
cbar = fig_T_S.colorbar(heatmap, ax=ax_T_S,
                         orientation='horizontal',
                         fraction=0.04,
                         pad=0.30)
cbar.set_label('Objective Value', fontsize=10, labelpad=2.5)
ax_T_S.set_xlim([0, 6])
ax_T_S.set_ylim([0, 200])

fig_T_S.savefig("OptimizerPath (T - S).png")
