import os
import csv
import numpy as np
from typing import Dict, Optional, List

import matplotlib
matplotlib.use("Agg")  # Headless plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon, FancyBboxPatch, Patch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors


# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

for directory in [PLOTS_DIR, OUTPUT_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


__all__ = [
    'plot_convergence_history',
    'plot_design_space_2d',
    'plot_design_space_3d',
    'plot_design_space_4d',
    'generate_design_space_samples',
    'plot_parallel_coordinates',
    'plot_constraint_margins',
    'plot_tail_geometry',
    'plot_weight_breakdown',
    'plot_drag_breakdown',
    'plot_sensitivity_analysis',
    'plot_trade_studies',
    'plot_monte_carlo_robustness',
    'plot_optimization_path_3d',
]


# rst Objective Function
def _objfunc_for_viz(xdict):

    x = xdict['xvars']
    l_boom, S_h, S_v, tpr_h, tpr_v = x

    q = 0.5 * AC.rho * AC.V_cruise**2

    # Geometry
    h_geom = tail_geometry(S_h, TAIL.AR_h, tpr_h)
    v_geom = tail_geometry(S_v, TAIL.AR_v, tpr_v)
    l_h, l_v, h_vt = get_moment_arms(l_boom, S_v, tpr_v, TAIL)

    # Tail volumes
    V_H = l_h * S_h / (AC.MAC * AC.S)
    V_V = l_v * S_v / (AC.S * AC.b)

    # Weights
    W_h = tail_weight(S_h, TAIL.AR_h, tpr_h)
    W_v = tail_weight(S_v, TAIL.AR_v, tpr_v)
    m_tail = (W_h + W_v) / 9.81

    # Boom 
    omega = 2 * np.pi * 7.0
    I_theta = m_tail * (h_vt / 2 * np.sin(TAIL.swp_v)) ** 2
    GJ_freq = I_theta * m_tail * l_boom * omega ** 2
    Y_v = q * S_v * 0.7
    T_boom = h_vt / 2 * Y_v
    GJ_twist = T_boom * l_boom / np.deg2rad(1)
    GJ_req = max(GJ_freq, GJ_twist)
    boom = get_boom_properties(GJ_req, STRUCT, l_boom, S_h)
    W_boom = boom['weight_per_m'] * l_boom

    # Direct TOGW coupling 
    W_tail = W_h + W_v + W_boom
    m_tail = W_tail / 9.81  # kg
    TOGW = m_fixed + m_tail  # kg 

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

    # Induced drag (trim requirement)
    l_h, _, _ = get_moment_arms(l_boom, S_v, tpr_v, TAIL)
    V_H = l_h * S_h / (AC.MAC * AC.S)

    C_m_wing = AC.C_mowf + C_L * (AC.h_cg - AC.h_ac_wing)
    C_L_tail = (C_m_wing / (V_H * TAIL.eta_h)) if V_H > 0.01 else 0

    e_h = 0.90
    D_induced = (C_L_tail**2 / (np.pi * e_h * TAIL.AR_h)) * q * S_h

    D_total = D_parasite + D_induced
    
    # References
    W_ref = 30.0    # N
    D_ref = 2.0     # N
    l_ref = 1.5     # m 

    # Weights
    w_weight = 0.60
    w_drag   = 0.25
    w_boom   = 0.15

    J = (w_drag * (D_total / D_ref) +
        w_weight * (W_tail / W_ref) +
        w_boom * (l_boom / l_ref) ** 2)
    
    funcs = {'obj': J}
    fail = False
    return funcs, fail



def _userfunc_for_viz(xdict):
    obj_funcs, fail1 = _objfunc_for_viz(xdict)
    con_funcs, fail2 = confunc(xdict)
    funcs = {**obj_funcs, **con_funcs}
    fail = fail1 or fail2
    return funcs, fail


def plot_convergence_history(history_file: str):

    try:
        from pyoptsparse import History
        hist = History(history_file)
        values = hist.getValues()
    except Exception as e:
        print(f"  Warning: Could not load history file: {e}")
        return

    if 'obj' not in values:
        print("  Warning: No objective history found in history file")
        return

    obj_vals = np.asarray(values['obj']).flatten()
    n_iters = len(obj_vals)

    if n_iters < 2:
        print(f"  Warning: Only {n_iters} iteration(s) in history, skipping convergence plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'T-Tail v5.0 Optimization Convergence ({n_iters} iterations)', fontsize=14, fontweight='bold')

    # 1. Objective vs iteration
    ax = axes[0, 0]
    ax.plot(obj_vals, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title('Objective History')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(1, n_iters-1)])

    # 2. Design variables vs iteration
    ax = axes[0, 1]
    if 'xvars' in values:
        xvars = values['xvars']
        var_names = ['l_boom', 'S_h', 'S_v', 'tpr_h', 'tpr_v']
        for i, name in enumerate(var_names):
            ax.plot(xvars[:, i], label=name, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Design Variable Value')
        ax.set_title('Design Variables History')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. Max constraint violation vs iteration
    ax = axes[0, 2]
    constraint_names = [
        'boom_deflection', 'boom_stress', 'vtail_bending', 'vtail_torsion', 
        'vtail_vm', 'torsion', 'pitch_trim', 'tail_no_stall', 'deep_stall', 
        'static_margin', 'pitch_control', 'yaw_control',
        'v_h_lower', 'v_v_lower', 'v_v_upper'
    ]
    max_violations = None
    for con_name in constraint_names:
        if con_name in values:
            con_vals = np.asarray(values[con_name]).flatten()
            violations = np.minimum(con_vals, 0)
            if max_violations is None:
                max_violations = np.zeros(len(con_vals))
            min_len = min(len(max_violations), len(violations))
            max_violations[:min_len] = np.minimum(max_violations[:min_len], violations[:min_len])

    if max_violations is not None and len(max_violations) > 0:
        ax.plot(-max_violations, 'r-o', linewidth=2, markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Constraint Violation')
        ax.set_title('Constraint Violation History')
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'No constraint data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Constraint Violation History')

    # 4. Individual design variables - normalized
    ax = axes[1, 0]
    if 'xvars' in values:
        xvars = values['xvars']
        lower = np.array([1.0, 0.35, 0.18, 0.40, 0.40])
        upper = np.array([2.5, 1.20, 0.60, 0.90, 0.95])
        xvars_norm = (xvars - lower) / (upper - lower)
        for i, name in enumerate(var_names):
            ax.plot(xvars_norm[:, i], label=name, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Normalized Value [0, 1]')
        ax.set_title('Normalized Design Variables')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

    # 5. Objective improvement
    ax = axes[1, 1]
    obj_improvement = float(obj_vals[0]) - obj_vals
    obj_improvement = np.asarray(obj_improvement).flatten()
    ax.plot(obj_improvement, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Improvement')
    ax.set_title('Cumulative Improvement')
    ax.grid(True, alpha=0.3)
    if len(obj_improvement) >= 2:
        ax.fill_between(range(len(obj_improvement)), 0, obj_improvement, alpha=0.3, color='g')
    else:
        ax.bar(0, obj_improvement[0] if len(obj_improvement) > 0 else 0, alpha=0.3, color='g')

    # 6. Step size
    ax = axes[1, 2]
    if 'xvars' in values:
        xvars = values['xvars']
        step_sizes = np.sqrt(np.sum(np.diff(xvars, axis=0)**2, axis=1))
        ax.semilogy(step_sizes, 'm-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Step Size')
        ax.set_title('Step Size History')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'convergence_history.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_design_space_2d(x_opt: np.ndarray):

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('T-Tail Design Space Analysis', fontsize=14, fontweight='bold')

    n_grid = 50

    # Slice 1: l_boom vs S_h
    ax = axes[0, 0]
    l_boom_grid = np.linspace(1.0, 2.0, n_grid)
    S_h_grid = np.linspace(0.35, 1.1, n_grid)
    L, SH = np.meshgrid(l_boom_grid, S_h_grid)

    obj_vals = np.zeros_like(L)
    feasible = np.zeros_like(L, dtype=bool)

    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([L[i, j], SH[i, j], x_opt[2], x_opt[3], x_opt[4]])
            xdict = {'xvars': x}
            funcs, _ = _userfunc_for_viz(xdict)
            obj_vals[i, j] = funcs['obj']
            cons = {k: v for k, v in funcs.items() if k != 'obj'}
            feasible[i, j] = all(v >= -1e-6 for v in cons.values())

    contour = ax.contourf(L, SH, obj_vals, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Objective')
    ax.contourf(L, SH, ~feasible, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
    ax.contour(L, SH, feasible.astype(float), levels=[0.5], colors=['red'], linewidths=2)
    ax.plot(x_opt[0], x_opt[1], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
    ax.set_xlabel('Boom Length [m]')
    ax.set_ylabel('H-tail Area [m$^2$]')
    ax.set_title('l_boom vs S_h (S_v, tpr fixed)')
    ax.legend()

    # Slice 2: l_boom vs S_v
    ax = axes[0, 1]
    S_v_grid = np.linspace(0.18, 0.55, n_grid)
    L2, SV = np.meshgrid(l_boom_grid, S_v_grid)

    obj_vals2 = np.zeros_like(L2)
    feasible2 = np.zeros_like(L2, dtype=bool)

    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([L2[i, j], x_opt[1], SV[i, j], x_opt[3], x_opt[4]])
            xdict = {'xvars': x}
            funcs, _ = _userfunc_for_viz(xdict)
            obj_vals2[i, j] = funcs['obj']
            cons = {k: v for k, v in funcs.items() if k != 'obj'}
            feasible2[i, j] = all(v >= -1e-6 for v in cons.values())

    contour2 = ax.contourf(L2, SV, obj_vals2, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(contour2, ax=ax, label='Objective')
    ax.contourf(L2, SV, ~feasible2, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
    ax.contour(L2, SV, feasible2.astype(float), levels=[0.5], colors=['red'], linewidths=2)
    ax.plot(x_opt[0], x_opt[2], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
    ax.set_xlabel('Boom Length [m]')
    ax.set_ylabel('V-tail Area [m$^2$]')
    ax.set_title('l_boom vs S_v (S_h, tpr fixed)')
    ax.legend()

    # Slice 3: S_h vs S_v
    ax = axes[1, 0]
    SH2, SV2 = np.meshgrid(S_h_grid, S_v_grid)

    obj_vals3 = np.zeros_like(SH2)
    feasible3 = np.zeros_like(SH2, dtype=bool)

    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([x_opt[0], SH2[i, j], SV2[i, j], x_opt[3], x_opt[4]])
            xdict = {'xvars': x}
            funcs, _ = _userfunc_for_viz(xdict)
            obj_vals3[i, j] = funcs['obj']
            cons = {k: v for k, v in funcs.items() if k != 'obj'}
            feasible3[i, j] = all(v >= -1e-6 for v in cons.values())

    contour3 = ax.contourf(SH2, SV2, obj_vals3, levels=30, cmap='viridis', alpha=0.8)
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
            funcs, _ = _userfunc_for_viz(xdict)
            obj_vals4[i, j] = funcs['obj']
            cons = {k: v for k, v in funcs.items() if k != 'obj'}
            feasible4[i, j] = all(v >= -1e-6 for v in cons.values())

    contour4 = ax.contourf(TPH, TPV, obj_vals4, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(contour4, ax=ax, label='Objective')
    ax.contourf(TPH, TPV, ~feasible4, levels=[0.5, 1.5], colors=['red'], alpha=0.3)
    ax.contour(TPH, TPV, feasible4.astype(float), levels=[0.5], colors=['red'], linewidths=2)
    ax.plot(x_opt[3], x_opt[4], 'w*', markersize=15, markeredgecolor='k', label='Optimal')
    ax.set_xlabel('H-tail Taper Ratio')
    ax.set_ylabel('V-tail Taper Ratio')
    ax.set_title('tpr_h vs tpr_v (l_boom, S_h, S_v fixed)')
    ax.legend()

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'design_space_2d.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_design_space_3d(x_opt: np.ndarray):

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
            funcs, _ = _userfunc_for_viz(xdict)
            obj_vals[i, j] = funcs['obj']
            cons = {k: v for k, v in funcs.items() if k != 'obj'}
            feasible[i, j] = all(v >= -1e-6 for v in cons.values())

    surf = ax.plot_surface(L, SH, obj_vals, facecolors=plt.cm.viridis((obj_vals - obj_vals.min()) / (obj_vals.max() - obj_vals.min())),
                          alpha=0.8, linewidth=0.5, edgecolor='gray')

    infeasible_obj = np.where(~feasible, obj_vals, np.nan)
    ax.plot_wireframe(L, SH, infeasible_obj, color='red', alpha=0.3, linewidth=0.5)

    xdict_opt = {'xvars': x_opt}
    funcs_opt, _ = _userfunc_for_viz(xdict_opt)
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


def plot_design_space_4d():

    csv_file = os.path.join(DATA_DIR, 'design_space_samples.csv')

    if not os.path.exists(csv_file):
        print(f"  Generating design space samples (CSV not found)...")
        generate_design_space_samples()

    try:
        data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'l_boom': float(row['l_boom']),
                    'S_h': float(row['S_h']),
                    'S_v': float(row['S_v']),
                    'tpr_h': float(row['tpr_h']),
                    'tpr_v': float(row['tpr_v']),
                    'objective': float(row['objective']),
                    'feasible': row['feasible'].lower() == 'true'
                })
    except Exception as e:
        print(f"  Warning: Could not read CSV file: {e}")
        return

    l_boom = np.array([d['l_boom'] for d in data])
    S_h = np.array([d['S_h'] for d in data])
    S_v = np.array([d['S_v'] for d in data])
    obj = np.array([d['objective'] for d in data])
    feasible = np.array([d['feasible'] for d in data])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    feas_mask = feasible
    infeas_mask = ~feasible

    scatter_feas = ax.scatter(S_h[feas_mask], S_v[feas_mask], l_boom[feas_mask],
                             c=obj[feas_mask], cmap='viridis', s=50, alpha=0.8,
                             label=f'Feasible ({feas_mask.sum()} pts)')

    ax.scatter(S_h[infeas_mask], S_v[infeas_mask], l_boom[infeas_mask],
               c='lightgray', s=10, alpha=0.3, label=f'Infeasible ({infeas_mask.sum()} pts)')

    x_opt = np.array([1.498, 1.200, 0.600, 0.521, 0.900])
    ax.scatter([x_opt[1]], [x_opt[2]], [x_opt[0]], color='red', s=300, marker='*',
               edgecolor='black', linewidth=2, label='Expected Optimal', zorder=10)

    ax.set_xlabel('H-tail Area [m$^2$]')
    ax.set_ylabel('V-tail Area [m$^2$]')
    ax.set_zlabel('Boom Length [m]')
    ax.set_title('T-Tail v5.0 Design Space (4D Visualization)\nColor = Objective Value')
    ax.legend(loc='upper left')

    plt.colorbar(scatter_feas, ax=ax, shrink=0.5, label='Objective')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'design_space_4d.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def generate_design_space_samples():

    csv_file = os.path.join(DATA_DIR, 'design_space_samples.csv')

    n_per_dim = 8
    l_boom_vals = np.linspace(1.0, 2.5, n_per_dim)
    S_h_vals = np.linspace(0.35, 1.2, n_per_dim)
    S_v_vals = np.linspace(0.18, 0.60, n_per_dim)
    tpr_h_vals = np.linspace(0.40, 0.90, 4)
    tpr_v_vals = np.linspace(0.40, 0.95, 4)

    samples = []

    for l in l_boom_vals:
        for sh in S_h_vals:
            for sv in S_v_vals:
                for th in tpr_h_vals:
                    for tv in tpr_v_vals:
                        x = np.array([l, sh, sv, th, tv])
                        xdict = {'xvars': x}
                        funcs, _ = _userfunc_for_viz(xdict)
                        obj = funcs['obj']
                        cons = {k: v for k, v in funcs.items() if k != 'obj'}
                        feasible = all(v >= -1e-6 for v in cons.values())

                        samples.append({
                            'l_boom': l,
                            'S_h': sh,
                            'S_v': sv,
                            'tpr_h': th,
                            'tpr_v': tv,
                            'objective': obj,
                            'feasible': feasible
                        })

    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['l_boom', 'S_h', 'S_v', 'tpr_h', 'tpr_v', 'objective', 'feasible']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"  Generated {len(samples)} samples -> {csv_file}")


def plot_parallel_coordinates(x_opt: np.ndarray, history_file: str = None):

    fig, ax = plt.subplots(figsize=(14, 8))

    var_names = ['l_boom', 'S_h', 'S_v', 'tpr_h', 'tpr_v', 'Obj']
    lower = np.array([1.0, 0.35, 0.18, 0.40, 0.40, 0.0])
    upper = np.array([2.5, 1.20, 0.60, 0.90, 0.95, 2.0])

    history_data = None
    if history_file and os.path.exists(history_file):
        try:
            from pyoptsparse import History
            hist = History(history_file)
            values = hist.getValues()
            if 'xvars' in values and 'obj' in values:
                history_data = np.hstack([values['xvars'], values['obj'].reshape(-1, 1)])
        except Exception:
            pass

    if history_data is None:
        n_samples = 50
        samples = np.random.rand(n_samples, 5)
        samples = lower[:5] + samples * (upper[:5] - lower[:5])
        obj_vals = []
        for s in samples:
            xdict = {'xvars': s}
            funcs, _ = _objfunc_for_viz(xdict)
            obj_vals.append(funcs['obj'])
        history_data = np.hstack([samples, np.array(obj_vals).reshape(-1, 1)])

    data_norm = (history_data - lower) / (upper - lower)
    x_coords = np.arange(len(var_names))

    obj_vals = history_data[:, -1]
    norm = plt.Normalize(obj_vals.min(), obj_vals.max())
    cmap = plt.cm.viridis_r

    for i, row in enumerate(data_norm):
        color = cmap(norm(obj_vals[i]))
        alpha = 0.3 if i < len(data_norm) - 1 else 1.0
        linewidth = 1 if i < len(data_norm) - 1 else 3
        ax.plot(x_coords, row, color=color, alpha=alpha, linewidth=linewidth)

    xdict_opt = {'xvars': x_opt}
    funcs_opt, _ = _objfunc_for_viz(xdict_opt)
    opt_data = np.append(x_opt, funcs_opt['obj'])
    opt_norm = (opt_data - lower) / (upper - lower)
    ax.plot(x_coords, opt_norm, 'r-', linewidth=3, marker='o', markersize=10, label='Optimal')

    ax.set_xticks(x_coords)
    ax.set_xticklabels(var_names)
    ax.set_ylim([-0.05, 1.05])

    for i, (name, lo, up) in enumerate(zip(var_names, lower, upper)):
        ax.annotate(f'{lo:.2f}', xy=(i, 0), xytext=(i, -0.08), ha='center', fontsize=8)
        ax.annotate(f'{up:.2f}', xy=(i, 1), xytext=(i, 1.06), ha='center', fontsize=8)

    ax.set_title('T-Tail v5.0 Parallel Coordinates\n(Line color = objective value)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Objective')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'parallel_coordinates.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_constraint_margins(x_opt: np.ndarray):

    xdict = {'xvars': x_opt}
    funcs, _ = confunc(xdict)

    sorted_cons = sorted(funcs.items(), key=lambda x: x[1])
    names = [c[0] for c in sorted_cons]
    values = [c[1] for c in sorted_cons]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['green' if v >= 0 else 'red' for v in values]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = bar.get_width()
        if x_pos >= 0:
            ax.annotate(f'{val:+.4f}', xy=(x_pos + 0.01, bar.get_y() + bar.get_height()/2),
                       va='center', ha='left', fontsize=8)
        else:
            ax.annotate(f'{val:+.4f}', xy=(x_pos - 0.01, bar.get_y() + bar.get_height()/2),
                       va='center', ha='right', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Constraint Value (g >= 0 is feasible)')
    ax.set_title('T-Tail v5.0 Constraint Margins at Optimal Point')

    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Satisfied'),
                       Patch(facecolor='red', alpha=0.7, label='Violated')]
    ax.legend(handles=legend_elements, loc='lower right')

    n_violated = sum(1 for v in values if v < 0)
    n_satisfied = len(values) - n_violated
    ax.annotate(f'Satisfied: {n_satisfied}/{len(values)}\nViolated: {n_violated}/{len(values)}',
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'constraint_margins.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_tail_geometry(x_opt: np.ndarray):

    derived = compute_derived_quantities(x_opt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # H-tail planform
    ax = axes[0]
    b_h = derived['b_h']
    c_root_h = derived['c_root_h']
    c_tip_h = derived['c_tip_h']

    h_tail_pts = np.array([
        [-b_h/2, 0],
        [-b_h/2, c_tip_h],
        [0, c_root_h],
        [b_h/2, c_tip_h],
        [b_h/2, 0],
        [0, 0],
        [-b_h/2, 0]
    ])

    h_tail = plt.Polygon(h_tail_pts, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(h_tail)

    elev_chord = TAIL.elevator_ratio * c_root_h
    elev_pts = np.array([
        [-b_h/2, c_tip_h - TAIL.elevator_ratio * c_tip_h],
        [-b_h/2, c_tip_h],
        [0, c_root_h],
        [b_h/2, c_tip_h],
        [b_h/2, c_tip_h - TAIL.elevator_ratio * c_tip_h],
        [0, c_root_h - TAIL.elevator_ratio * c_root_h],
        [-b_h/2, c_tip_h - TAIL.elevator_ratio * c_tip_h]
    ])
    elev = plt.Polygon(elev_pts, fill=True, facecolor='yellow', edgecolor='orange', linewidth=1, alpha=0.7)
    ax.add_patch(elev)

    ax.annotate(f'Span = {b_h:.3f} m', xy=(0, -0.1), ha='center', fontsize=10)
    ax.annotate(f'Root = {c_root_h:.3f} m', xy=(b_h/2 + 0.05, c_root_h/2), ha='left', fontsize=10)
    ax.annotate(f'Tip = {c_tip_h:.3f} m', xy=(-b_h/2 - 0.05, c_tip_h/2), ha='right', fontsize=10)
    ax.annotate(f'Area = {x_opt[1]:.3f} m$^2$', xy=(0, c_root_h + 0.1), ha='center', fontsize=10)
    ax.annotate('Elevator', xy=(0, c_root_h - elev_chord/2), ha='center', fontsize=9, color='orange')

    ax.set_xlim([-b_h/2 - 0.3, b_h/2 + 0.3])
    ax.set_ylim([-0.2, c_root_h + 0.2])
    ax.set_aspect('equal')
    ax.set_xlabel('Spanwise [m]')
    ax.set_ylabel('Chordwise [m]')
    ax.set_title(f'Horizontal Tail Planform\nAR = {TAIL.AR_h:.1f}, Taper = {x_opt[3]:.2f}')
    ax.grid(True, alpha=0.3)

    # V-tail planform
    ax = axes[1]
    b_v = derived['b_v']
    c_root_v = derived['c_root_v']
    c_tip_v = derived['c_tip_v']
    sweep_rad = np.deg2rad(TAIL.swp_v)

    le_tip_x = b_v * np.tan(sweep_rad)
    v_tail_pts = np.array([
        [0, 0],
        [c_root_v, 0],
        [le_tip_x + c_tip_v, b_v],
        [le_tip_x, b_v],
        [0, 0]
    ])

    v_tail = plt.Polygon(v_tail_pts, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(v_tail)

    rud_pts = np.array([
        [c_root_v * (1 - TAIL.rudder_ratio), 0],
        [c_root_v, 0],
        [le_tip_x + c_tip_v, b_v],
        [le_tip_x + c_tip_v * (1 - TAIL.rudder_ratio), b_v],
        [c_root_v * (1 - TAIL.rudder_ratio), 0]
    ])
    rud = plt.Polygon(rud_pts, fill=True, facecolor='yellow', edgecolor='orange', linewidth=1, alpha=0.7)
    ax.add_patch(rud)

    ax.annotate(f'Height = {b_v:.3f} m', xy=(-0.1, b_v/2), ha='right', va='center', fontsize=10, rotation=90)
    ax.annotate(f'Root = {c_root_v:.3f} m', xy=(c_root_v/2, -0.05), ha='center', fontsize=10)
    ax.annotate(f'Tip = {c_tip_v:.3f} m', xy=(le_tip_x + c_tip_v/2, b_v + 0.05), ha='center', fontsize=10)
    ax.annotate(f'Area = {x_opt[2]:.3f} m$^2$', xy=(c_root_v + 0.1, b_v/2), ha='left', fontsize=10)
    ax.annotate(f'Sweep = {TAIL.swp_v:.0f} deg', xy=(le_tip_x/2, b_v*0.7), ha='center', fontsize=9, rotation=TAIL.swp_v)
    ax.annotate('Rudder', xy=(c_root_v, b_v/3), ha='left', fontsize=9, color='orange')

    ax.set_xlim([-0.2, max(c_root_v, le_tip_x + c_tip_v) + 0.2])
    ax.set_ylim([-0.15, b_v + 0.15])
    ax.set_aspect('equal')
    ax.set_xlabel('Chordwise [m]')
    ax.set_ylabel('Spanwise/Height [m]')
    ax.set_title(f'Vertical Tail Planform (Side View)\nAR = {TAIL.AR_v:.1f}, Taper = {x_opt[4]:.2f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'tail_geometry.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_weight_breakdown(x_opt: np.ndarray):

    derived = compute_derived_quantities(x_opt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    weights = [derived['W_h']/9.81, derived['W_v']/9.81, derived['W_boom']/9.81]
    labels = ['H-tail', 'V-tail', 'Boom']
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(weights, labels=labels, autopct='%1.1f%%',
                                       colors=colors, explode=explode, startangle=90,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    ax.set_title(f'Tail Weight Breakdown\nTotal: {derived["TOGW"]/9.81:.3f} kg')

    for i, (weight, label) in enumerate(zip(weights, labels)):
        angle = (wedges[i].theta1 + wedges[i].theta2) / 2
        x = 1.3 * np.cos(np.deg2rad(angle))
        y = 1.3 * np.sin(np.deg2rad(angle))
        ax.annotate(f'{weight:.3f} kg', xy=(x, y), ha='center', va='center', fontsize=10)

    ax = axes[1]
    categories = ['H-tail', 'V-tail', 'Boom', 'Total Tail', 'Aircraft']
    values = [derived['W_h']/9.81, derived['W_v']/9.81, derived['W_boom']/9.81,
              derived['TOGW']/9.81, AC.TOGW]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'gray']

    bars = ax.bar(categories, values, color=colors, edgecolor='black')

    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f} kg', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

    ax.set_ylabel('Mass [kg]')
    ax.set_title(f'Weight Comparison\nTail = {derived["TOGW"]/(AC.TOGW*9.81)*100:.1f}% of TOGW')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'weight_breakdown.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_drag_breakdown(x_opt: np.ndarray):

    derived = compute_derived_quantities(x_opt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    drags = [derived['D_parasite'], derived['D_induced']]
    labels = ['Parasitic', 'Induced (Trim)']
    colors = ['lightblue', 'lightcoral']

    wedges, texts, autotexts = ax.pie(drags, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    ax.set_title(f'Tail Drag Breakdown\nTotal: {derived["D_total"]:.3f} N')

    ax = axes[1]
    components = ['H-tail', 'V-tail', 'Boom']
    S_wet = [derived['S_wet_h'], derived['S_wet_v'], derived['S_wet_boom']]

    bars = ax.bar(components, S_wet, color=['lightblue', 'lightgreen', 'lightyellow'],
                  edgecolor='black')

    for bar, val in zip(bars, S_wet):
        ax.annotate(f'{val:.3f} m$^2$', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

    ax.set_ylabel('Wetted Area [m$^2$]')
    ax.set_title(f'Wetted Area Breakdown\nTotal: {derived["S_wet_total"]:.3f} m$^2$')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'drag_breakdown.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_sensitivity_analysis(x_opt: np.ndarray):

    var_names = ['l_boom', 'S_h', 'S_v', 'tpr_h', 'tpr_v']

    h = 1e-5
    xdict = {'xvars': x_opt}
    f0, _ = _objfunc_for_viz(xdict)
    f0_val = f0['obj']

    sensitivities = []
    for i in range(5):
        x_pert = x_opt.copy()
        x_pert[i] += h
        f_pert, _ = _objfunc_for_viz({'xvars': x_pert})
        sens = (f_pert['obj'] - f0_val) / h
        sensitivities.append(sens)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue' if s < 0 else 'red' for s in sensitivities]
    bars = ax.bar(var_names, sensitivities, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, sensitivities):
        y_pos = bar.get_height()
        ax.annotate(f'{val:+.4f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                   xytext=(0, 3 if y_pos >= 0 else -10), textcoords='offset points',
                   ha='center', fontsize=10)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Sensitivity (dJ/dx)')
    ax.set_title('T-Tail v5.0 Objective Sensitivity Analysis\n(Positive = increasing DV increases objective)')
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Increases Obj'),
                       Patch(facecolor='blue', alpha=0.7, label='Decreases Obj')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'sensitivity_analysis.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_trade_studies(x_opt: np.ndarray):

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Trade 1: V_H vs Objective
    ax = axes[0, 0]
    l_boom_range = np.linspace(1.0, 2.5, 30)
    V_H_vals = []
    obj_vals = []

    for l in l_boom_range:
        x = np.array([l, x_opt[1], x_opt[2], x_opt[3], x_opt[4]])
        derived = compute_derived_quantities(x)
        xdict = {'xvars': x}
        funcs, _ = _objfunc_for_viz(xdict)
        V_H_vals.append(derived['V_H'])
        obj_vals.append(funcs['obj'])

    ax.plot(V_H_vals, obj_vals, 'b-', linewidth=2)

    derived_opt = compute_derived_quantities(x_opt)
    xdict_opt = {'xvars': x_opt}
    funcs_opt, _ = _objfunc_for_viz(xdict_opt)
    ax.plot(derived_opt['V_H'], funcs_opt['obj'], 'r*', markersize=15, label='Optimal')

    ax.set_xlabel('Horizontal Tail Volume (V_H)')
    ax.set_ylabel('Objective')
    ax.set_title('V_H vs Objective Trade')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trade 2: Static Margin vs Weight
    ax = axes[0, 1]
    SM_range = np.linspace(0.05, 0.20, 20)
    weights = []

    for sm in SM_range:
        scale = 1 + 2 * (sm - 0.15)
        W_approx = derived_opt['TOGW'] * scale
        weights.append(W_approx / 9.81)

    ax.plot(SM_range * 100, weights, 'g-', linewidth=2)
    ax.axvline(x=15, color='r', linestyle='--', label=f'Design SM = 15%')
    ax.plot(15, derived_opt['TOGW']/9.81, 'r*', markersize=15)

    ax.set_xlabel('Static Margin [%]')
    ax.set_ylabel('Tail Weight [kg]')
    ax.set_title('Static Margin vs Weight Trade')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trade 3: l_boom vs Structural Weight
    ax = axes[1, 0]
    boom_weights = []

    for l in l_boom_range:
        x = np.array([l, x_opt[1], x_opt[2], x_opt[3], x_opt[4]])
        derived = compute_derived_quantities(x)
        boom_weights.append(derived['W_boom'] / 9.81)

    ax.plot(l_boom_range, boom_weights, 'm-', linewidth=2)
    ax.plot(x_opt[0], derived_opt['W_boom']/9.81, 'r*', markersize=15, label='Optimal')

    ax.set_xlabel('Boom Length [m]')
    ax.set_ylabel('Boom Weight [kg]')
    ax.set_title('Boom Length vs Structural Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trade 4: S_h vs Drag
    ax = axes[1, 1]
    S_h_range = np.linspace(0.35, 1.2, 30)
    drags = []

    for sh in S_h_range:
        x = np.array([x_opt[0], sh, x_opt[2], x_opt[3], x_opt[4]])
        derived = compute_derived_quantities(x)
        drags.append(derived['D_total'])

    ax.plot(S_h_range, drags, 'c-', linewidth=2)
    ax.plot(x_opt[1], derived_opt['D_total'], 'r*', markersize=15, label='Optimal')

    ax.set_xlabel('H-tail Area [m$^2$]')
    ax.set_ylabel('Total Drag [N]')
    ax.set_title('H-tail Area vs Drag')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'trade_studies.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_monte_carlo_robustness(x_opt: np.ndarray):

    n_samples = 1000
    perturbation = 0.05

    lower = np.array([1.0, 0.35, 0.18, 0.40, 0.40])
    upper = np.array([2.5, 1.20, 0.60, 0.90, 0.95])

    violations = []
    objectives = []
    feasible_count = 0

    for _ in range(n_samples):
        noise = np.random.uniform(-perturbation, perturbation, 5)
        x_pert = x_opt * (1 + noise)
        x_pert = np.clip(x_pert, lower, upper)

        xdict = {'xvars': x_pert}
        funcs, _ = _userfunc_for_viz(xdict)

        obj = funcs.pop('obj')
        objectives.append(obj)

        n_viol = sum(1 for v in funcs.values() if v < 0)
        violations.append(n_viol)

        if n_viol == 0:
            feasible_count += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    unique, counts = np.unique(violations, return_counts=True)
    ax.bar(unique, counts / n_samples * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Number of Violated Constraints')
    ax.set_ylabel('Frequency [%]')
    ax.set_title(f'Constraint Violations Distribution\n(n={n_samples}, perturbation={perturbation*100:.0f}%)')
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate(f'Feasibility Rate: {feasible_count/n_samples*100:.1f}%',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1]
    ax.hist(objectives, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(objectives), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(objectives):.3f}')

    xdict_nom = {'xvars': x_opt}
    funcs_nom, _ = _objfunc_for_viz(xdict_nom)
    ax.axvline(x=funcs_nom['obj'], color='b', linestyle='-', linewidth=2, label=f'Nominal: {funcs_nom["obj"]:.3f}')

    ax.set_xlabel('Objective Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Objective Distribution under Perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    obj_by_viol = {}
    for obj, viol in zip(objectives, violations):
        if viol not in obj_by_viol:
            obj_by_viol[viol] = []
        obj_by_viol[viol].append(obj)

    positions = sorted(obj_by_viol.keys())
    data = [obj_by_viol[p] for p in positions]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Number of Violated Constraints')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective vs Constraint Violations')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'monte_carlo_robustness.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def plot_optimization_path_3d(history_file: str, x_opt: np.ndarray):

    try:
        from pyoptsparse import History
        hist = History(history_file)
        values = hist.getValues()
        if 'xvars' not in values:
            print("  Warning: No xvars in history, skipping 3D path plot")
            return
        xvars = values['xvars']
    except Exception as e:
        print(f"  Warning: Could not load history: {e}")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xvars[:, 0], xvars[:, 1], xvars[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax.scatter(xvars[:, 0], xvars[:, 1], xvars[:, 2], c=np.arange(len(xvars)),
               cmap='viridis', s=30, alpha=0.8)

    ax.scatter([xvars[0, 0]], [xvars[0, 1]], [xvars[0, 2]], color='green', s=200,
               marker='o', label='Start', edgecolor='black', linewidth=2)
    ax.scatter([xvars[-1, 0]], [xvars[-1, 1]], [xvars[-1, 2]], color='red', s=200,
               marker='*', label='End', edgecolor='black', linewidth=2)

    ax.set_xlabel('Boom Length [m]')
    ax.set_ylabel('H-tail Area [m$^2$]')
    ax.set_zlabel('V-tail Area [m$^2$]')
    ax.set_title('T-Tail v5.0 Optimization Path in 3D Design Space')
    ax.legend()

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'optimization_path_3d.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")
