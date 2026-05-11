# M-BRAACE

**Michaigan - Boeing Research in Aircraft Architecture for Cryofuel Efficiency** — a multidisciplinary design optimization (MDO) framework for a cryogenically-fueled aircraft concept, developed at the University of Michigan.

## Overview

This repository is a Python-based design and analysis toolkit covering aircraft sizing, empennage optimization, and Monte Carlo feasibility analysis for a cryofuel aircraft concept. The framework was used to iterate on aircraft configuration, evaluate design feasibility across required mission performance and stability parameters, and produce the final design point presented at Boeing design reviews.

## Repository Structure

```
M-BRAACE/
├── Aircraft_Sizeopt/   # Mission-level sizing & weight estimation
├── Aircraft_Tailopt/   # T-tail structural & aerodynamic optimization
└── montecarlo/         # Monte Carlo feasibility & sensitivity analysis
```

### `Aircraft_Sizeopt/`

Conceptual sizing loop — Drives the top-level design variables passed downstream to the tail optimizer.

### `Aircraft_Tailopt/`

T-tail multidisciplinary optimization using `pyoptsparse` with the SLSQP gradient-based solver. Five design variables, seventeen constraints spanning boom deflection, spar stress, von Mises yield, and torsional rigidity. Couples aerodynamic loading with structural sizing.

### `montecarlo/`

Monte Carlo sampling over uncertainty in aerodynamic and structural parameters to characterize feasibility margins and identify driving constraints in the design space.

## Context

Developed during 2025-26 as part of year 1 of the M-BRAACE MBSE design project at the University of Michigan. Results were presented at Boeing design reviews with engineering staff. Code is shared as a portfolio reference; not actively maintained.

## Author

Jared Tuatini — University of Michigan, Aerospace Engineering and Computer Science
