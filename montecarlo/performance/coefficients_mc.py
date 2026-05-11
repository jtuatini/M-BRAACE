import numpy as np
import sizeopt as so

coefficients = {
    "CD0": 0.0441,
    "e": 0.881,
    "CL_max": 1.7693,
    "eta_prop_mechanical": 0.5,
    "w_net": 129827,
    "phi_eq": 0.26
}

def get_permutation(coefficients: dict, sigma: float, seed = None):

    np.random.seed(seed)

    coefficients_permuted = coefficients.copy()

    for key in coefficients.keys():
        coefficients_permuted[key] = np.random.normal(coefficients[key], sigma * coefficients[key])
    
    return coefficients_permuted

def main():

    import matplotlib.pyplot as plt

    rng_req = 7.9
    end_req = 725
    takeoff_req = 26

    n_permutations = 100
    n_req = 0

    req_permuted = {
        "Rng": [],
        "End": [],
        "Takeoff": [],
    }
    for i in range(n_permutations):
        coefficients_permuted = get_permutation(coefficients, 0.05)
        try:
            AC_permuted = so.sizing_loop(coefficients_permuted)
            rng_permuted = so.breguet_range(AC_permuted)
            end_permuted = so.breguet_endurance(AC_permuted)
            takeoff_permuted = so.TO_field_length(AC_permuted)
        except Exception as e:
            print(f"Permutation {i} failed: {e}")
            continue

        if np.abs(rng_permuted - rng_req) / rng_req <= 0.10:
            if np.abs(end_permuted - end_req) / end_req <= 0.10:
                if np.abs(takeoff_permuted - takeoff_req) / takeoff_req <= 0.10:
                    n_req += 1

        req_permuted["Rng"].append(rng_permuted)
        req_permuted["End"].append(end_permuted)
        req_permuted["Takeoff"].append(takeoff_permuted)

        print(f"Permutation {i} passed: {rng_permuted} m, {end_permuted} s, {takeoff_permuted} m")
    
    prob_req = n_req / n_permutations

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    axs[0].hist(req_permuted["Rng"], bins=40, color='#1f77b4', edgecolor='black', alpha=0.8)
    axs[1].hist(req_permuted["End"], bins=40, color='#1f77b4', edgecolor='black', alpha=0.8)
    axs[2].hist(req_permuted["Takeoff"], bins=40, color='#1f77b4', edgecolor='black', alpha=0.8)

    plt.savefig("requirements_distribution.png")
    plt.close()
    
if __name__ == "__main__":
    main()