import mpmath as mp

#aero coeffs

# coarse, medium, fine grids
CL_c, CL_m, CL_f = 0.281592, 0.307826, 0.312398
CD_c, CD_m, CD_f = 0.048475, 0.036949, 0.029382
CM_c, CM_m, CM_f = 0.032090, 0.033091, 0.033006

# storing
functionals = {
    "CL": (CL_c, CL_m, CL_f),
    "CD": (CD_c, CD_m, CD_f),
    "CM": (CM_c, CM_m, CM_f)
}

# R
def compute_R(Fc, Fm, Ff):
    return (Ff - Fm) / (Fm - Fc)

# solv for p using newtons method
def compute_p(R):
    # if R is non-positive p is not defined
    if R <= 0:
        return None

    # def eq for p
    f = lambda p: (2**p - 1)/(4**p - 2**p) - R

    # newton iteration
    try:
        p_solution = mp.findroot(f, 2)   # guess p=2
        return float(p_solution)
    except:
        return None

# calculation for each functional

results = {}

for name, (Fc, Fm, Ff) in functionals.items():
    R = compute_R(Fc, Fm, Ff)
    p = compute_p(R)
    results[name] = (R, p)

#resultsss

print("\n=== ORDER OF ACCURACY RESULTS ===\n")

for name, (R_value, p_value) in results.items():
    print(f"{name}:")
    print(f"  R = {R_value:.6f}")

    if p_value is None:
        print("  p = undefined (non-monotonic or no real solution)\n")
    else:
        print(f"  p = {p_value:.4f}\n")
