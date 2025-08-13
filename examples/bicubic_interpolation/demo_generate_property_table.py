import os
import pickle
import coolpropx as cpx


outdir = "fluid_tables"


# ---------------------------
# Test parameters
# ---------------------------
fluid = "CO2"
hmin = 200e3     # J/kg
hmax = 600e3     # J/kg
Pmin = 2e6       # Pa
Pmax = 20e6      # Pa
Np = 20          # Grid size for quick test
Nh = 15

# ---------------------------
# Run table generation
# ---------------------------
cpx.bicubic.generate_property_table(hmin, hmax, Pmin, Pmax, fluid, Nh=Np, Np=Np, outdir=outdir)


# ---------------------------
# Load and preview output
# ---------------------------

pkl_file = os.path.join(outdir, f"{fluid}_{Np}_x_{Np}.pkl")

print("\n Verifying saved table structure...\n")

# Load pickled dict
with open(pkl_file, 'rb') as f:
    table = pickle.load(f)

# Check and print summary
print(" Keys in table:", list(table.keys()))

print("\n h values:", table['h'][:5])
print(" P values:", table['P'][:5])

print("\n Preview for property 'T':")
if 'T' in table:
    print(" - Value shape:", table['T']['value'].shape)
    print(" - Sample values (top-left corner):")
    print(table['T']['value'][:3, :3])
else:
    print(" - Property 'T' not found.")
