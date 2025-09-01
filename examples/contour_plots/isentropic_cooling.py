import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import coolpropx as cpx

cpx.set_plot_options(grid=False)


# ----------------------------------------------------------------------------------------------- #
# Plot sensitivity of temperature with respect to pressure at constant entropy for water
# ----------------------------------------------------------------------------------------------- #
# Define fluid and property ranges
fluid = cpx.Fluid(name="water")
x_prop = "s"
y_prop = "p"
z_prop = "Z"
x_range = np.linspace(500, 9000, 50)
y_range = np.logspace(np.log10(1e3), np.log10(1e8), 50)
states1 = fluid.get_states(cpx.PSmass_INPUTS, y_range, x_range)
states2 = fluid.get_states(cpx.PSmass_INPUTS, y_range+1e-3*y_range, x_range)
prop = (states2["T"] - states1["T"]) / (1e-3*y_range)

# Create figure
fig_1, ax_1 = plt.subplots(figsize=(6, 5))
ax_1.grid(False)
ax_1.set_yscale("log")
ax_1.set_xlabel(cpx.LABEL_MAPPING.get(x_prop, x_prop))
ax_1.set_ylabel(cpx.LABEL_MAPPING.get(y_prop, y_prop))
ax_1.set_xlim([x_range.min(), x_range.max()])
ax_1.set_ylim([y_range.min(), y_range.max()])

# Plot phase diagram and contour
fluid.plot_phase_diagram(x_prop, y_prop, axes=ax_1)
vmin = np.nanmin(prop)
vmax = np.nanmax(prop)
levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
colors = plt.get_cmap("Greys")(np.linspace(0.8, 0.2, len(levels) - 1))
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
contour = ax_1.contourf(states1[x_prop], states1[y_prop], prop, levels=levels, cmap=cmap, norm=norm)
cbar = fig_1.colorbar(contour, ax=ax_1, pad=0.02)
cbar.set_label(r"$(\partial T / \partial p)_s$")
cbar.set_ticks(levels[::2])
cbar.set_ticklabels([f"{v:.2e}" for v in levels[::2]])
plt.tight_layout(pad=1)


# ----------------------------------------------------------------------------------------------- #
# Plot sensitivity of temperature with respect to pressure at constant entropy for nitrogen
# ----------------------------------------------------------------------------------------------- #
# Define fluid and property ranges
fluid = cpx.Fluid(name="nitrogen")
x_prop = "s"
y_prop = "p"
z_prop = "Z"
x_range = np.linspace(2500, 5750, 100)
y_range = np.logspace(np.log10(2e4), np.log10(2e7), 100)
states1 = fluid.get_states(cpx.PSmass_INPUTS, y_range, x_range)
states2 = fluid.get_states(cpx.PSmass_INPUTS, y_range+1e-3*y_range, x_range)
prop = (states2["T"] - states1["T"]) / (1e-3*y_range)

# Create figure
fig_1, ax_1 = plt.subplots(figsize=(6, 5))
ax_1.grid(False)
ax_1.set_yscale("log")
ax_1.set_xlabel(cpx.LABEL_MAPPING.get(x_prop, x_prop))
ax_1.set_ylabel(cpx.LABEL_MAPPING.get(y_prop, y_prop))
ax_1.set_xlim([x_range.min(), x_range.max()])
ax_1.set_ylim([y_range.min(), y_range.max()])

# Plot phase diagram and contour
fluid.plot_phase_diagram(x_prop, y_prop, axes=ax_1)
vmin = np.nanmin(prop)
vmax = np.nanmax(prop)
levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
colors = plt.get_cmap("Greys")(np.linspace(0.8, 0.2, len(levels) - 1))
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
contour = ax_1.contourf(states1[x_prop], states1[y_prop], prop, levels=levels, cmap=cmap, norm=norm)
cbar = fig_1.colorbar(contour, ax=ax_1, pad=0.02)
cbar.set_label(r"$(\partial T / \partial p)_s$")
cbar.set_ticks(levels[::2])
cbar.set_ticklabels([f"{v:.2e}" for v in levels[::2]])
plt.tight_layout(pad=1)


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()

    