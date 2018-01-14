from simulationData import SimulationData
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np


# data = SimulationData()
# data.loadGridData()

# rho = np.load("avg_rho.npy")
# data.variables["rho"] = rho
# data.variables["prs"] = np.load("avg_prs.npy")
# data.variables["vx1"] = np.load("avg_vx1.npy")
# data.variables["vx2"] = np.load("avg_vx2.npy")
# radii, losses = Tools.computeRadialMassLosses(data)

# np.save("radii.npy", radii)
# np.save("losses.npy", losses)

radii = np.load("radii.npy")
losses = np.load("losses.npy")
cumulativeLosses = []

for i, loss in enumerate(losses):
	cumulativeLosses.append(np.sum(losses[:i]))

plt.plot(radii, cumulativeLosses)
plt.show()

# data.loadData("data.0500.dbl.h5")

# Tools.plotVariable(data, data.variables["rho"], show=True)
# Tools.computeStreamlines(data)
# Tools.plotVelocityField(data, show=True, filename="velfield_temp")
