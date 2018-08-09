import simulationData as sim
from simulationData import Tools
# import matplotlib.pyplot as plt
import numpy as np
# import os



data = sim.SimulationData()
data.loadGridData()


rho = Tools.averageFrames("./", "rho", range(1000, 3000))

# rho = np.load("avg_rho.npy")
# data.variables["rho"] = rho
# data.variables["prs"] = np.load("avg_prs.npy")
# data.variables["vx1"] = np.load("avg_vx1.npy")
# data.variables["vx2"] = np.load("avg_vx2.npy")
# v = np.sqrt(data.variables["vx1"]**2 + data.variables["vx2"]**2)

# temp = Tools.computeTemperature(data)
# pressureScaleHeight = 4.8 * Tools.pressureScaleHeightFlat(data)

# cs = np.sqrt(data.kb * temp / (data.mu * data.mp)) / data.unitVelocity
# r = np.tile(data.x1, (len(data.x2), 1))
# vesc = np.sqrt(2.0 / r)
# ratio = v / vesc

# x_range = [0, 99, 1000]
# y_range = [0, 99, 1000]
# Tools.plotVariable(data, rho, show=False, clear=False, interpolate=True,
#                   x_range=x_range, y_range=y_range, figsize=(10, 7))
# Tools.plotLineData(data, pressureScaleHeight, x_range=[0.4, 99, 100], show=False, filename="pressureScaleHeight")
# Tools.plotVelocityField(data, show=True, filename="velfield_temp")#,
                             #x_range=x_range, y_range=y_range)
