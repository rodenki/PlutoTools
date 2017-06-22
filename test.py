import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np



data = sim.SimulationData()
data.loadFrame("0560")
data.loadGridData()

Tools.interpolateRadialGrid(data, np.linspace(0.4, 49.5, 500))
Tools.plotVariable(data, data.variables["rho"], "test", log=True)
Tools.plotSonicBarrier(data, "test")
#Tools.plotVelocityField(data, "field", dx1=5, dx2=4, scale=60, width=0.001, overlay=True, x1_start=10)
plt.clf()
