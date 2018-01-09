from simulationData import SimulationData
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np


data = SimulationData()
data.loadData("data.0500.dbl.h5")

# Tools.plotVariable(data, data.variables["rho"], show=True)
Tools.computeStreamlines(data)
# Tools.plotVelocityField(data, show=True, filename="velfield_temp")
