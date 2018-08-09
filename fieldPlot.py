import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
import os

data = sim.SimulationData()
data.loadData("data.0034.dbl.h5")

x_range = [0, 4, 1000]
y_range = [0, 4, 1000]

rho = data.variables["rho"] * data.unitNumberDensity
# v = Tools.computeAbsoluteVelocities(data)
Tools.plotVariable(data, rho, show=False, clear=False, filename="magfield_34z", interpolate=True,
                   x_range=x_range, y_range=y_range, figsize=(10, 7))
Tools.plotMagneticFieldLines(data, filename="magfield_34z", x_range=x_range, y_range=y_range)
