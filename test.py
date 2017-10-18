import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plotUsefulQuantities(data):
    for current_file in os.listdir(path):
        if current_file.endswith(".h5"):
            data.loadData(os.path.join(path, current_file))
            data.loadGridData()
            frame = current_file.split('.')[1]
            print("Plotting for frame " + frame)

            # Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
            # Tools.plotVariable(data, data.variables["rho"], "sonic_" + frame, log=True, clear=False)
            # Tools.plotSonicBarrier(data, "sonic_" + frame)
            # Tools.plotVariable(data, data.variables["rho"], "field_" + frame, log=True)
            # Tools.plotVelocityField(data, "field_" + frame, dx1=7, dx2=5, scale=60, width=0.001, overlay=True, x1_start=10)
            # Tools.interpolateRadialGrid(data, np.linspace(0.4, 10.0, 500))
            # Tools.plotVariable(data, data.variables["rho"], "field_zoom_" + frame, log=True)
            temp = Tools.computeTemperature(data)
            # mach = Tools.computeMachNumbers(data)
            Tools.plotVariable(data, temp, "vel_field_" + frame, log=True, clear=False)
            Tools.plotVelocityField(data, "vel_field_" + frame, dx1=7, dx2=5, scale=50, width=0.002, x1_start=10, wind_only=True, norm=True)

            # Tools.plotVariable(data, mach, "mach_" + frame, log=False)
            # Tools.plotIonizationParameter(data, "ionization_param_" + frame)


path = "./"
data = sim.SimulationData()
parser = argparse.ArgumentParser()
parser.add_argument("frame")
args = parser.parse_args()
frame = args.frame
for i in range(4-len(frame)):
    frame = "0" + frame
frame = "data." + frame + ".dbl.h5"
data.loadData(frame)
data.loadGridData()

temp = Tools.computeTemperature(data)
x_range=[0.0, 99.0, 1000]
y_range=[0.0, 99.0, 1000]
rho = data.variables["rho"] * data.unitNumberDensity
bx = data.variables["bx1"]
by = data.variables["bx2"]
bz = data.variables["bx3"]
b_tot = np.sqrt(bx**2 + by**2 + bz**2)
Tools.plotVariable(data, rho, show=False, log=True, clear=False, interpolate=True, x_range=x_range, y_range=y_range)
Tools.plotMagneticFieldLines(data, show=True, norm=True, x_range=x_range, y_range=y_range)


# masses, times = Tools.computeTotalMasses("./")
# Tools.plotMassLosses('./')
# Tools.plotIonizationParameter(data, "ionization_param")
# Tools.removeFilesWithStride("./", 20)
# plotUsefulQuantities(data)
