import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
import os

def plotUsefulQuantities(data):
    for current_file in os.listdir(path):
        if current_file.endswith(".h5"):
            data.loadData(os.path.join(path, current_file))
            data.loadGridData()
            frame = current_file.split('.')[1]
            print("Plotting for frame " + frame)

            Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
            Tools.plotVariable(data, data.variables["rho"], "sonic_" + frame, log=True, clear=False)
            Tools.plotSonicBarrier(data, "sonic_" + frame)
            # Tools.plotVariable(data, data.variables["rho"], "field_" + frame, log=True)
            # Tools.plotVelocityField(data, "field_" + frame, dx1=7, dx2=5, scale=60, width=0.001, overlay=True, x1_start=10)
            # Tools.interpolateRadialGrid(data, np.linspace(0.4, 10.0, 500))
            # Tools.plotVariable(data, data.variables["rho"], "field_zoom_" + frame, log=True)
            # Tools.plotVelocityField(data, "field_zoom_" + frame, dx1=7, dx2=5, scale=50, width=0.002, overlay=True, x1_start=10)
            temp = Tools.computeTemperature(data)
            mach = Tools.computeMachNumbers(data)
            Tools.plotVariable(data, temp, "temperature_" + frame, log=True)
            Tools.plotVariable(data, mach, "mach_" + frame, log=False)
            Tools.plotIonizationParameter(data, "ionization_param_" + frame)


path = "./"
data = sim.SimulationData()
data.loadData("data.0000.dbl.h5")
data.loadGridData()
# Tools.plotVariable(data, data.variables["bx1"], show=True, log=False)
# Tools.plotVariable(data, data.variables["bx2"], show=True, log=False)
Tools.plotVariable(data, np.sqrt(data.variables["bx1"]**2 + data.variables["bx2"]**2), show=True, log=True)
Tools.plotDensity(data, show=False, clear=False)
Tools.plotMagneticFieldLines(data, show=True, norm=False)
# masses, times = Tools.computeTotalMasses("./")
# Tools.plotMassLosses('./')
# Tools.plotIonizationParameter(data, "ionization_param")
# Tools.removeFilesWithStride("./", 20)
# plotUsefulQuantities(data)