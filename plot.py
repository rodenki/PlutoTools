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

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame")
    parser.add_argument("x_min")
    parser.add_argument("x_max")
    parser.add_argument("x_res")
    parser.add_argument("y_min")
    parser.add_argument("y_max")
    parser.add_argument("y_res")
    args = parser.parse_args()
    frame = args.frame
    for i in range(4-len(frame)):
        frame = "0" + frame
    frame = "data." + frame + ".dbl.h5"
    return frame, [float(args.x_min), float(args.x_max), float(args.x_res)], [float(args.y_min), float(args.y_max), float(args.y_res)]



path = "./"
data = sim.SimulationData()
frame, x_range, y_range = getArgs()
data.loadData(frame)
data.loadGridData()

# velocities = np.sqrt(data.variables["vx1"]**2 + data.variables["vx2"]**2) * data.unitVelocity
# escapeVel = np.sqrt(2.0*data.G * data.solarMass / (data.x1 * data.unitLength))
# escapeVel = np.tile(escapeVel, (velocities.shape[0], 1))
#
# escapeVel = velocities / escapeVel
# Tools.plotVariable(data, escapeVel, show=False, clear=False, log=False, interpolate=True, x_range=x_range,
#                    y_range=y_range, vlimits=(1, 1.5))
# Tools.plotVelocityFieldLines(data, show=True, filename="vel_fieldlines_50_temp", norm=True, x_range=x_range, y_range=y_range)


# Tools.plotCumulativeMassloss(frame)

#temp = Tools.computeTemperature(data)
rho = data.variables["rho"] * data.unitNumberDensity
# bx = data.variables["bx1"]
# by = data.variables["bx2"]
# bz = data.variables["bx3"]
# b_tot = np.sqrt(bx**2 + by**2 + bz**2)
Tools.plotVariable(data, rho, show=True, log=True, clear=False, interpolate=False, x_range=x_range, y_range=y_range)
# Tools.plotMagneticFieldLines(data, show=True, filename="vel_fieldlines_50_temp", norm=True, x_range=x_range, y_range=y_range)
# Tools.computeTotalMasses(path)

# masses, times = Tools.computeTotalMasses("./")
# Tools.plotMassLosses('./')
# Tools.plotIonizationParameter(data, "ionization_param")
# Tools.removeFilesWithStride("./", 20)
# plotUsefulQuantities(data)
