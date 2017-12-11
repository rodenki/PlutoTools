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

def plotArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frame', nargs=1)
    parser.add_argument('--magfield', nargs=6)
    parser.add_argument('--velfield', nargs=6)
    parser.add_argument('--var', nargs=1)
    parser.add_argument('--mach', nargs='?')
    parser.add_argument('--temp', nargs='?')
    parser.add_argument('-l', '--log')

    args = parser.parse_args()
    frame = ''
    path = "./"
    data = None

    if args.frame:
        frame = args.frame[0]
        for i in range(4-len(frame)):
            frame = "0" + frame
        frame = "data." + frame + ".dbl.h5"
        data = sim.SimulationData()
        data.loadData(frame)
        data.loadGridData()

    if args.var:
        try:
            variable = data.variables[args.var[0]]
            if args.log:
                Tools.plotVariable(data, variable, show=True, log=True, clear=False, interpolate=False)
            else:
                Tools.plotVariable(data, variable, show=True, log=False, clear=False, interpolate=False)

        except KeyError:
            print("Variable not found in data")

    if args.magfield:
        rho = data.variables["rho"] * data.unitNumberDensity
        x_range = [float(i) for i in args.magfield[:3]]
        y_range = [float(i) for i in args.magfield[3:]]
        h = Tools.pressureScaleHeight(data)

        Tools.plotVariable(data, rho, show=False, log=True, clear=False, interpolate=True, x_range=x_range, y_range=y_range)
        Tools.plotMagneticFieldLines(data, show=True, clear=False, filename="mag_fieldlines", norm=True, x_range=x_range, y_range=y_range)
        # Tools.plotLineData(data, h, x_range=x_range)

    if args.velfield:
        rho = data.variables["rho"] * data.unitNumberDensity
        x_range = [float(i) for i in args.velfield[:3]]
        y_range = [float(i) for i in args.velfield[3:]]

        Tools.plotVariable(data, rho, show=False, log=True, clear=False, interpolate=True, x_range=x_range, y_range=y_range)
        Tools.plotVelocityFieldLines(data, show=True, filename="vel_fieldlines", norm=True, x_range=x_range, y_range=y_range)

    if args.mach:
        mach = Tools.computeMachNumbers(data)
        Tools.plotVariable(data, mach, show=True, log=True, clear=False, interpolate=False)

    if args.temp:
        temp = Tools.computeTemperature(data)
        Tools.plotVariable(data, temp, show=True, log=True, clear=False, interpolate=False)



plotArguments()

# avgRange = range(40, 135)
#
# rho = Tools.averageFrames(path, "rho", avgRange)
# prs = Tools.averageFrames(path, "prs", avgRange)
# vx1 = Tools.averageFrames(path, "vx1", avgRange)
# vx2 = Tools.averageFrames(path, "vx2", avgRange)

# data = sim.SimulationData()
# frame, x_range, y_range = getArgs()

# data.variables["rho"] = rho
# data.variables["prs"] = prs
# data.variables["vx1"] = vx1
# data.variables["vx2"] = vx2


# velocities = np.sqrt(data.variables["vx1"]**2 + data.variables["vx2"]**2) * data.unitVelocity
# escapeVel = np.sqrt(2.0*data.G * data.solarMass / (data.x1 * data.unitLength))
# soundspeed = np.sqrt(5.0/3.0 * data.variables["prs"] / data.variables["rho"]) * data.unitVelocity
# escapeVel = np.tile(escapeVel, (soundspeed.shape[0], 1))
#
# escapeVel = soundspeed / escapeVel
# Tools.plotVariable(data, escapeVel, show=False, clear=False, log=False, interpolate=True, x_range=x_range,
#                     y_range=y_range, vlimits=(0.1, 1.5))
# Tools.plotVelocityFieldLines(data, show=False, filename="vel_escape", norm=True, x_range=x_range, y_range=y_range)


# Tools.plotCumulativeMassloss(frame)

# temp = Tools.computeTemperature(data)
# rho = data.variables["rho"] * data.unitNumberDensity
# bx = data.variables["bx1"]
# by = data.variables["bx2"]
# bz = data.variables["bx3"]
# b_tot = np.sqrt(bx**2 + by**2 + bz**2)
# Tools.plotVariable(data, rho, show=False, log=True, clear=False, interpolate=True, x_range=x_range, y_range=y_range)
# Tools.plotMagneticFieldLines(data, show=False, filename="mag_fieldlines_90z", norm=True, x_range=x_range, y_range=y_range)
# # Tools.computeTotalMasses(path)

# masses, times = Tools.computeTotalMasses("./")
# Tools.plotMassLosses('./')
# Tools.plotIonizationParameter(data, "ionization_param")
# Tools.removeFilesWithStride("./", 20)
# plotUsefulQuantities(data)
