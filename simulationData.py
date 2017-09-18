import sys
import os

import h5py
import numpy as np
import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate
from scipy import stats
import xml.etree.cElementTree as xml
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
np.set_printoptions(threshold=500)


class SimulationData:
    def __init__(self):
        self.filename = ""
        self.unitDensity = 5.974e-07
        self.unitNumberDensity = 3.572e+17
        self.unitPressure = 5.329e+06
        self.unitVelocity = 2.987e+06
        self.unitLength = 1.496e+13
        self.unitTimeYears = 1.588e-01
        self.solarMass = 2e33
        self.year = 31536000
        self.mu = 1.37125
        self.kb = 1.3806505e-16
        self.mp = 1.67262171e-24
        self.G = 6.6726e-8
        self.time = 0.0
        self.cell_coordinates_x = np.array([])
        self.cell_coordinates_x = np.array([])
        self.cell_coordinates_x = np.array([])
        self.x1 = np.array([])
        self.x2 = np.array([])
        self.dx1 = np.array([])
        self.dx2 = np.array([])
        self.variables = {}
        self.rho = np.array([])
        self.prs = np.array([])
        self.vx1 = np.array([])
        self.vx2 = np.array([])
        self.vx3 = np.array([])
        self.timestep = ""
        self.hdf5File = None

    def orbits(self, radius, time):
        return np.sqrt(self.G * self.solarMass / (radius*self.unitLength)**3) * time * self.year / (2*np.pi)

    def loadVariable(self, title):
        try:
            self.hdf5File = h5py.File(self.filename, 'r')
        except NameError:
            print("File " + self.filename + " not found")

        # Getting timestep
        data = list(self.hdf5File.items())
        self.timestep = data[0][0]


        return np.array(self.hdf5File[self.timestep]['vars'][title])
        self.hdf5File.close()

    def loadData(self, filename):
        self.filename = filename

        try:
            self.hdf5File = h5py.File(filename, 'r')
        except NameError:
            print("File " + filename + " not found")

        # self.cell_coordinates_x = np.array(self.hdf5File['cell_coords']['X'])
        # self.cell_coordinates_y = np.array(self.hdf5File['cell_coords']['Y'])
        # self.cell_coordinates_z = np.array(self.hdf5File['cell_coords']['Z'])

        # Getting timestep
        data = list(self.hdf5File.items())
        self.timestep = data[0][0]

        # Getting variable data
        self.variables["rho"] = np.array(self.hdf5File[self.timestep]['vars']['rho'])
        self.variables["prs"] = np.array(self.hdf5File[self.timestep]['vars']['prs'])
        self.variables["vx1"] = np.array(self.hdf5File[self.timestep]['vars']['vx1'])
        self.variables["vx2"] = np.array(self.hdf5File[self.timestep]['vars']['vx2'])
        self.variables["vx3"] = np.array(self.hdf5File[self.timestep]['vars']['vx3'])
        self.hdf5File.close()

        xmlPath = self.filename[:-2] + "xmf"
        tree = xml.parse(xmlPath)
        root = tree.getroot()
        self.time = root[0][0][0].get("Value")

    def loadFrame(self, frame):
        self.loadData("data." + frame + ".dbl.h5")

    def loadGridData(self):
        lines = [line.rstrip('\n') for line in open('grid.out')][9:]
        n_coords = int(lines[0])
        lines = lines[1:]
        x1_lines = lines[:n_coords]
        x2_lines = lines[n_coords+1:-2]
        x1_coords = []
        x2_coords = []
        [x1_coords.append(line.split('   ')) for line in x1_lines]
        [x2_coords.append(line.split('   ')) for line in x2_lines]
        x1_coords = np.asarray(x1_coords, dtype=np.float)
        x2_coords = np.asarray(x2_coords, dtype=np.float)
        self.x1 = [0.5*(x1_coords[i][1] + x1_coords[i][2]) for i in range(len(x1_coords))]
        self.x2 = [0.5*(x2_coords[i][1] + x2_coords[i][2]) for i in range(len(x2_coords))]
        self.dx1 = [x1_coords[i][2] - x1_coords[i][1] for i in range(len(x1_coords))]
        self.dx2 = [x2_coords[i][2] - x2_coords[i][1] for i in range(len(x2_coords))]

    def insertData(self, data, title):
        self.hdf5File = h5py.File(self.filename, 'a')
        self.hdf5File[self.timestep]["vars"].create_dataset(title, data=data)
        self.hdf5File.close()

        xmlPath = self.filename[:-2] + "xmf"
        tree = xml.parse(xmlPath)
        root = tree.getroot()
        grid = root[0][0]
        for child in grid:
            if child.get("Name") == "rho":
                newChild = deepcopy(child)
                newChild.set("Name", title)
                newChild[0].text = ".//" + self.filename + ":" + self.timestep + "/vars/" + title
                grid.append(newChild)
                break
        tree.write(xmlPath)

    def removeData(self, title):
        self.hdf5File = h5py.File(self.filename, 'a')
        del self.hdf5File[self.timestep]["vars"][title]
        self.hdf5File.close()

        xmlPath = self.filename[:-2] + "xmf"
        tree = xml.parse(xmlPath)
        root = tree.getroot()
        grid = root[0][0]
        for child in grid:
            if child.get("Name") == title:
                grid.remove(child)
                break
        tree.write(xmlPath)


class Tools:

    @staticmethod
    def removeFilesWithStride(path, stride):
        for current_file in os.listdir(path):
            if current_file.endswith(".h5") or current_file.endswith(".xmf"):
                frame = int(current_file.split('.')[1])
                if frame % stride != 0:
                    print("deleting frame " + str(frame))
                    os.remove(os.path.join(path, current_file))

    @staticmethod
    def computeTotalMass(data):
        pass

    @staticmethod
    def computeMachNumbers(data):
        vabs = Tools.computeAbsoluteVelocities(data) * data.unitVelocity
        temp = Tools.computeTemperature(data)
        cs = np.sqrt(data.kb * temp / (data.mu * data.mp))
        mach = vabs / cs
        return mach

    @staticmethod
    def computeSonicPoints(data):
        vabs = Tools.computeAbsoluteVelocities(data) * data.unitVelocity
        temp = Tools.computeTemperature(data)
        cs = np.sqrt(data.kb * temp / (data.mu * data.mp))
        mach = vabs / cs
        mach = stats.threshold(mach, threshmin=0.95, threshmax=1.05)
        return mach

    @staticmethod
    def computeTemperature(data):
        kelvin = 1.072914e+05
        mu = 1.37125
        return data.variables["prs"] / data.variables["rho"] * kelvin * mu

    @staticmethod
    def computeAbsoluteVelocities(data):
        return np.sqrt(data.variables["vx1"]**2 + data.variables["vx2"]**2)

    @staticmethod
    def computeTemperatureToFile(path, replace=False):
        sim = SimulationData()
        sim.loadData(path)
        kelvin = 1.072914e+05
        mu = 1.37125
        if replace:
            sim.removeData("Temp")
        temp = sim.variables["prs"] / sim.variables["rho"] * kelvin * mu
        sim.insertData(temp, "Temp")

    @staticmethod
    def computeTemperaturesToFile(path, replace=False):
        for file in os.listdir(path):
            if file.endswith(".h5"):
                print("Computing T for " + file)
                Tools.computeTemperatureToFile(os.path.join(path, file), replace=replace)

    @staticmethod
    def computeVelocityToFile(path, replace=False):
        sim = SimulationData()
        sim.loadData(path)
        if replace:
            sim.removeData("VABS")
        v = np.sqrt(sim.variables["vx1"]**2 + sim.variables["vx2"]**2)
        sim.insertData(v, "VABS")

    @staticmethod
    def computeVelocitiesToFile(path, replace=False):
        for file in os.listdir(path):
            if file.endswith(".h5"):
                print("Computing v for " + file)
                Tools.computeVelocityToFile(os.path.join(path, file), replace=replace)

    @staticmethod
    def computeMassLoss(path):
        sim = SimulationData()
        sim.loadData(path)
        sim.loadGridData()
        computeLimit = int(len(sim.dx1) * 0.95)
        temp = Tools.computeTemperature(sim)[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000]
        tempRange = range(min(tempRange), max(tempRange))
        rho = sim.variables["rho"][:,computeLimit] * sim.unitDensity
        vx1 = sim.variables["vx1"][:,computeLimit] * sim.unitVelocity


        surface = 0.5*np.pi / len(sim.x2) * sim.x1[computeLimit]**2 * 2.0 * np.pi * sim.unitLength**2
        massLoss = rho[tempRange] * surface * vx1[tempRange]
        totalMassLoss = np.add.reduce(massLoss)
        return totalMassLoss * sim.year / sim.solarMass, sim

    @staticmethod
    def computeMassLosses(path):
        losses = []
        times = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                loss, sim = Tools.computeMassLoss(os.path.join(path, file))
                times.append(float(sim.time) * sim.unitTimeYears)
                losses.append(loss)
                print("Massflux for " + file + ": " + str(loss))
        return losses, times

    @staticmethod
    def plotMassLosses(path, filename="losses.eps"):
        losses, times = Tools.computeMassLosses("./")
        losses = np.array(losses, dtype=np.double)
        times = np.array(times, dtype=np.double)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(times, losses)
        plt.xlabel(r't [yr]')
        plt.ylabel(r'$\dot{M}_w $ [$\frac{M_{\odot}}{\mathrm{yr}}$]')
        plt.savefig(filename)

    @staticmethod
    def computeCumulativeMassLoss(path):
        sim = SimulationData()
        sim.loadData(path)
        sim.loadGridData()

        losses = []
        cumulative_loss = 0.0

        for r in range(len(sim.dx1)):
            computeLimit = r
            temp = Tools.computeTemperature(sim)[:,computeLimit]
            tempRange = [i for i,v in enumerate(temp) if v > 1000]
            tempRange = range(min(tempRange), max(tempRange))
            rho = sim.variables["rho"][:,computeLimit] * sim.unitDensity
            vx1 = sim.variables["vx1"][:,computeLimit] * sim.unitVelocity


            surface = 0.5*np.pi / len(sim.x2) * sim.x1[computeLimit]**2 * 2.0 * np.pi * sim.unitLength**2
            massLoss = rho[tempRange] * surface * vx1[tempRange]
            totalMassLoss = np.add.reduce(massLoss)
            totalMassLoss *= sim.year / sim.solarMass
            if totalMassLoss < 0.0:
                totalMassLoss = 0.0
            cumulative_loss += totalMassLoss
            losses.append(cumulative_loss)
        return losses, sim

    @staticmethod
    def plotCumulativeMassloss(path, filename="cumulative_losses.eps"):
        losses, sim = Tools.computeCumulativeMassLoss(path)
        losses = np.array(losses, dtype=np.double)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(sim.x1, losses)
        plt.xlabel(r'r [AU]')
        plt.ylabel(r'$\dot{M}_w $ [$\frac{M_{\odot}}{\mathrm{yr}}$]')
        plt.savefig(filename)

    @staticmethod
    def polarCoordsToCartesian(x1, x2):
        r_matrix, th_matrix = np.meshgrid(x1, x2)
        x = r_matrix * np.sin(th_matrix)
        y = r_matrix * np.cos(th_matrix)
        return x, y

    @staticmethod
    def plotVariable(data, variable, filename, log=True, show=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.figure(figsize=(6,4))
        if log:
            plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=variable.min(), vmax=variable.max()), cmap=cm.inferno)
        else:
            plt.pcolormesh(x, y, variable, cmap=cm.inferno)
        plt.colorbar()
        plt.xlabel(r'r')
        plt.ylabel(r'z')
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400)
        plt.cla()
        plt.close()

    @staticmethod
    def plotDensity(data, filename, show=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.figure(figsize=(6,4))
        rho = data.variables["rho"]
        plt.pcolormesh(x, y, rho, norm=LogNorm(vmin=rho.min(), vmax=rho.max()), cmap=cm.inferno)
        plt.colorbar()
        plt.xlabel(r'r')
        plt.ylabel(r'z')
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400)
        plt.cla()
        plt.close()

    @staticmethod
    def plotSonicBarrier(data, filename, show=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        mach = Tools.computeSonicPoints(data)
        plt.scatter(x, y, 0.2*mach, c='r')
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400)
        plt.cla()
        plt.close()

    @staticmethod
    def plotVelocityField(data, filename, dx1=10, dx2=5, scale=40, width=0.001, x1_start=0, overlay=False, wind_only=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        if not overlay:
            plt.clf()
            plt.figure(figsize=(6,4))

        if wind_only:
            temp = Tools.computeTemperature(data)

            for r in range(x1_start, len(data.x1), dx1):
                tempRange = [i for i,t in enumerate(temp[:,r]) if t > 1000]
                plt.quiver(x[:,r][tempRange[0]:tempRange[-1]:dx2],
                           y[:,r][tempRange[0]:tempRange[-1]:dx2],
                           data.variables["vx1"][:,r][tempRange[0]:tempRange[-1]:dx2],
                           data.variables["vx2"][:,r][tempRange[0]:tempRange[-1]:dx2],
                           width=width, scale=scale, color='k')
        else:
            for r in range(x1_start, len(data.x1), dx1):
                plt.quiver(x[:,r][::dx2],
                           y[:,r][::dx2],
                           data.variables["vx1"][:,r][::dx2],
                           data.variables["vx2"][:,r][::dx2],
                           width=width, scale=scale, color='k')



        plt.savefig(filename + ".png", dpi=400)
        plt.close('all')

    @staticmethod
    def plotIonizationParameter(data, filename="ion_param", overlay=False, show=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.figure(figsize=(8, 6))
        rho = data.variables["rho"] * data.unitNumberDensity
        temp = Tools.computeTemperature(data)
        t = np.argwhere(temp < 1000)
        for element in t:
            rho[element[0], element[1]] = np.nan

        r2 = x**2 + y**2
        r2 *= data.unitLength**2
        ion_param = np.log10(2e30 / (r2 * rho))
        plt.pcolormesh(x, y, ion_param, vmin=np.nanmin(ion_param), vmax=np.nanmax(ion_param), cmap=cm.inferno)
        plt.colorbar()
        plt.xlabel(r'r')
        plt.ylabel(r'z')
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400)
        plt.cla()
        plt.close()

    @staticmethod
    def interpolateRadialGrid(data, newTicks):
        for key, value in data.variables.items():
            x1 = data.x1
            interpolated = np.array(np.zeros(shape=(value.shape[0], len(newTicks))))

            for i in range(value.shape[0]):
                f = interpolate.interp1d(x1, value[i])
                interpolated[i] = f(newTicks)

            data.variables[key] = interpolated
        data.x1 = newTicks
