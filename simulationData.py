import sys
import os

import h5py
import numpy as np
import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import griddata
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
        self.time = 0.0
        self.cell_coordinates_x = np.array([])
        self.cell_coordinates_x = np.array([])
        self.cell_coordinates_x = np.array([])
        self.x1 = np.array([])
        self.x2 = np.array([])
        self.dx1 = np.array([])
        self.dx2 = np.array([])
        self.rho = np.array([])
        self.prs = np.array([])
        self.vx1 = np.array([])
        self.vx2 = np.array([])
        self.vx3 = np.array([])
        self.timestep = ""
        self.hdf5File = None

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

        self.cell_coordinates_x = np.array(self.hdf5File['cell_coords']['X'])
        self.cell_coordinates_y = np.array(self.hdf5File['cell_coords']['Y'])
        self.cell_coordinates_z = np.array(self.hdf5File['cell_coords']['Z'])

        # Getting timestep
        data = list(self.hdf5File.items())
        self.timestep = data[0][0]

        # Getting variable data
        self.rho = np.array(self.hdf5File[self.timestep]['vars']['rho'])
        self.prs = np.array(self.hdf5File[self.timestep]['vars']['prs'])
        self.vx1 = np.array(self.hdf5File[self.timestep]['vars']['vx1'])
        self.vx2 = np.array(self.hdf5File[self.timestep]['vars']['vx2'])
        self.vx3 = np.array(self.hdf5File[self.timestep]['vars']['vx3'])
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
    def computeTemperatureToFile(path, replace=False):
        sim = SimulationData(path)
        kelvin = 1.072914e+05
        mu = 1.37125
        if replace:
            sim.removeData("Temp")
        temp = sim.prs / sim.rho * kelvin * mu
        sim.insertData(temp, "Temp")

    @staticmethod
    def computeTemperaturesToFile(path, replace=False):
        for file in os.listdir(path):
            if file.endswith(".h5"):
                print("Computing T for " + file)
                DataHandler.computeTemperatureToFile(os.path.join(path, file), replace=replace)

    @staticmethod
    def computeVelocityToFile(path, replace=False):
        sim = SimulationData(path)
        if replace:
            sim.removeData("VABS")
        v = np.sqrt(sim.vx1**2 + sim.vx2**2)
        sim.insertData(v, "VABS")

    @staticmethod
    def computeVelocitiesToFile(path, replace=False):
        for file in os.listdir(path):
            if file.endswith(".h5"):
                print("Computing v for " + file)
                DataHandler.computeVelocityToFile(os.path.join(path, file), replace=replace)

    @staticmethod
    def computeMassLoss(path):
        sim = SimulationData()
        sim.loadData(path)
        sim.loadGridData()
        computeLimit = int(len(sim.dx1) * 0.95)
        temp = sim.loadVariable("Temp")[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000]
        tempRange = range(min(tempRange), max(tempRange))
        rho = sim.rho[:,computeLimit] * sim.unitDensity
        vx1 = sim.vx1[:,computeLimit] * sim.unitVelocity


        surface = 0.5*np.pi / len(sim.x2) * sim.r[computeLimit]**2 * 2.0 * np.pi * sim.unitLength**2
        massLoss = rho[tempRange] * surface * vx1[tempRange]
        totalMassLoss = np.add.reduce(massLoss)
        return totalMassLoss * sim.year / sim.solarMass, sim

    @staticmethod
    def computeMassLosses(path):
        losses = []
        times = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                loss, sim = DataHandler.computeMassLoss(os.path.join(path, file))
                times.append(float(sim.time) * sim.unitTimeYears)
                losses.append(loss)
                print("Massflux for " + file + ": " + str(loss))
        return losses, times

    @staticmethod
    def plotMassLosses(path):
        losses, times = DataHandler.computeMassLosses("./")
        losses = np.array(losses, dtype=np.double)
        times = np.array(times, dtype=np.double)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(times, losses)
        plt.xlabel(r't [yr]')
        plt.ylabel(r'$\dot{M}_w $ [$\frac{M_{\odot}}{\mathrm{yr}}$]')
        plt.show()

    @staticmethod
    def polarCoordsToCartesian(x1, x2):
        r_matrix, th_matrix = np.meshgrid(x1, x2)
        x = r_matrix * np.sin(th_matrix)
        y = r_matrix * np.cos(th_matrix)
        return x, y

    @staticmethod
    def plotDensity(data, filename):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.clf()
        plt.figure(figsize=(6,4))
        plt.pcolormesh(x, y, data.rho, norm=LogNorm(vmin=data.rho.min(), vmax=data.rho.max()), cmap=cm.inferno)
        plt.colorbar()
        plt.xlabel(r'r')
        plt.ylabel(r'z')
        plt.savefig(filename + ".png", dpi=200)

    @staticmethod
    def plotVelocityField(data, filename, overlay=False, wind_only=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        if not overlay:
            plt.clf()
            plt.figure(figsize=(6,4))

        plt.quiver(x, y, data.vx1, data.vx2)
        plt.savefig(filename + ".png", dpi=200)




data = SimulationData()
data.loadFrame("0001")
data.loadGridData()

Tools.plotDensity(data, "test")
Tools.plotVelocityField(data, "field")


#xx, yy = np.meshgrid(data.x1, data.x2)
