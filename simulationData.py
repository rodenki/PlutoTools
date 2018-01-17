import os

import h5py
import numpy as np
import scipy
from scipy import stats
from scipy.ndimage import map_coordinates
import xml.etree.cElementTree as xml
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker, cm
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
        self.cell_coordinates_x1 = np.array([])
        self.cell_coordinates_x2 = np.array([])
        self.cell_coordinates_x3 = np.array([])
        self.x1 = np.array([])
        self.x2 = np.array([])
        self.x3 = np.array([])
        self.dx1 = np.array([])
        self.dx2 = np.array([])
        self.dx3 = np.array([])
        self.variables = {}
        self.timestep = ""
        self.hdf5File = None

    def orbits(self, radius, time):
        return time * self.year * np.sqrt(self.G * self.solarMass / (radius*self.unitLength)**3) / (2.0 * np.pi)

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

        self.cell_coordinates_x1 = np.array(self.hdf5File['cell_coords']['X'])
        self.cell_coordinates_x2 = np.array(self.hdf5File['cell_coords']['Y'])
        self.cell_coordinates_x3 = np.array(self.hdf5File['cell_coords']['Z'])

        # Getting timestep
        data = list(self.hdf5File.items())
        self.timestep = data[0][0]

        # Getting variable data
        self.variables["rho"] = np.array(self.hdf5File[self.timestep]['vars']['rho'])
        self.variables["prs"] = np.array(self.hdf5File[self.timestep]['vars']['prs'])
        self.variables["vx1"] = np.array(self.hdf5File[self.timestep]['vars']['vx1'])

        try:
            self.variables["vx2"] = np.array(self.hdf5File[self.timestep]['vars']['vx2'])
        except KeyError:
            print("No vx2 data")


        try:
            self.variables["vx3"] = np.array(self.hdf5File[self.timestep]['vars']['vx3'])
        except KeyError:
            print("No vx3 data")

        try:
            self.variables["bx1"] = np.array(self.hdf5File[self.timestep]['vars']['bx1'])
            self.variables["bx2"] = np.array(self.hdf5File[self.timestep]['vars']['bx2'])
            self.variables["bx3"] = np.array(self.hdf5File[self.timestep]['vars']['bx3'])
        except KeyError:
            print("Magnetic field not present.")

        self.hdf5File.close()

        xmlPath = self.filename[:-2] + "xmf"
        tree = xml.parse(xmlPath)
        root = tree.getroot()
        self.time = float(root[0][0][0].get("Value"))
        self.loadGridData()

    def loadFrame(self, frame):
        self.loadData("data." + frame + ".dbl.h5")

    def loadGridData(self):
        lines = [line.rstrip('\n') for line in open('grid.out')]
        for i, line in enumerate(lines):
            if line[0] != '#':
                lines = lines[i:]
                break

        n1_coords = int(lines[0])
        lines = lines[1:]
        x1_lines = lines[:n1_coords]
        n2_coords = int(lines[n1_coords])
        lines = lines[n1_coords+1:]
        x2_lines = lines[:n2_coords]
        n3_coords = int(lines[n2_coords])
        lines = lines[n2_coords+1:]
        x3_lines = lines

        x1_coords = []
        x2_coords = []
        x3_coords = []
        [x1_coords.append(line.split('   ')) for line in x1_lines]
        [x2_coords.append(line.split('   ')) for line in x2_lines]
        [x3_coords.append(line.split('   ')) for line in x3_lines]

        x1_coords = np.asarray(x1_coords, dtype=np.float)
        x2_coords = np.asarray(x2_coords, dtype=np.float)
        x3_coords = np.asarray(x3_coords, dtype=np.float)

        self.x1 = np.array([0.5*(x1_coords[i][1] + x1_coords[i][2]) for i in range(len(x1_coords))])
        self.x2 = np.array([0.5*(x2_coords[i][1] + x2_coords[i][2]) for i in range(len(x2_coords))])
        self.x3 = np.array([0.5*(x3_coords[i][1] + x3_coords[i][2]) for i in range(len(x3_coords))])
        self.dx1 = np.array([x1_coords[i][2] - x1_coords[i][1] for i in range(len(x1_coords))])
        self.dx2 = np.array([x2_coords[i][2] - x2_coords[i][1] for i in range(len(x2_coords))])
        self.dx3 = np.array([x3_coords[i][2] - x3_coords[i][1] for i in range(len(x3_coords))])

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
    def computeAbsoluteVelocities(data):
        return np.sqrt(data.variables["vx1"]**2 + data.variables["vx2"]**2)

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
    def computeTotalMass(path):
        data = SimulationData()
        data.loadData(path)
        data.loadGridData()
        rho = data.variables["rho"]
        # dx2 = 0.5*np.pi / len(sim.x2)
        dV = (data.x1**2 - (data.x1 - data.dx1)**2) / (4.0*len(data.x2)) * 2.0 * np.pi * data.x1
        dV = np.tile(dV, (len(data.x2), 1))
        mass = rho * dV
        total = np.sum(mass) * data.unitDensity * data.unitLength**3 / data.solarMass
        return total, data

    @staticmethod
    def computeTotalMasses(path):
        masses = []
        times = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                mass, sim = Tools.computeTotalMass(os.path.join(path, file))
                times.append(float(sim.time) * sim.unitTimeYears)
                masses.append(mass)
                print("Mass for " + file + ": " + str(mass))
        return np.array(masses), np.array(times)

    # Returns single interpolated value on a regular grid (faster than griddata)
    @staticmethod
    def singlePointInterpolation(t, p, vx1, vx2, x_range, y_range):
        pp = [[(p[1] - y_range[0]) * y_range[2] / (y_range[1] - y_range[0])],
                      [(p[0] - x_range[0]) * x_range[2] / (x_range[1] - x_range[0])]]
        pp = np.array(pp)
        return [map_coordinates(vx1, pp, order=1), map_coordinates(vx2, pp, order=1)]

    @staticmethod
    def computeStreamline(data, point, x, y, vx1, vx2, x_range, y_range):
        # print(Tools.singlePointInterpolation(0, p0, vx1, vx2, x_range, y_range))
        # vx, vy = np.ravel(data.variables["vx1"]), np.ravel(data.variables["vx2"])

        p0 = point
        t0 = 0.0
        t1 = 1000
        # print(Tools.singlePointInterpolation(t0, p0, vx1, vx2, x_range, y_range))
        solver = scipy.integrate.ode(Tools.singlePointInterpolation)
        solver.set_integrator("vode", rtol=1e-10)
        solver.set_f_params(vx1, vx2, x_range, y_range)
        solver.set_initial_value(p0, t0)
        # x, y = [], []

        # mimics the wind launching front
        H = 4.8 * Tools.pressureScaleHeightFlat(data)
        xticks = data.x1

        while solver.y[1] > Tools.interpolatePoint(xticks, H, solver.y[0]):
            solver.integrate(t1, step=True)
            # x.append(solver.y[0])
            # y.append(solver.y[1])
        print(solver.y)
        return solver.y[0]

    @staticmethod
    def computeRadialMassLosses(data):
        computeLimit = int(len(data.dx1) * 0.99)
        rho = data.variables["rho"][:,computeLimit] * data.unitDensity
        vx1 = data.variables["vx1"][:,computeLimit] * data.unitVelocity
        temp = Tools.computeTemperature(data)[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000 and vx1[i] > 0]
        tempRange = range(min(tempRange), max(tempRange))
        r = data.x1[computeLimit]
        theta = data.x2[tempRange]

        surface = 0.5*np.pi / len(data.x2) * r**2 * 2.0 * np.pi * data.unitLength**2
        losses = surface * rho[tempRange] * vx1[tempRange] * data.year / data.solarMass
        x_start = r * np.sin(theta)
        y_start = r * np.cos(theta)

        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)

        Tools.transformVelocityFieldToCylindrical(data)
        x_range = [1, 100, 1000]
        y_range = [0, 100, 1000]
        x, y, vx1 = Tools.interpolateToUniformGrid(data, data.variables["vx1"], x_range, y_range)
        x, y, vx2 = Tools.interpolateToUniformGrid(data, data.variables["vx2"], x_range, y_range)
        vx1 = -vx1
        vx2 = -vx2

        losses = losses[19:]
        x_start = x_start[19:]
        y_start = y_start[19:]

        radii = []

        for i, j in zip(x_start, y_start):
            radii.append(Tools.computeStreamline(data, (i, j), x, y, vx1, vx2, x_range, y_range))

        return radii, losses

    @staticmethod
    def computeMassLoss(path):
        sim = SimulationData()
        sim.loadData(path)
        sim.loadGridData()
        computeLimit = int(len(sim.dx1) * 0.99)
        rho = sim.variables["rho"][:,computeLimit] * sim.unitDensity
        vx1 = sim.variables["vx1"][:,computeLimit] * sim.unitVelocity
        temp = Tools.computeTemperature(sim)[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000]
        tempRange = range(min(tempRange), max(tempRange))
        surface = 0.5*np.pi / len(sim.x2) * sim.x1[computeLimit]**2 * 2.0 * np.pi * sim.unitLength**2
        massLoss = surface * rho[tempRange] * vx1[tempRange]
        totalMassLoss = np.sum(massLoss)
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
        np.savetxt("losses.dat", (times, losses))

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(times, losses)
        plt.xlabel(r't [yr]')
        plt.ylabel(r'$\dot{M}_w $ [$\frac{M_{\odot}}{\mathrm{yr}}$]')
        plt.savefig(filename)

    @staticmethod
    def averageFrames(path, variable, frameRange):
        frames = []
        sim = SimulationData()

        for filename in os.listdir(path):
            if filename.endswith(".h5"):
                frameIndex = int(filename.split('.')[1])
                if frameIndex in frameRange:
                    sim.loadData(os.path.join(path, filename))
                    sim.loadGridData()
                    frames.append(sim.variables[variable])
        frames = np.array(frames)
        averaged = np.mean(frames, axis=0)
        return averaged

    @staticmethod
    def pressureScaleHeight(data):
        temp = Tools.computeTemperature(data)
        cs = np.sqrt(data.kb * temp / (data.mu * data.mp)) / data.unitVelocity
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        omega = np.sqrt(1.0 / x**3)
        H = cs / omega
        return H[-1]

    @staticmethod
    def pressureScaleHeightFlat(data):
        temp = Tools.computeTemperature(data)
        cs = np.sqrt(data.kb * temp / (data.mu * data.mp)) / data.unitVelocity
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        omega = np.sqrt(1.0 / x**3)
        H = cs / omega
        return np.power(H[-1], 4.5/5.0)

    @staticmethod
    def interpolatePoint(ticks, data, point):
        f = scipy.interpolate.interp1d(ticks, data)
        return f(point)

    @staticmethod
    def plotLineData(data, lineData, show=True, filename="data", x_range=[0.33, 99, 100]):
        newTicks = np.linspace(*x_range)
        f = scipy.interpolate.interp1d(data.x1, lineData)
        interpolated = f(newTicks)

        plt.plot(newTicks, interpolated, 'g')

        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

    @staticmethod
    def plotVariable(data, variable, filename="data", log=True, show=True,
                     clear=True, interpolate=False, x_range=[0.33, 99, 100],
                     y_range=[0.33, 99, 100], vlimits=(0, 1), figsize=(10, 7)):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.figure(figsize=figsize)
        # plt.tight_layout()

        if interpolate:
            x, y, variable = Tools.interpolateToUniformGrid(data, variable, x_range, y_range)

        if log:
            plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=np.nanmin(variable),
                                                        vmax=np.nanmax(variable)), cmap=cm.inferno)
            # plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=100,
            #                                             vmax=1e4), cmap=cm.inferno)
        else:
            plt.pcolormesh(x, y, variable, cmap=cm.inferno) #, vmin=vlimits[0], vmax=vlimits[1])

        cb = plt.colorbar()
        tick_locator = ticker.LogLocator(numdecs=10)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.xlabel('Radius [AU]')
        plt.ylabel('z [AU]')
        orbits = data.orbits(1.0, data.time)
        plt.title("t = " + str(data.time) + ", " + str(int(orbits)) + " orbits")
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')
        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def plotSonicBarrier(data, filename="sonic", show=False, clear=True):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        mach = Tools.computeSonicPoints(data)
        plt.scatter(x, y, 0.2*mach, c='r')
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400)

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def plotVelocityField(data, filename="vel_field", dx1=10, dx2=5, scale=40,
                          width=0.001, x1_start=0, wind_only=False, clear=True,
                          show=False, norm=True):
        Tools.transformVelocityFieldToCylindrical(data)
        Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)

        # vx1 = data.variables["vx1"]
        # vx2 = data.variables["vx2"]
        # x = data.x1
        # y = data.x2

        if norm:
            n = np.sqrt(vx1**2 + vx2**2)
            vx1 /= n
            vx2 /= n

        if wind_only:
            temp = Tools.computeTemperature(data)

            for r in range(x1_start, len(data.x1), dx1):
                tempRange = [i for i,t in enumerate(temp[:,r]) if t > 1000]
                plt.quiver(x[:,r][tempRange[0]:tempRange[-1]:dx2],
                           y[:,r][tempRange[0]:tempRange[-1]:dx2],
                           vx1[:,r][tempRange[0]:tempRange[-1]:dx2],
                           vx2[:,r][tempRange[0]:tempRange[-1]:dx2],
                           width=width, scale=scale, color='k')
        else:
            for r in range(x1_start, len(data.x1), dx1):
                plt.quiver(x[:,r][::dx2],
                           y[:,r][::dx2],
                           vx1[:,r][::dx2],
                           vx2[:,r][::dx2],
                           width=width, scale=scale, color='k')

        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def transformVelocityFieldToCylindrical(data):
        vx1 = data.variables["vx1"]
        vx2 = data.variables["vx2"]
        x2 = np.transpose(np.tile(data.x2, (len(data.x1), 1)))

        data.variables["vx1"] = vx1 * np.sin(x2) + vx2 * np.cos(x2)
        data.variables["vx2"] = vx1 * np.cos(x2) - vx2 * np.sin(x2)

    @staticmethod
    def transformMagneticFieldToCylindrical(data):
        bx1 = data.variables["bx1"]
        bx2 = data.variables["bx2"]
        x2 = np.transpose(np.tile(data.x2, (len(data.x1), 1)))

        data.variables["bx1"] = bx1 * np.sin(x2) + bx2 * np.cos(x2)
        data.variables["bx2"] = bx1 * np.cos(x2) - bx2 * np.sin(x2)

    @staticmethod
    def plotMagneticField(data, filename="mag_field", dx1=10, dx2=5, scale=40,
                          width=0.001, x1_start=0, clear=True, show=False,
                          norm=True, x_range=[0.33, 99, 100], y_range=[0.33, 99, 100]):

        Tools.transformMagneticFieldToCylindrical(data)
        # Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        bx1 = data.variables["bx1"]
        bx2 = data.variables["bx2"]

        x, y, bx1 = Tools.interpolateToUniformGrid(data, bx1, x_range, y_range)
        x, y, bx2 = Tools.interpolateToUniformGrid(data, bx2, x_range, y_range)

        if norm:
            n = np.sqrt(bx1**2 + bx2**2)
            bx1 /= n
            bx2 /= n

        # plt.figure(figsize=(10, 7))
        plt.quiver(x, y, bx1, bx2,
                   width=width, scale=scale, color='k')

        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def plotMagneticFieldLines(data, filename="mag_fieldlines", dx1=10, dx2=5, scale=40,
                          width=0.001, x1_start=0, clear=True, show=False,
                          norm=True, x_range=[0.33, 99, 100], y_range=[0.33, 99, 100]):

        Tools.transformMagneticFieldToCylindrical(data)
        # Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
        # x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        bx1 = data.variables["bx1"]
        bx2 = data.variables["bx2"]

        x, y, bx1 = Tools.interpolateToUniformGrid(data, bx1, x_range, y_range)
        x, y, bx2 = Tools.interpolateToUniformGrid(data, bx2, x_range, y_range)

        if norm:
            n = np.sqrt(bx1**2 + bx2**2)
            bx1 /= n
            bx2 /= n

        # plt.figure(figsize=(10, 7))
        plt.streamplot(x, y, bx1, bx2, density=3, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def plotVelocityFieldLines(data, filename="vel_fieldlines", dx1=10, dx2=5, scale=40,
                          width=0.001, x1_start=0, clear=True, show=False,
                          norm=True, x_range=[0.33, 99, 100], y_range=[0.33, 99, 100]):

        Tools.transformVelocityFieldToCylindrical(data)
        # Tools.interpolateRadialGrid(data, np.linspace(0.4, 98.5, 500))
        # x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        vx1 = data.variables["vx1"]
        vx2 = data.variables["vx2"]

        x, y, vx1 = Tools.interpolateToUniformGrid(data, vx1, x_range, y_range)
        x, y, vx2 = Tools.interpolateToUniformGrid(data, vx2, x_range, y_range)

        if norm:
            n = np.sqrt(vx1**2 + vx2**2)
            vx1 /= n
            vx2 /= n

        # plt.figure(figsize=(10, 7))
        plt.streamplot(x, y, vx1, vx2, density=3, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

        if show:
            plt.show()
        else:
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def plotIonizationParameter(data, filename="ion_param", clear=True,
                                show=False):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        plt.figure(figsize=(10, 8))
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
            plt.savefig(filename + ".png", dpi=400, bbox_inches='tight')

        if clear:
            plt.cla()
            plt.close()

    @staticmethod
    def polarCoordsToCartesian(x1, x2):
        r_matrix, th_matrix = np.meshgrid(x1, x2)
        x = r_matrix * np.sin(th_matrix)
        y = r_matrix * np.cos(th_matrix)
        return x, y

    @staticmethod
    def interpolateRadialGrid(data, newTicks):
        for key, value in data.variables.items():
            x1 = data.x1
            interpolated = np.array(np.zeros(shape=(value.shape[0], len(newTicks))))

            for i in range(value.shape[0]):
                f = scipy.interpolate.interp1d(x1, value[i])
                interpolated[i] = f(newTicks)

            data.variables[key] = interpolated
        data.x1 = newTicks

    @staticmethod
    def interpolateToUniformGrid(data, variable, x_range, y_range):
        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)
        x = np.ravel(x)
        y = np.ravel(y)
        variable = np.ravel(variable)
        points = np.column_stack((x, y))
        grid_x, grid_y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))
        newVariable = scipy.interpolate.griddata(points, variable, (grid_x, grid_y))
        grid_r = np.sqrt(grid_x**2 + grid_y**2)
        newVariable[grid_r < 1.0] = np.nan
        return grid_x, grid_y, newVariable
