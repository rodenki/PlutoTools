import os
import numpy as np
import scipy
from scipy import stats
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
np.set_printoptions(threshold=500)

from . import Data


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
        data = Data()
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
    def computeStreamline(data, point, x, y, vx1, vx2, vx3, rho, prs, x_range, y_range):
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

        # mimics the wind launching front
        H = 4.8 * Tools.pressureScaleHeightFlat(data)
        xticks = data.x1

        x, y = [], []

        while solver.y[1] > Tools.interpolatePoint(xticks, H, solver.y[0]):
            solver.integrate(t1, step=True)
            x.append(solver.y[0])
            y.append(solver.y[1])
            # print(solver.y)
        print(solver.y)
        print("Computing Jacobi potential...")
        potential = Tools.computeJacobiPotential(x, y, vx3, rho, prs, x_range, y_range)
        print("Computed Jacobi potential")
        return solver.y[0], potential

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
        losses = 2 * surface * rho[tempRange] * vx1[tempRange] * data.year / data.solarMass
        x_start = r * np.sin(theta)
        y_start = r * np.cos(theta)

        x, y = Tools.polarCoordsToCartesian(data.x1, data.x2)

        Tools.transformVelocityFieldToCylindrical(data)
        x_range = [1, 100, 1000]
        y_range = [0, 100, 1000]
        x, y, vx1 = Tools.interpolateToUniformGrid(data, data.variables["vx1"], x_range, y_range)
        x, y, vx2 = Tools.interpolateToUniformGrid(data, data.variables["vx2"], x_range, y_range)
        x, y, vx3 = Tools.interpolateToUniformGrid(data, data.variables["vx3"], x_range, y_range)
        x, y, rho = Tools.interpolateToUniformGrid(data, data.variables["rho"], x_range, y_range)
        x, y, prs = Tools.interpolateToUniformGrid(data, data.variables["prs"], x_range, y_range)


        vx1 = -vx1
        vx2 = -vx2

        losses = losses[20:]
        x_start = x_start[20:]
        y_start = y_start[20:]

        radii = []
        potentials = []

        for i, j in zip(x_start, y_start):
            radius, potential = Tools.computeStreamline(data, (i, j), x, y, vx1, vx2, vx3, rho, prs, x_range, y_range)
            radii.append(radius)
            potentials.append(potential)

        return radii, losses, potentials

    @staticmethod
    def jacobiPotential(rho, prs, vx3, r):
        return prs / rho + 1 / r - 0.5 * r**4 * vx3**2

    @staticmethod
    def computeJacobiPotential(x, y, vx3, rho, prs, x_range, y_range):
        # xticks = np.linspace(x_range[0], x_range[1], x_range[2])
        # yticks = np.linspace(y_range[0], y_range[1], y_range[2])
        potential = []

        for xx, yy in zip(x, y):
            rho_i = Tools.interpolatePoint2D(x_range, y_range, rho, (xx, yy))
            prs_i = Tools.interpolatePoint2D(x_range, y_range, prs, (xx, yy))
            vx3_i = Tools.interpolatePoint2D(x_range, y_range, vx3, (xx, yy))
            r = np.linalg.norm((xx, yy))
            potential.append(Tools.jacobiPotential(rho_i, prs_i, vx3_i, r)[0])
        return potential


    @staticmethod
    def computeMassLoss(path):
        sim = Data()
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
        sim = Data()
        sim.loadGridData()

        for filename in os.listdir(path):
            if filename.endswith(".h5"):
                frameIndex = int(filename.split('.')[1])
                if frameIndex in frameRange:
                    print("Averaging " + str(variable) + " frame: " + str(frameIndex))
                    sim.loadData(os.path.join(path, filename))
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
    def interpolatePoint(ticks, data, point):
        f = scipy.interpolate.interp1d(ticks, data)
        return f(point)

    @staticmethod
    def interpolatePoint2D(x_range, y_range, data, p):
        pp = [[(p[1] - y_range[0]) * y_range[2] / (y_range[1] - y_range[0])],
                      [(p[0] - x_range[0]) * x_range[2] / (x_range[1] - x_range[0])]]
        pp = np.array(pp)
        return map_coordinates(data, pp, order=1)

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
