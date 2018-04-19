import os
import numpy as np
import scipy
from scipy import stats
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
np.set_printoptions(threshold=500)

from .Data import Data


class Compute:

    def __init__(self, data):
        self.data = data

    def computeAbsoluteVelocities(self):
        return np.sqrt(self.data.variables["vx1"]**2 + self.data.variables["vx2"]**2)

    def computeMachNumbers(self):
        vabs = self.computeAbsoluteVelocities() * self.data.unitVelocity
        temp = self.computeTemperature()
        cs = np.sqrt(self.data.kb * temp / (self.data.mu * self.data.mp))
        mach = vabs / cs
        return mach

    def computeSonicPoints(self):
        mach = self.computeMachNumbers()
        res = np.logical_and(mach > 0.995, mach < 1.005)
        res = np.where(res)
        x1 = self.data.x1[res[1]]
        x2 = self.data.x2[res[0]]
        x = x1 * np.sin(x2)
        y = x1 * np.cos(x2)
        sort = x.argsort()
        return x[sort], y[sort]

    def computeTemperature(self):
        kelvin = 1.072914e+05
        mu = 1.37125
        return self.data.variables["prs"] / self.data.variables["rho"] * kelvin * mu

    def computeTotalMass(self, data):
        rho = self.data.variables["rho"]
        dV = (self.data.x1**2 - (self.data.x1 - self.data.dx1)**2) / (4.0*len(self.data.x2)) * 2.0 * np.pi * self.data.x1
        dV = np.tile(dV, (len(self.data.x2), 1))
        mass = rho * dV
        total = np.sum(mass) * self.data.unitDensity * self.data.unitLength**3 / self.data.solarMass
        return total

    def computeTotalMasses(self, path):
        masses = []
        times = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                data = Data(os.path.join(path, file))
                mass = self.computeTotalMass(data)
                times.append(float(data.time) * data.unitTimeYears)
                masses.append(mass)
                print("Mass for " + file + ": " + str(mass))
        return np.array(masses), np.array(times)

    def jacobiPotential(self, rho, prs, vx3, r):
        return prs / rho + 1 / r# - 0.5 * r**4 * vx3**2

    def computeJacobiPotential(self, x, y, vx3, rho, prs, x_range, y_range):
        # xticks = np.linspace(x_range[0], x_range[1], x_range[2])
        # yticks = np.linspace(y_range[0], y_range[1], y_range[2])
        potential = []


        for xx, yy in zip(x, y):
            rho_i = Interpolate.interpolatePoint2D(x_range, y_range, rho, (xx, yy))
            prs_i = Interpolate.interpolatePoint2D(x_range, y_range, prs, (xx, yy))
            vx3_i = Interpolate.interpolatePoint2D(x_range, y_range, vx3, (xx, yy))
            r = np.linalg.norm((xx, yy))
            potential.append(self.jacobiPotential(rho_i, prs_i, vx3_i, r)[0])
        return potential

    def computeStreamline(self, point, x, y, vx1, vx2, vx3, rho, prs, x_range, y_range):
        # print(Tools.singlePointInterpolation(0, p0, vx1, vx2, x_range, y_range))
        # vx, vy = np.ravel(self.data.variables["vx1"]), np.ravel(self.data.variables["vx2"])

        p0 = point
        t0 = 0.0
        t1 = 1000
        # print(Tools.singlePointInterpolation(t0, p0, vx1, vx2, x_range, y_range))
        solver = scipy.integrate.ode(Interpolate.singlePointInterpolation)
        solver.set_integrator("vode", rtol=1e-10)
        solver.set_f_params(vx1, vx2, x_range, y_range)
        solver.set_initial_value(p0, t0)

        # mimics the wind launching front
        H = 4.75 * self.pressureScaleHeightFlat()
        xticks = self.data.x1

        x, y = [], []

        while solver.y[1] > Interpolate.interpolatePoint(xticks, H, solver.y[0]):
            solver.integrate(t1, step=True)
            x.append(solver.y[0])
            y.append(solver.y[1])
            print(solver.y)
        #print(solver.y)
        #print("Computing Jacobi potential...")
        #potential = self.computeJacobiPotential(x, y, vx3, rho, prs, x_range, y_range)
        potential = 0
        return solver.y[0], potential

    def computeRadialMassLosses(self):
        computeLimit = int(len(self.data.dx1) * 0.99)
        rho = self.data.variables["rho"][:,computeLimit] * self.data.unitDensity
        vx1 = self.data.variables["vx1"][:,computeLimit] * self.data.unitVelocity
        temp = self.computeTemperature()[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000 and vx1[i] > 0]
        tempRange = range(min(tempRange), max(tempRange))
        r = self.data.x1[computeLimit]
        theta = self.data.x2[tempRange]

        surface = 0.5*np.pi / len(self.data.x2) * r**2 * 2.0 * np.pi * self.data.unitLength**2
        losses = 2 * surface * rho[tempRange] * vx1[tempRange] * self.data.year / self.data.solarMass
        x_start = r * np.sin(theta)
        y_start = r * np.cos(theta)

        trans = Transform(self.data)

        x, y = trans.polarCoordsToCartesian()
        vx1, vx2 = trans.transformVelocityFieldToCylindrical()
        x_range = [1, 100, 1000]
        y_range = [0, 100, 1000]
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, x_range, y_range)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, x_range, y_range)
        x, y, vx3 = Interpolate.interpolateToUniformGrid(self.data, self.data.variables["vx3"], x_range, y_range)
        x, y, rho = Interpolate.interpolateToUniformGrid(self.data, self.data.variables["rho"], x_range, y_range)
        x, y, prs = Interpolate.interpolateToUniformGrid(self.data, self.data.variables["prs"], x_range, y_range)


        vx1 = -vx1
        vx2 = -vx2

        losses = losses[150:151]
        x_start = x_start[150:151]
        y_start = y_start[150:151]

        radii = []
        potentials = []

        for i, j in zip(x_start, y_start):
            radius, potential = self.computeStreamline((i, j), x, y, vx1, vx2, vx3, rho, prs, x_range, y_range)
            radii.append(radius)
            potentials.append(potential)

        return radii, losses, potentials

    def computeMassLoss(self, data):
        computeLimit = int(len(data.dx1) * 0.99)
        rho = data.variables["rho"][:,computeLimit] * data.unitDensity
        vx1 = data.variables["vx1"][:,computeLimit] * data.unitVelocity
        temp = self.computeTemperature()[:,computeLimit]
        tempRange = [i for i,v in enumerate(temp) if v > 1000]
        tempRange = range(min(tempRange), max(tempRange))
        surface = 0.5*np.pi / len(data.x2) * data.x1[computeLimit]**2 * 2.0 * np.pi * data.unitLength**2
        massLoss = surface * rho[tempRange] * vx1[tempRange]
        totalMassLoss = np.sum(massLoss)
        return totalMassLoss * data.year / data.solarMass

    def computeMassLosses(self, path):
        losses = []
        times = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                data = Data(os.path.join(path, file))
                loss = self.computeMassLoss(data)
                times.append(float(data.time) * data.unitTimeYears)
                losses.append(loss)
                print("Massflux for " + file + ": " + str(loss))
        return losses, times

    def plotMassLosses(self, path, filename="losses.eps"):
        losses, times = self.computeMassLosses(path)
        losses = np.array(losses, dtype=np.double)
        times = np.array(times, dtype=np.double)

        key = times.argsort()
        times = times[key]
        losses = losses[key]
        np.savetxt("losses.dat", (times, losses))

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogy(times, losses)
        plt.xlabel(r't [yr]')
        plt.ylabel(r'$\dot{M}_w $ [$\frac{M_{\odot}}{\mathrm{yr}}$]')
        plt.ylim(1e-8, 4e-7)
        plt.savefig(filename)

    def averageFrames(self, path, variable, frameRange):
        frames = []

        for filename in os.listdir(path):
            if filename.endswith(".h5"):
                frameIndex = int(filename.split('.')[1])
                if frameIndex in frameRange:
                    print("Averaging " + str(variable) + " frame: " + str(frameIndex))
                    data = Data(os.path.join(path, filename))
                    frames.append(data.variables[variable])
        frames = np.array(frames)
        averaged = np.mean(frames, axis=0)
        return averaged

    def pressureScaleHeight(self):
        temp = self.computeTemperature()
        cs = np.sqrt(self.data.kb * temp / (self.data.mu * self.data.mp)) / self.data.unitVelocity
        trans = Transform(self.data)
        x, y = trans.polarCoordsToCartesian()
        omega = np.sqrt(1.0 / x**3)
        H = cs / omega
        return H[-1]

    def pressureScaleHeightFlat(self):
        temp = self.computeTemperature()
        cs = np.sqrt(self.data.kb * temp / (self.data.mu * self.data.mp)) / self.data.unitVelocity
        trans = Transform(self.data)
        x, y = trans.polarCoordsToCartesian()
        omega = np.sqrt(1.0 / x**3)
        H = cs / omega
        return np.power(H[-1], 4.33/5.0)



class Transform:

    def __init__(self, data):
        self.data = data

    def transformVelocityFieldToCylindrical(self):
        vx1 = self.data.variables["vx1"]
        vx2 = self.data.variables["vx2"]
        x2 = np.transpose(np.tile(self.data.x2, (len(self.data.x1), 1)))

        vx1_t = vx1 * np.sin(x2) + vx2 * np.cos(x2)
        vx2_t = vx1 * np.cos(x2) - vx2 * np.sin(x2)
        return vx1_t, vx2_t

    def transformMagneticFieldToCylindrical(self):
        bx1 = self.data.variables["bx1"]
        bx2 = self.data.variables["bx2"]
        x2 = np.transpose(np.tile(self.data.x2, (len(self.data.x1), 1)))

        bx1_t = bx1 * np.sin(x2) + bx2 * np.cos(x2)
        bx2_t = bx1 * np.cos(x2) - bx2 * np.sin(x2)
        return bx1_t, bx2_t

    def polarCoordsToCartesian(self):
        r_matrix, th_matrix = np.meshgrid(self.data.x1, self.data.x2)
        x = r_matrix * np.sin(th_matrix)
        y = r_matrix * np.cos(th_matrix)
        return x, y


class Interpolate:

    def __init__(self):
    	pass

    @staticmethod
    def interpolatePoint(ticks, data, point):
        f = scipy.interpolate.interp1d(ticks, data)
        return f(point)

    # Returns single interpolated value on a regular grid (faster than griddata)
    @staticmethod
    def singlePointInterpolation(t, p, vx1, vx2, x_range, y_range):
        pp = [[(p[1] - y_range[0]) * y_range[2] / (y_range[1] - y_range[0])],
                      [(p[0] - x_range[0]) * x_range[2] / (x_range[1] - x_range[0])]]
        pp = np.array(pp)
        return [map_coordinates(vx1, pp, order=1), map_coordinates(vx2, pp, order=1)]

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
        return data

    @staticmethod
    def interpolateToUniformGrid(data, variable, x_range, y_range):
        t = Transform(data)
        x, y = t.polarCoordsToCartesian()
        x = np.ravel(x)
        y = np.ravel(y)
        variable = np.ravel(variable)
        points = np.column_stack((x, y))
        grid_x, grid_y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))
        newVariable = scipy.interpolate.griddata(points, variable, (grid_x, grid_y))
        grid_r = np.sqrt(grid_x**2 + grid_y**2)
        newVariable[grid_r < 1.0] = np.nan
        return grid_x, grid_y, newVariable


class Tools:

    def __init__(self):
        pass

    @staticmethod
    def removeFilesWithStride(path, stride):
        for current_file in os.listdir(path):
            if current_file.endswith(".h5") or current_file.endswith(".xmf"):
                frame = int(current_file.split('.')[1])
                if frame % stride != 0:
                    print("deleting frame " + str(frame))
                    os.remove(os.path.join(path, current_file))
