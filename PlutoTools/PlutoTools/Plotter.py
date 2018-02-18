import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
np.set_printoptions(threshold=500)

from .Data import Data
from .Tools import Transform
from .Tools import Interpolate


class Plotter:

    def __init__(self, data):
        self.data = data
        self.resetValues()

    def resetValues(self):
        self.xrange = [0, 99, 500]
        self.yrange = [0, 99, 500]
        self.vlimits = []
        self.figsize = (9, 6)
        self.filename = "data"
        self.log = True
        self.interpolate = False
        self.orbitalDistance = 1.0

    def setXrange(self, start, stop, n):
        self.xrange = [start, stop, n] 

    def setYrange(self, start, stop, n):
        self.yrange = [start, stop, n]

    def setVlimits(self, start, stop):
        self.vlimits = (start, stop)

    def setFigsize(self, width, height):
        self.figsize = (width, height)

    def show(self):
        plt.show()

    def savefig(self, filename=None):
        if filename == None:
            plt.savefig(self.filename + ".png")
        else:
            plt.savefig(filename)            

    def clear(self):
        plt.cla()
        plt.close()

    def plotVariable(self, variable):
        t = Transform(self.data)
        x, y = t.polarCoordsToCartesian()
        plt.figure(figsize=self.figsize)

        if self.interpolate:
            x, y, variable = Interpolate.interpolateToUniformGrid(self.data, variable, self.xrange, self.yrange)

        if self.log:
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
        orbits = self.data.orbits(self.orbitalDistance, self.data.time)
        plt.title("t = " + str(self.data.time) + ", " + str(int(orbits)) + " orbits")

    def plotLineData(self, lineData):
        newTicks = np.linspace(*self.xrange)
        f = scipy.interpolate.interp1d(self.data.x1, lineData)
        interpolated = f(newTicks)

        plt.plot(newTicks, interpolated, 'g')

    def plotVelocityFieldLines(self):

        self.interpolate = True
        self.plotVariable(self.data.variables["rho"])

        t = Transform(self.data)
        vx1, vx2 = t.transformVelocityFieldToCylindrical()
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, self.xrange, self.yrange)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, self.xrange, self.yrange)

        n = np.sqrt(vx1**2 + vx2**2)
        vx1 /= n
        vx2 /= n

        plt.streamplot(x, y, vx1, vx2, density=3, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

    def plotMagneticFieldLines(self):

        self.interpolate = True
        self.plotVariable(self.data.variables["rho"])

        t = Transform(self.data)
        bx1, bx2 = t.transformMagneticFieldToCylindrical()
        x, y, bx1 = Interpolate.interpolateToUniformGrid(self.data, bx1, self.xrange, self.yrange)
        x, y, bx2 = Interpolate.interpolateToUniformGrid(self.data, bx2, self.xrange, self.yrange)

        n = np.sqrt(bx1**2 + bx2**2)
        bx1 /= n
        bx2 /= n

        plt.streamplot(x, y, bx1, bx2, density=3, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

    def plotMagneticField(self, dx1=10, dx2=5, scale=40, width=0.001, x1_start=0):

        self.interpolate = True
        self.plotVariable(self.data.variables["rho"])

        t = Transform(self.data)
        bx1, bx2 = t.transformMagneticFieldToCylindrical()
        x, y = t.polarCoordsToCartesian()
        bx1 = self.data.variables["bx1"]
        bx2 = self.data.variables["bx2"]

        x, y, bx1 = Interpolate.interpolateToUniformGrid(self.data, bx1, self.xrange, self.yrange)
        x, y, bx2 = Interpolate.interpolateToUniformGrid(self.data, bx2, self.xrange, self.yrange)

        n = np.sqrt(bx1**2 + bx2**2)
        bx1 /= n
        bx2 /= n

        plt.quiver(x, y, bx1, bx2, width=width, scale=scale, color='k')


