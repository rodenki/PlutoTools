import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, warnings
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
np.set_printoptions(threshold=500)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        self.linewidth = 1.0
        self.title = ""

    def setXrange(self, start, stop, n):
        self.xrange = [start, stop, n]

    def setYrange(self, start, stop, n):
        self.yrange = [start, stop, n]

    def setVlimits(self, start, stop):
        self.vlimits = (start, stop)

    def setFigsize(self, width, height):
        self.figsize = (width, height)

    def setInterpolation(self, inter):
    	self.interpolate = inter

    def setLogscale(self, log):
        self.log = log

    def setLimits(self, limits):
        self.vlimits = limits

    def setFontSize(self, size):
        rcParams.update({'font.size': size})

    def setTitle(self, title):
        self.title = title

    def setOrbitalDistance(self, distance):
        self.orbitalDistance = distance

    def setLineWidth(self, width):
        self.width = width

    def show(self):
        plt.show()

    def savefig(self, filename=None):
        if filename == None:
            plt.savefig(self.filename + ".png", dpi=300)
        else:
            plt.savefig(filename, dpi=300)

    def clear(self):
        plt.cla()
        plt.close()

    def plotVariable(self, variable):
        #self.data.x2 = self.data.x2[:450]
        t = Transform(self.data)
        x, y = t.polarCoordsToCartesian()
        plt.figure(figsize=self.figsize)

        if self.interpolate:
            x, y, variable = Interpolate.interpolateToUniformGrid(self.data, variable, self.xrange, self.yrange)

        if self.log:
            if len(self.vlimits) > 0:
                plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=self.vlimits[0], vmax=self.vlimits[1]), cmap=cm.inferno)
            else:
                plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=np.nanmin(variable), vmax=np.nanmax(variable)), cmap=cm.inferno)
        else:
            if len(self.vlimits) > 0:
                plt.pcolormesh(x, y, variable, vmin=self.vlimits[0], vmax=self.vlimits[1], cmap=cm.inferno)
            else:
                plt.pcolormesh(x, y, variable, cmap=cm.inferno)

        cb = plt.colorbar()
        tick_locator = ticker.LogLocator(numdecs=10)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.xlabel('Radius [AU]')
        plt.ylabel('z [AU]')
        plt.xlim(*self.xrange)
        plt.ylim(*self.yrange)
        orbits = self.data.orbits(self.orbitalDistance, self.data.time)
        plt.title(self.title + " t = " + str(self.data.time) + ", " + str(int(orbits)) + " orbits")

    def plotLineData(self, x, y):
        newTicks = np.linspace(np.min(x), np.max(x), self.xrange[2])
        f = scipy.interpolate.InterpolatedUnivariateSpline(x, y, k=1)
        interpolated = f(newTicks)
        plt.plot(newTicks, interpolated, 'w', linewidth=self.width)

    def plotVelocityFieldLines(self, density=3, variable=None):
        self.interpolate = True
        if variable is not None:
            self.plotVariable(variable)
        else:
            self.plotVariable(self.data.variables["rho"])

        t = Transform(self.data)
        vx1, vx2 = t.transformVelocityFieldToCylindrical()
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, self.xrange, self.yrange)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, self.xrange, self.yrange)

        n = np.sqrt(vx1**2 + vx2**2)
        vx1 /= n
        vx2 /= n

        plt.streamplot(x, y, vx1, vx2, density=density, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

    def plotMagneticFieldLines(self, density=3, variable=None):
        self.interpolate = True
        if variable:
            self.plotVariable(variable)
        else:
            self.plotVariable(self.data.variables["rho"])

        t = Transform(self.data)
        bx1, bx2 = t.transformMagneticFieldToCylindrical()
        x, y, bx1 = Interpolate.interpolateToUniformGrid(self.data, bx1, self.xrange, self.yrange)
        x, y, bx2 = Interpolate.interpolateToUniformGrid(self.data, bx2, self.xrange, self.yrange)

        n = np.sqrt(bx1**2 + bx2**2)
        bx1 /= n
        bx2 /= n

        plt.streamplot(x, y, bx1, bx2, density=density, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)

    def plotMagneticField(self, scale=40, width=0.001):

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
