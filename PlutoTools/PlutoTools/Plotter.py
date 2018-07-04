import numpy as np
import yt
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

    def plotVariable(self, variable, cmap=cm.inferno):
        #self.data.x2 = self.data.x2[:450]
        t = Transform(self.data)
        x, y = t.polarCoordsToCartesian()
        plt.figure(figsize=self.figsize)

        if self.interpolate:
            x, y, variable = Interpolate.interpolateToUniformGrid(self.data, variable, self.xrange, self.yrange)

        if self.log == True:
            if len(self.vlimits) > 0:
                plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=self.vlimits[0], vmax=self.vlimits[1]), cmap=cmap)
            else:
                plt.pcolormesh(x, y, variable, norm=LogNorm(vmin=np.nanmin(variable), vmax=np.nanmax(variable)), cmap=cmap)
            cb = plt.colorbar()
            tick_locator = ticker.LogLocator(numdecs=10)
            cb.locator = tick_locator
            cb.update_ticks()
        else:
            if len(self.vlimits) > 0:
                plt.pcolormesh(x, y, variable, vmin=self.vlimits[0], vmax=self.vlimits[1], cmap=cmap)
            else:
                plt.pcolormesh(x, y, variable, cmap=cmap)
            cb = plt.colorbar()

        plt.xlabel('Radius [AU]')
        plt.ylabel('z [AU]')
        plt.xlim(*self.xrange)
        plt.ylim(*self.yrange)
        orbits = self.data.orbits(self.orbitalDistance, self.data.time)
        plt.title(self.title + " t = " + str(self.data.time) + ", " + str(int(orbits)) + " orbits")
        return plt

    def plotLineData(self, x, y):
        newTicks = np.linspace(np.min(x), np.max(x), self.xrange[2])
        f = scipy.interpolate.InterpolatedUnivariateSpline(x, y, k=1)
        interpolated = f(newTicks)
        plt.plot(newTicks, interpolated, 'w', linewidth=self.width)
        return plt

    def plotVelocityFieldLines(self, density=3, variable=None, cmap=cm.inferno):
        self.interpolate = True
        if variable is not None:
            self.plotVariable(variable, cmap=cmap)
        else:
            self.plotVariable(self.data.variables["rho"] * self.data.unitNumberDensity, cmap=cmap)

        t = Transform(self.data)
        vx1, vx2 = t.transformVelocityFieldToCylindrical()
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, self.xrange, self.yrange)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, self.xrange, self.yrange)

        n = np.sqrt(vx1**2 + vx2**2)
        vx1 /= n
        vx2 /= n

        plt.streamplot(x, y, vx1, vx2, density=density, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)
        return plt

    def plotVelocityLIC(self, variable=None, cmap=cm.inferno):
        self.interpolate = True
        # if variable is not None:
        #     self.plotVariable(variable, cmap=cmap)
        # else:
        #     self.plotVariable(self.data.variables["rho"] * self.data.unitNumberDensity, cmap=cmap)

        t = Transform(self.data)
        vx1, vx2 = t.transformVelocityFieldToCylindrical()
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, self.xrange, self.yrange)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, self.xrange, self.yrange)
        vx = np.transpose(np.sqrt(vx1**2 + vx2**2))
        vx1 = np.transpose(vx1)
        vx2 = np.transpose(vx2)

        data = dict()
        data["velocity_x"] = vx1[..., None]
        data["velocity_y"] = vx2[..., None]
        data["velocity_abs"] = vx[..., None]
        bbox = np.array([[np.min(x), np.max(x)],
                         [np.min(y), np.max(y)],
                         [0.0, 1.0]])
        ds = yt.load_uniform_grid(data, data["velocity_x"].shape, bbox=bbox, nprocs=4, length_unit=(1.0,"AU"))
        s = yt.SlicePlot(ds, 'z', 'velocity_abs', origin='left-window')
        s.set_buff_size((2000, 2000))
        s.set_cmap('all', cm.inferno)
        s.annotate_line_integral_convolution('velocity_x', 'velocity_y', lim=(0.46,0.54), cmap=cm.coolwarm, kernellen=500, alpha=1.0, const_alpha=True)
        s.save("test1", mpl_kwargs={'dpi': 250})

    def plotMagneticFieldLIC(self, variable=None, cmap=cm.inferno):
        self.interpolate = True
        # if variable is not None:
        #     self.plotVariable(variable, cmap=cmap)
        # else:
        #     self.plotVariable(self.data.variables["rho"] * self.data.unitNumberDensity, cmap=cmap)

        t = Transform(self.data)
        vx1, vx2 = t.transformMagneticFieldToCylindrical()
        x, y, vx1 = Interpolate.interpolateToUniformGrid(self.data, vx1, self.xrange, self.yrange)
        x, y, vx2 = Interpolate.interpolateToUniformGrid(self.data, vx2, self.xrange, self.yrange)
        vx = np.transpose(np.sqrt(vx1**2 + vx2**2))
        vx1 = np.transpose(vx1)
        vx2 = np.transpose(vx2)

        data = dict()
        data["magnetic_field_x"] = vx1[..., None]
        data["magnetic_field_y"] = vx2[..., None]
        data["magnetic_field_strength"] = vx[..., None]
        bbox = np.array([[np.min(x), np.max(x)],
                         [np.min(y), np.max(y)],
                         [0.0, 1.0]])
        ds = yt.load_uniform_grid(data, data["magnetic_field_x"].shape, bbox=bbox, nprocs=4, length_unit=(1.0,"AU"))
        s = yt.SlicePlot(ds, 'z', 'magnetic_field_strength', origin='left-window')
        s.set_buff_size((2000, 2000))
        s.set_cmap('all', cm.inferno)
        s.annotate_line_integral_convolution('magnetic_field_x', 'magnetic_field_y', lim=(0.46,0.54), cmap=cm.Greys, alpha=0.3, const_alpha=True)
        s.save("test1", mpl_kwargs={'dpi': 250})



    def plotMagneticFieldLines(self, density=3, variable=None, cmap=cm.inferno):
        self.interpolate = True
        if variable is not None:
            self.plotVariable(variable, cmap=cmap)
        else:
            self.plotVariable(self.data.variables["rho"] * self.data.unitNumberDensity, cmap=cmap)

        t = Transform(self.data)
        bx1, bx2 = t.transformMagneticFieldToCylindrical()
        x, y, bx1 = Interpolate.interpolateToUniformGrid(self.data, bx1, self.xrange, self.yrange)
        x, y, bx2 = Interpolate.interpolateToUniformGrid(self.data, bx2, self.xrange, self.yrange)

        n = np.sqrt(bx1**2 + bx2**2)
        bx1 /= n
        bx2 /= n

        plt.streamplot(x, y, bx1, bx2, density=density, arrowstyle='->', linewidth=1,
                       arrowsize=1.5)
        return plt

    def plotMagneticField(self, scale=40, width=0.001):

        self.interpolate = True
        self.plotVariable(self.data.variables["rho"] * self.data.unitNumberDensity)

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
        return plt
