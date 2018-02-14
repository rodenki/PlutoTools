import h5py
import numpy as np

import xml.etree.cElementTree as xml
from copy import deepcopy

np.set_printoptions(threshold=500)


class Data:
    def __init__(self):
        self.filename = ""
        self.unitDensity = 5.974e-07
        self.unitNumberDensity = 3.572e+17
        self.unitPressure = 5.329e+06
        self.unitVelocity = 2.987e+06
        self.unitLength = 1.496e+13
        self.unitTimeYears = 1.588e-01
        self.solarMass = 1.989e33
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
            #print("Magnetic field not present.")
            pass

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