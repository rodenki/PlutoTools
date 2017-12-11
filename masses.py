import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse



data = sim.SimulationData()
masses, t = Tools.computeTotalMasses("./nofloor/")
print(masses)
