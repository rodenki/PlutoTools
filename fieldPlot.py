import simulationData as sim
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


masses = Tools.computeTotalMasses("./")
print(masses[0])
