from simulationData import SimulationData
from simulationData import Tools
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def computeStreamlines():
	data = SimulationData()
	data.loadGridData()

	data.variables["rho"] = np.load("avg_rho.npy")
	data.variables["prs"] = np.load("avg_prs.npy")
	data.variables["vx1"] = np.load("avg_vx1.npy")
	data.variables["vx2"] = np.load("avg_vx2.npy")
	data.variables["vx3"] = np.load("avg_vx3.npy")
	radii, losses, potentials = Tools.computeRadialMassLosses(data)

	np.save("radii.npy", radii)
	np.save("losses.npy", losses)
	np.save("jacobi.npy", potentials)

def plotJacobiePotential():
	jacobi = np.load("jacobi.npy")
	print(jacobi)

plotJacobiePotential()

def computeLosses():
	data = SimulationData()
	data.loadGridData()
	radii = np.load("radii.npy")
	losses = np.load("losses.npy")
	losses *= 2
	cumulativeLosses = []

	for i, loss in enumerate(losses):
		cumulativeLosses.append(np.sum(losses[:i]))

	dA = []
	for i in range(len(radii)):
		if i == 0:
			dr = 0.5 * (radii[i+1] - radii[i])
			a = np.pi * (np.power(radii[i]+dr, 2) - np.power(radii[i]-dr, 2))
			dA.append(a)
		elif i == len(radii)-1:
			dr = 0.5 * (radii[i] - radii[i-1])
			a = np.pi * (np.power(radii[i]+dr, 2) - np.power(radii[i]-dr, 2))
			dA.append(a)
		else:
			drp = 0.5*(radii[i+1] - radii[i])
			drm = 0.5*(radii[i] - radii[i-1])
			a = np.pi * (np.power(radii[i]+drp, 2) - np.power(radii[i]-drm, 2))
			dA.append(a)

	dA = np.array(dA)

	losses *= data.solarMass / (365 * 86400)
	dA *= data.unitLength**2
	losses /= dA


	f = interpolate.interp1d(radii, losses, kind='cubic')
	r = np.linspace(radii[0], radii[-1], 1000)
	interLosses = f(r)
	data = np.vstack((r, interLosses))
	np.savetxt("surfaceLoss.dat", data)

	plt.plot(r, interLosses, linewidth=1.0)
	#ax = plt.axes()
	#ax.set_ticklabel_format("sci")
	plt.xlabel("r (AU)")
	plt.ylabel("$ \dot{\Sigma}_w [g / (s cm^2)]$")
	plt.savefig("surfaceDensityLoss.eps")


	plt.clf()
	plt.plot(radii, cumulativeLosses, linewidth=1)
	plt.xlabel("r (AU)")
	plt.ylabel("$ \dot{M}_w  [M_{\odot} / yr] $(Cumulative)")
	plt.savefig("cumulativeLoss.eps")
