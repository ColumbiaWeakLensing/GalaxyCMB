#!/usr/bin/env python
import os

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sns

import lenstools as lt
from lenstools.simulations.nicaea import NicaeaSettings

#Options
z_cross = (1.0,1.5,2.0)
z_cmb = 38.
thetaG = 0.1*u.arcmin

settings_lin = NicaeaSettings()
settings_lin["snonlinear"] = "linear"
settings_nl = NicaeaSettings()
settings_nl["snonlinear"] = "smith03_revised"

#Book keeping
batch = lt.SimulationBatch.current()
cosmo = batch["m0"].cosmology
mpset = batch["m0c0"].getMapSet("kappaCMB")

#Set up plot
fig,ax = plt.subplots(1,2,figsize=(16,8))
ell = np.load(os.path.join(batch.home,"ell_nb100.npy"))
smooth = np.exp(-(ell*thetaG.to(u.rad).value)**2)

#Cycle over redshifts
for zc in z_cross:

	#Load measured cross power
	pc = np.load(os.path.join(mpset.home,"cross_power_z{0}_s0_nb100.npy".format(int(zc*100))))

	#Compute linear cross power with NICAEA
	pc_nicaea_lin = cosmo.convergencePowerSpectrum(ell,z=None,distribution=("single","single"),distribution_parameters=((zc,zc),(z_cmb,z_cmb)),settings=settings_lin)
	pc_nicaea_nl = cosmo.convergencePowerSpectrum(ell,z=None,distribution=("single","single"),distribution_parameters=((zc,zc),(z_cmb,z_cmb)),settings=settings_nl)

	#Plot
	line = ax[0].plot(ell,ell*(ell+1)*pc.mean(0)/(2.*np.pi),label=r"$z_g={0:.2f}$".format(zc))
	ax[0].plot(ell,ell*(ell+1)*smooth*pc_nicaea_lin[:,1]/(2.*np.pi),linestyle="--",color=line[0].get_color(),label=r"${\rm NICAEA(lin)}$")
	ax[0].plot(ell,ell*(ell+1)*smooth*pc_nicaea_nl[:,1]/(2.*np.pi),linestyle="-.",color=line[0].get_color(),label=r"${\rm NICAEA(nl)}$")

	#Born vs raytracing
	pb = np.load(os.path.join(batch["m0c0"].getMapSet("kappaCMBBorn").home,"cross_power_z{0}_s0_nb100.npy".format(int(zc*100))))
	ax[1].plot(ell,1.-pb.mean(0)/pc.mean(0),label=r"$z_g={0:.2f}$".format(zc),color=line[0].get_color())


#Labels
for n in (0,1):
	ax[n].set_xscale("log")
	ax[n].set_xlabel(r"$\ell$",fontsize=18)

ax[0].set_ylabel(r"$\ell(\ell+1)P_{z_g,z_{\rm CMB}}(\ell)/2\pi$",fontsize=18)
ax[1].set_ylabel(r"$1-P_{z_g,z_{\rm CMB}}^{\rm born}/P_{z_g,z_{\rm CMB}}^{\rm ray}$",fontsize=18)

ax[0].legend(loc="upper left")
ax[1].legend(loc="upper left")

#Save
fig.tight_layout()
fig.savefig("cross_power.png")