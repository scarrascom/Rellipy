#!/home/seismo/venvs/py3seismo/bin/python
# -*- coding: utf-8 -*-
"""

@author  : Sebastian Carrasco
@e-mail  : acarrasc@uni-koeln.de
@purpose : Derive a final ellipticity curve for a specific station.
           It must be run inside a directory containing events subdirectories
"""

import sys
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime

from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.signal import savgol_filter

#station = 'DREG'
#fname = f'Full_catalogue_{station}.txt'

fname = sys.argv[1]
station = fname.split('.')[0].split('_')[-1]
fields = ['ot', 'lat', 'lon', 'epidist', 'baz', 'depth', 'mag']
data = pd.read_csv(fname, header=None, names=fields)
data['evname'] = [ ''.join(ot.split('T')[0].split('-')) + '_' + ''.join(ot.split('T')[1].split(':')) for ot in data.ot ]

thresz = 10         # Impose a threshold on the vertical component
thresr = 5          # Impose a threshold on the horizontal component

periods = np.arange(8, 161, 1)
logfmt = FuncFormatter(lambda y, _: '{:g}'.format(y))

alfa = 0.5
coledge = 'black'
## Set colors and markers, in case we can identify points belonging to the same earthquake
main_markers = ['o', 'v', 's', '*', 'D', 'P']           # Various markers
ncolors = int(len(data.evname)/len(main_markers))+1     # Various colours    
main_cmap = plt.get_cmap('rainbow')                     # Color map
dis_colors = [ main_cmap(nc/ncolors) for nc in range(ncolors)]
pairs = [ (mark, col) for mark in main_markers for col in dis_colors]

saiz = 40
msaiz = 6
edge = 0.3
dict_err = dict(alpha=alfa, markeredgecolor=coledge,
                markeredgewidth=edge, markersize=msaiz)
dict_scat = dict(alpha=alfa, s=saiz, edgecolors=coledge, linewidth=edge)
dict_noerr = dict(color='lightgray', alpha=alfa, markeredgecolor='lightgray',
                  markeredgewidth=edge, markersize=msaiz)
dict_no = dict(s=saiz, color='lightgray', alpha=alfa)

minper, maxper = 8, 110
figsum, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6.5), sharex=True)
axsum, axphsh = axs

axphsh.axhline(y=90, xmin=minper, xmax=maxper, color='red', linewidth=2)

dict_ellip = { evn: { t: [] for t in periods } for evn in data.evname}
fin_ellip = { t: [] for t in periods  }

sel_evts = []
phase_shifts = []
phase_shifts_min = []
nphases = []
for e, evnm in enumerate(data.evname):
    print(evnm)
    with open(f'{evnm}/{evnm}.pkl', 'rb') as fl:
        datad = pickle.load(fl)
    aux_ph = []
    for p, period in enumerate(datad.keys()):
        if not datad[period]['hv']:
            continue
        thishv = datad[period]['hv'][0]
        minhv, medhv, maxhv = thishv
        phsh = datad[period]['phsh']
        snrz = datad[period]['SNRZ']
        snrr = datad[period]['SNRR']
        this_marker = pairs[e][0]
        this_color = pairs[e][1]
        ## Enough energy and positive phase shifts
        if (snrz>=thresz and snrr>thresr and phsh>0):
            sel_evts.append(evnm)
            dict_ellip[evnm][period].append(thishv)
            fin_ellip[period].append(thishv)
            axsum.scatter(x=period, y=medhv, marker=this_marker, color=this_color, 
                           **dict_scat)
            axphsh.scatter(x=period, y=phsh, marker=this_marker, color=this_color, 
                           **dict_scat)
            aux_ph.append(phsh)
    phase_shifts.append(np.nanmedian(aux_ph))
    phase_shifts_min.append(np.nanpercentile(aux_ph, 15))
    nphases.append(len(aux_ph))

data['mphase'] = phase_shifts
data['mphase_15p'] = phase_shifts_min
data['mphase_n'] = nphases

print('Creating phase-shift plots!')
labs = ['Median', '15p']

tini = UTCDateTime(data.ot.iloc[-1])
tend = UTCDateTime(data.ot.iloc[0])

fullt = tend-tini
elapsedt = np.asarray([ UTCDateTime(day) - tini for day in data.ot ])
cmap_time = plt.get_cmap('viridis')
colorst = cmap_time(elapsedt/fullt)

for d, dphase in enumerate([data.mphase, data.mphase_15p]):
    figph, axsph = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharey=True)
    axbaz, axdis = axsph[0]
    axdep, axtim = axsph[1]
        
    dates = [ pd.to_datetime(dd) for dd in data.ot ]
    
    alfap = 0.6
    dictsc = dict(c=colorst, alpha=alfap, s=data.mphase_n*2+30)
    axbaz.scatter(data.baz, dphase, **dictsc)
    axbaz.set_xlabel('Back azimuth [$^o$]')
    axbaz.set_ylabel('Phase shift $\phi$ [$^o$]')
    
    axdis.scatter(data.epidist, dphase, **dictsc)
    axdis.set_xlabel('Epi dist [km]')
    
    axdep.scatter(data.depth, dphase, **dictsc)
    axdep.set_xlabel('Focal depth [km]')
    axdep.set_ylabel('Phase shift $\phi$ [$^o$]')
    
    axtim.scatter(dates, dphase, **dictsc)
    axtim.set_xlabel('Origin time')
    
    for axx in axsph.flatten():
        axx.grid(ls='--', alpha=0.5)
        axx.set_ylim(0, 180)
       
    phname = f'PhaseShifts{labs[d]}_{station}'
    figph.suptitle(phname)
    figph.tight_layout()
    figph.savefig(f'{phname}.png', dpi=300)
    figph.savefig(f'{phname}.pdf')

## Export the data
data.to_csv(f'PhaseShifts_{station}.csv', index=False)

############# Get smooth median ellipticity ############
events = np.unique(sel_evts)
fellip_full = { }
aux_fin_ellip = {k: val for k, val in fin_ellip.items() if val}
fin_ellip = aux_fin_ellip
new_periods = np.array(list(fin_ellip.keys()))

for k, cP in enumerate(new_periods):
    before = k-1
    forward = k+1
    if before < 0:
        before = 0
    if forward > len(new_periods)-1:
        forward = len(new_periods)-1
    minT = new_periods[before]
    maxT = new_periods[forward]
    bool_vals = np.logical_and(new_periods>=minT, new_periods<=maxT)
    fellip_full[cP] = []
    for pp in new_periods[bool_vals]:
        for els in fin_ellip[pp]:
            fellip_full[cP].append(els[1])

print('Getting some statistics...')
minHV, medHV, maxHV = [], [], []
for p, ell in fellip_full.items():
    if len(ell)>0:
        minHV.append(np.nanpercentile(ell, 16))
        medHV.append(np.nanpercentile(ell, 50))
        maxHV.append(np.nanpercentile(ell, 84))
    else:
        minHV.append(np.nan)
        medHV.append(np.nan)
        maxHV.append(np.nan)

HVmin = np.asarray(minHV)
HVmed = np.asarray(medHV)
HVmax = np.asarray(maxHV)

ns = 15
print('Smoothing...')
interp_ave = savgol_filter(HVmed, ns, 3)
interp_min = savgol_filter(HVmin, ns, 3)
interp_max = savgol_filter(HVmax, ns, 3)

std_n = np.array(interp_ave)/np.array(interp_min)
std_x = np.array(interp_max)/np.array(interp_ave)
fin_std = np.average([std_x, std_n], axis=0)        # Average of standard deviation error

fin_minstd = interp_ave/fin_std
fin_maxstd = interp_ave*fin_std

axsum.plot(new_periods, interp_ave, lw=2, ls='-',
           color='k', zorder=1)
axsum.fill_between(new_periods, fin_minstd, fin_maxstd, 
                   color='gray', zorder=1, alpha=0.4)
axsum.legend(loc='lower right')

print('Summary csv file...')
np.savetxt(f'FinSmoothEllipticity_{station}.txt', 
           np.transpose([new_periods, fin_minstd, interp_ave, fin_maxstd]), 
           fmt='%.1f %.5f %.5f %.5f')

xlims = (minper, maxper)
axsum.grid(True, which='both', ls='--', color='gray', alpha=0.6)
axsum.set_ylabel('$\epsilon$', fontsize=11)
axsum.set_ylim(-0.5, 0.5)
axsum.set_xlim(xlims)
axsum.set_xscale('log')

axsum.set_ylim(0.3, 4)
axsum.set_yscale('log')
axsum.grid(which='major', ls='--', color='gray', alpha=0.8)
axsum.grid(which='minor', ls='--', color='gray', alpha=0.4)
axsum.yaxis.set_major_formatter(logfmt)
axsum.yaxis.set_minor_formatter(logfmt)

axphsh.grid(which='major', ls='--', color='gray', alpha=0.8)
axphsh.grid(which='minor', ls='--', color='gray', alpha=0.4)
axphsh.set_ylabel('Phase shift [$\phi$]', fontsize=11)
axphsh.set_xlabel('Period [s]', fontsize=11)
axphsh.yaxis.set_major_locator(MultipleLocator(30))
axphsh.set_ylim(0, 180)
axphsh.set_xlim(xlims)

axphsh.xaxis.set_major_formatter(logfmt)
axphsh.xaxis.set_minor_formatter(logfmt)

axphsh.yaxis.set_minor_locator(MultipleLocator(10))

figsum.subplots_adjust(top=0.95, bottom=0.08, left=0.117,
                       right=0.98, hspace=0.1, wspace=0.2)

print('Saving summary figure')
fnamesum = f'Summary_Ellip_phase_{station}'
figsum.savefig(f'{fnamesum}.png', dpi=300)
figsum.savefig(f'{fnamesum}_trans.png', dpi=300, transparent=True)
figsum.savefig(f'{fnamesum}.pdf')