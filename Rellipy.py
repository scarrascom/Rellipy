#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author  : Sebastian Carrasco
@e-mail  : acarrasc@uni-koeln.de
@purpose : Script to extract the ellipticity and phase shift 
           of "direct" Rayleigh waves from earthquakes.

"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from obspy import Stream, UTCDateTime
from obspy.signal.rotate import rotate2zne
from obspy.clients.filesystem.sds import Client as sdsClient
from obspy.signal.cross_correlation import correlate as obcorrelate
from obspy.signal.cross_correlation import xcorr_max

from scipy.signal import correlation_lags, correlate
from scipy.signal import hilbert

def corr_phaseshift(zaux, zaur, period, srate):
    corrs_ph = correlate(zaux, zaur)
    lags = correlation_lags(len(zaux), len(zaur))
    corrs_ph /= np.max(corrs_ph)
    ## Truncate array to search between -180 and +180 deg only
    maxshift = round(period*srate/2)
    minshift = -maxshift
    boolsh = np.logical_and(lags>=minshift, lags<=maxshift)
    newcorr = corrs_ph[boolsh]
    newlag = lags[boolsh]
    ## Find the index of the maximum in the cross-correlation output
    argmaxc = np.argmax(newcorr)
    best_lag = newlag[argmaxc]
    phase_shift = best_lag*360/(period*srate)        # Phase shift in degrees
    
    return phase_shift, best_lag

def calc_vars(radial, vertical, period, ref_time):
    """
    Calculate characteristic functions and ellipticity (hvs) as 
    a function of time and different phase shifts, given the 
    vertical and radial components.
    """    
    odict = {'ph_shift': [], 'char_funs': [], 
             'times_cf': [], 'hvs': [], 'ccs': []}
    phs = np.arange(0, 182, 2)
    lags = np.unique((period*vertical.stats.sampling_rate*phs/360).astype(int))
    for lag in lags:
        ph = lag*360/(period*vertical.stats.sampling_rate)
        wwz = vertical.copy()
        wwr = radial.copy()
        wwz.data = wwz.data[lag:]
        if lag!=0:
            wwr.data = wwr.data[:-lag]
        time_arr = wwr.times() + (wwr.stats.starttime - ref_time)
        wlen = int(period)
        step = int(round(period/5, 0))
        if step<1:
            step = 1
        if wlen<1:
            wlen = 1
        t0win = []
        corrs = []
        aux_zr = Stream(traces=[wwz, wwr])
        ## Calculate the cross-correlation
        for window in aux_zr.slide(window_length=wlen, step=step):
            auxz = window.select(component='Z')[0].data
            auxr = window.select(component='R')[0].data
            norm = (np.sum(auxz**2)*np.sum(auxr**2))**0.5
            fcc = obcorrelate(auxz, auxr, shift=0, demean=True, normalize=None)/norm
            shift, valuecc = xcorr_max(fcc)
            t0win.append(window[0].stats.starttime-ref_time)
            corrs.append(valuecc)
        ## Get the envelopes
        analytic_R = hilbert(wwr.data)
        analytic_Z = hilbert(wwz.data)
        envelope_R = np.abs(analytic_R)
        envelope_Z = np.abs(analytic_Z)
        ## Normalized envelope and interpolated cross-correlation factor
        norm_env = (envelope_Z*envelope_R)/max(envelope_Z*envelope_R)
        intp_cor = np.interp(time_arr, t0win, corrs, left=np.nan, right=np.nan)
        ## Characteristic function and H/V
        char_fun = norm_env*intp_cor
        hv = envelope_R/envelope_Z
        odict['times_cf'].append(time_arr)
        odict['ph_shift'].append(ph)
        odict['char_funs'].append(char_fun)
        odict['hvs'].append(hv)
        odict['ccs'].append(intp_cor)
    
    return(odict)
    
def grid_data(inx, inzdata):
    """
    Function to fill in the missing elements as np.nan values
    
    Parameters
    ----------
    inx     : nd-array
        Times of the samples
    inzdata  : nd-array
        z-data to be gridded
    
    Return
    ------
    in_x    : nd-arrays 
    in_zdata :
    """
    common_x = inx[0]      ## Assume all the data start with the same time
    length = common_x.size
    in_x = inx.copy()
    in_zdata = inzdata.copy()
    for it, etime in enumerate(in_x):
        thisl = len(etime)
        in_x[it] = np.concatenate([etime, np.nan*np.ones(length-thisl)])
        in_zdata[it] = np.concatenate([in_zdata[it], np.nan*np.ones(length-thisl)])
    
    return(in_x, in_zdata)

def shift_unc(xvals, yvals, zgrid, per, evname, thres=0.8, save=False):
    '''
    Estimate the phase shift uncertainty based on the characteristic function 
    and a defined threshold.
    
    Parameters
    ----------
    xvals   : 1-D array (float)
        Array of times (already fixed), size M
    yvals   : 1-D array (float)
        Array of phase shifts, size N
    zgrid   : MxN array (float)
        Array of characteristic functions (already fixed)
    per     : float
        Period of the characteristic function [s]
    evname  : str
        Reference name of the event
    thres   : float
        Threshold value to calculate the ellipticity from the characteristic function
    fname   : str
        Filename of the PNG file to export (if empty, no PNG file is exported)
    
    Return
    ------
    '''
    percs = [16, 50, 84]
    rd = 0.01    # Infinitesimal radius to consider points at the contour line
    xt, yph = np.meshgrid(xvals, yvals)
    newxt, newyph = xt.flatten(), yph.flatten()
    points = np.vstack((newxt, newyph)).T
    
    fig, ax = plt.subplots()
    #cmap = cmc.cm.roma
    #norm = BoundaryNorm(np.arange(-1, 1.1, 0.05), cmap.N)
    #ax.pcolormesh(xtimes[0], yph, zcharf, 
    #              cmap=cmc.cm.roma, norm=norm, alpha=0.7, 
    #              rasterized=True)
    CS = ax.contour(xt, yph, zgrid, levels=[0.0, 0.4, 0.6, 0.8], 
                    colors='gray', alpha=0.7)
    maxval = np.nanmax(zgrid)  # Get the maximum value of the char_fun grid
    # Uncertainty will be based on a 5% range around the maximum
    thres_unc = 0.95*maxval    
    # Get the contour lines for the 5% range
    CSun = ax.contour(xt, yph, zgrid, levels=[thres_unc], 
                      colors='black', alpha=0.8)    
    # Always get the last contour line (evtl. first ones might be overtones)
    p_un = CSun.collections[0].get_paths()[-1]    
    bool_sel = p_un.contains_points(points, radius=rd)
    ph_min, ph_med, ph_max = np.percentile(newyph[bool_sel], percs)
    xt_min, xt_med, xt_max = np.percentile(newxt[bool_sel], percs)
    # If the maximum value is larger than the threshold, then get the 
    # minimum and maximum times of the time window based on the contour line
    # and the phase shift uncertainty range.
    if maxval>=thres: ## Contour of the threshold is out
        CSthres = ax.contour(xt, yph, zgrid, levels=[thres], 
                             colors='red', alpha=0.8, linestyles='dashed')
        for c, contour in enumerate(CSthres.collections[0].get_paths()):
            if any(contour.contains_points(p_un.vertices)) or\
                any(p_un.contains_points(contour.vertices)):
                break
        if thres_unc<thres: ## If uncertainty of maximum is lower than the threshold
            new_bool = contour.contains_points(points, radius=rd)
            ph_min, ph_med, ph_max = np.percentile(newyph[new_bool], percs)
            xt_min, xt_med, xt_max = np.percentile(newxt[new_bool], percs)
        # Points with phase shift in the uncertainty range
        inph = np.logical_and(newyph<=ph_max, newyph>=ph_min)  
        # Points inside contour
        incon = contour.contains_points(points, radius=rd)                
        # Points under both conditions
        selpoints = points[np.logical_and(inph, incon)]     
        ## Min and maximum times for the phase shifts of the selected points
        minTmax = [ np.nanpercentile(selpoints[selpoints.T[1]==auxph].T[0], [0, 100]) 
                                        for auxph in np.unique(selpoints.T[1]) ]
        _, med_minT, _ = np.percentile(np.asarray(minTmax).T[0], percs)
        _, med_maxT, _ = np.percentile(np.asarray(minTmax).T[1], percs)
    else:
        print(f'Max value not exceeding the threshold [{maxval}/{thres}]')
        med_minT = np.nan
        med_maxT = np.nan

    unc_dict = dict(min_ph=ph_min, med_ph=ph_med, max_ph=ph_max, 
                    minT_thres=med_minT, maxT_thres=med_maxT, 
                    maxchf_mint=xt_min, maxchf_medt=xt_med, maxchf_maxt=xt_max)
    if save:
        ax.clabel(CS, CS.levels, inline=True, 
                  fmt=FuncFormatter(lambda y,_: '{:g}'.format(y)), fontsize=10)        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Phase shift [$^o$]')
        dict_unc = dict(color='red', alpha=0.6, ls='--')        
        for phs in [ph_min, ph_med, ph_max]:
            ax.axhline(phs, **dict_unc)
        for xts in [med_minT, med_maxT, xt_min, xt_med, xt_max]:
            ax.axvline(xts, **dict_unc)
        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(which='major', ls='--', alpha=0.5, color='gray')
        ax.text(0.01, 0.98, 'T$_0$ = ' + f'{per} s \n{evname}', ha='left', 
                va='top', fontweight='bold', fontsize=10, transform=ax.transAxes)
        ax.text(0.01, ph_med, '$\phi^*$ = ' + f'{ph_med:.1f}' + '$^o$', ha='right', 
                va='bottom', fontweight='bold', color='red',
                fontsize=10, transform=ax.get_yaxis_transform())
        fig.tight_layout()
        fig.savefig(f'{evname}_{per}s_chf.png', dpi=200)
    plt.close(fig)
    return(unc_dict)

        
def shift_waves(auxz, auxr, blag, time_ref):
    ## Shift the component depending on the phase shift
    if blag>=0:
        ## Positive means R is faster than Z == Retrograde motion
        newz = auxz.data[blag:]
        # Use the times of the R component as reference
        if blag==0:         # No phase shift
            newr = auxr.data
            arr_time = auxr.times() + (auxr.stats.starttime - time_ref)
        else:
            newr = auxr.data[:-blag]
            arr_time = auxr.times()[:-blag] + (auxr.stats.starttime - time_ref)
    else:
        ## Negative means Z is faster than R == Prograde motion
        newz = auxz.data[:blag]
        newr = auxr.data[-blag:]
        arr_time = auxz.times()[:blag] + (auxz.stats.starttime - time_ref)
    # Keep a copy of the unshifted Zcomponent
    unsh_zz = auxz.copy()
    auxz.data = newz
    auxr.data = newr
    
    return unsh_zz, arr_time


def calc_ccorr(inZ, inR, paux, tref):
    """
    Calculate cross-correlation between two traces [cc(t)]
    """
    wlen = int(paux)
    if wlen<1:
        wlen = 1        # Minimum length will be 1 second
    step = int(round(paux/5, 0))    # Step every fifth of the period of the signal
    if step<1:
        step = 1                    # If it is less than one, every 1 second
    t0win = []
    corrs = []
    aux_zr = Stream(traces=[inZ, inR])  # Single Stream for Z and R comps
    for window in aux_zr.slide(window_length=wlen, step=step):
        auxz = window.select(component='Z')[0].data
        auxr = window.select(component='R')[0].data
        fcc = obcorrelate(auxz, auxr, shift=0, demean=True, normalize='naive')
        shift, valuecc = xcorr_max(fcc)
        t0win.append(window[0].stats.starttime - tref)
        corrs.append(valuecc)
    
    return t0win, corrs

def get_ellipticity(znest, midbaz, inventory, 
                    mint, maxt, evname, ref_time, label_ref, 
                    save_dir='.', width=30, newsamp=5, 
                    periods=np.arange(8, 41, 1), savepkl=True):
    """
    Function to calculate ellipticity of Rayleigh waves and their phase shift.
    Provide a long enough stream so the SNR can be computed!!
    
    Parameters
    ----------
    znest         : Stream
        Three components stream
    midbaz        : float
        Back-azimuth of the event
    invenory      : Inventory object
        Metadata of the station
    mint          : UTCDateTime
        Minimum suspected arrival time of the Rayleigh wave
    maxt          : UTCDateTime
        Maximum suspected arrival time of the Rayleigh wave
    save_dir      : string
        Directory where all the outputs (Figures and files) will be stored
    evname        : string
        Reference name of the event (e.g., S1222a)
    ref_time   : UTCDateTime
        Reference UTC time, mainly for plotting purposes
    label_ref    : string
        Label for the reference time (e.g., 'P-wave', 'OT', etc)
    width         : float
        Bandwidth to filter around every discrete period [0-100%]
    newsamp       : float
        New sampling of the data [Hz]
    periods       : list or array
        List or array of central periods where the ellipticity will be calculated.
    savepkl       : Boolean
        If desired to save the output dictionaries (True) or not (False).
    
    Return
    ------
    ZRT_dict    : dict
        Dictionary containing the rotated waveforms
    data_dict   : dict
        Dictionary containing data. The keys are the central periods and the subkeys are
        'phsh' = Optimal phase shift to maximize ZR cross-correlation
        'ts'   = Minimum and maximum times of selected window(s)
        'tmax' = UTC Time of the maximum of the characteristic function
        'cc'   = Percentiles of cross-correlation factors [16, 50, 84]
        'chf'  = Maximum value of the characteristic function
        'hv'   = Percentiles of ellipticity [16, 50, 84], 
        'ti'   = Starting time of extended window where to look for Rayleigh wave
        'tf'   = Ending time of extended window where to look for Rayleigh wave
        'SNRZ' = signal-to-noise ratio on the vertical component (Z)
        'SNRR' = signal-to-noise ratio on the radial component (R)
        'SNRT' = signal-to-noise ratio on the transversal component (T)
    """
    ## Initializing settings for processing and plotting
    logfmt = FuncFormatter(lambda y, _: '{:g}'.format(y))
    comps = 'ZRT'               # New coord system to rotate into
    win = 0.15                  # Decimal percentage for tapering 
    dpi = 200                   # DPI of output figures
    dictf = dict(zerophase=True, corners=2) # Parameters for narrow band-pass filtering
    
    ## Initialization of plotting variables
    colZ = 'royalblue'  # Color of vertical component
    colR = 'red'        # Color of radial component
    colW = 'lightgray'  # Color of selected time window for ellipticity calc
    colE = 'bisque'     # Color of initial time window
    signalZ = dict(linestyle='-', color=colZ, linewidth=1.2) # Prop Z component
    signalR = dict(linestyle='--', color=colR, linewidth=1)  # Prop R component
    envZR = dict(linestyle='-', color='orange', linewidth=1) # Prop envelope ZR
    gridd = dict(linestyle='--', color='gray', alpha=0.5)    # Prop gridding
    colwin = dict(color=colW, alpha=0.5)                # Prop ellipticity window
    colene = dict(color=colE, alpha=0.4)                # Prop initial window
    colmax = dict(ls='--', lw=2, alpha=0.7, color='k')  # Prop for marking maximum in char function
    percs = [16, 50, 84]        # Percentiles
    threscf = 0.8               # Threshold for characteristic function
    right_top_it = dict(ha='right', va='top', fontsize=11, fontstyle='italic')
    right_bot_bo = dict(ha='right', va='bottom', fontsize=11, fontweight='bold')
    
    ## Dictionary where the data will be saved
    ZRT_dict = { pp: {'Z': [], 'Zf': [], 'R': [], 'Rf': [], 
                      'T': [], 'Tf': [], 'ti': np.nan, 'tf': np.nan} 
                                                            for pp in periods}
    
    ## Resample data and rotate into ZRT coordinate system
    znest.resample(newsamp, strict_length=False)
    zrt_st = znest.copy()
    zrt_st.rotate(method='NE->RT', back_azimuth=midbaz, inventory=inventory)
    zrt_st.detrend('linear')
    zrt_st.detrend('demean')
    zrt_st.taper(0.05)
    
    for comp in comps:
        print(comp)
        for i, cp in enumerate(periods):
            # Calculate min and max frequencies for narrow-filtering
            # - and + half-bandwidth
            fmin = 1/(cp*(1+0.5*width/100))
            fmax = 1/(cp*(1-0.5*width/100))
            # Extend the time window by once the period of the signal at each edge
            # Also needed because of the tapering
            mincut = mint - cp
            maxcut = maxt + cp
            # Select the waveform, filter and cut it to the narrow window
            auxx = zrt_st.select(channel=f'??{comp}')[0].copy()
            auxx.filter('bandpass', freqmin=fmin, freqmax=fmax, **dictf)
            aux = auxx.slice(mincut, maxcut).copy()
            # Detrend and taper the signal
            aux.detrend('linear')
            aux.taper(win, type='hann')
            ## Save the cut and full waveform to the dictionary
            ZRT_dict[cp][comp] = aux            # Cut waveform
            ZRT_dict[cp][f'{comp}f'] = auxx     # Raw waveform
            ZRT_dict[cp]['ti'] = mint           # Minimum time
            ZRT_dict[cp]['tf'] = maxt           # Maximum time
            
    print('###########')
    data_dict = { per: {'phsh': np.nan, 'ts': [], 'tmax': [],
                        'cc': [], 'chf': [], 
                        'hv': [], 'ti': np.nan, 'tf': np.nan, 
                        'SNRZ': np.nan, 'SNRR': np.nan, 'SNRT':np.nan}
                                      for per in periods}
    nr = 5
    nc = round(len(periods)/nr)        
    fsize = (15, 8.5)               # Eventually change this
    dict_fig = dict(nrows=nr, ncols=nc, figsize=fsize, sharex=True)
    figraw, axraw = plt.subplots(**dict_fig)    # Initialize figure with raw waveforms
    figsh, axsh = plt.subplots(**dict_fig)      # Initialize figure with shifted waveforms
    figcc, axcc = plt.subplots(sharey=True, **dict_fig)     # Initialize figure with cross-correlations
    figch, axch = plt.subplots(sharey=True, **dict_fig)     # Initialize figure with characteristic function
    figho, axho = plt.subplots(nrows=nr, ncols=nc, figsize=fsize)   # Initialize figure with hodograms
    fighv, axhv = plt.subplots(sharey='row', **dict_fig)    # Initialize figure with ellipticity (H/V)
    for ip, pp in enumerate(periods):
        compsd = ZRT_dict[pp]       # Subdictionary with the traces
        dicdata = data_dict[pp]     # Subdictionary with the output data (to create)    
        tini = compsd['ti']         # Starting time of suspected Rayleigh wave
        tend = compsd['tf']         # Ending time of suspected Rayleigh wave
        dicdata['ti'] = tini        # Replace the values
        dicdata['tf'] = tend
        ## Get the SNR for all the comps
        # For Z
        maxZs = np.max(np.abs(compsd['Zf'].slice(tini, tend).data))
        avepreZ = np.median(np.abs(compsd['Zf'].slice(tini-(tend-tini)-5, tini-5).data))
        SNRZ = maxZs/avepreZ
        dicdata['SNRZ'] = SNRZ
        # For R
        maxRs = np.max(np.abs(compsd['Rf'].slice(tini, tend).data))
        avepreR = np.median(np.abs(compsd['Rf'].slice(tini-(tend-tini)-5, tini-5).data))
        SNRR = maxRs/avepreR
        dicdata['SNRR'] = SNRR
        # For T
        maxTs = np.max(np.abs(compsd['Tf'].slice(tini, tend).data))
        avepreT = np.median(np.abs(compsd['Tf'].slice(tini-(tend-tini)-5, tini-5).data))
        SNRT = maxTs/avepreT
        dicdata['SNRT'] = SNRT
        ###
        inr = int(ip/nc)
        inc = int(ip%nc)
        print(f'Getting H/V for T = {pp} s')
        ## Radial component is 90 deg phase-advanced w.r.t. the vertical component. (Retrograde)
        ## Prograde = vertical component is 90 deg phase-advanced w.r.t radial comp
        ## The trace is already cut!
        zz = compsd['Z'].copy()
        rr = compsd['R'].copy()
        ## Preparing axes for hodograms
        htimes = (zz.stats.starttime - ref_time) + rr.times()
        axhp = axho[inr][inc]
        axhp.spines['top'].set_color('none')
        axhp.spines['bottom'].set_position('zero')
        axhp.spines['left'].set_position('zero')
        axhp.spines['right'].set_color('none')
        axhp.set_xticks([])
        axhp.set_yticks([])
        axhp.plot((1), (0), ls="", marker=">", ms=8, color="k",
                transform=axhp.get_yaxis_transform(), clip_on=False)
        axhp.plot((0), (1), ls="", marker="^", ms=8, color="k",
                transform=axhp.get_xaxis_transform(), clip_on=False)
        mapp = axhp.scatter(rr, zz, c=htimes, s=40, alpha=0.8, cmap='RdYlBu')
        axraw[inr][inc].set_yticks([])
        axsh[inr][inc].set_yticks([])
        ## Get the optimal phase shift (in deg) and lag (in samples) by cross-correlating Z and R
        phase_shift, best_lag = corr_phaseshift(zz, rr, pp, newsamp)
        ## Cut and synchronize the waveforms with the given lag sample
        ## Create a time array referred to the ref_time
        zzno, time_arr = shift_waves(zz, rr, best_lag, ref_time)
        dicdata['phsh'] = phase_shift       # Save phase shift to dict
        ## Calculate cross-correlation factors between the two signals
        t0win, corrs = calc_ccorr(zz, rr, pp, ref_time)
        ## Get the envelopes
        analytic_R = hilbert(rr.data)
        analytic_Z = hilbert(zz.data)
        envelope_R = np.abs(analytic_R)
        envelope_Z = np.abs(analytic_Z)
        ## Normalized envelope
        norm_env = (envelope_Z*envelope_R)/max(envelope_Z*envelope_R)
        intp_cor = np.interp(time_arr, t0win, corrs, left=np.nan, right=np.nan)
        ## Characteristic function
        char_fun = norm_env*intp_cor
        hv = envelope_R/envelope_Z
        choose_win = char_fun > threscf
        ##################### PLOTTING RESULTS ###############################
        ## Plotting detail Figures for different periods
        fig, ax = plt.subplots(nrows=5, figsize=(7,9), sharex=True)
        ttitle = f'Event {evname}, T = {pp} s, bw = {width}%'
        ttext = f'{pp} s'
        phtext = f'{phase_shift:.1f}$^o$'
        axhp.text(1, 0, ttext, transform=axhp.transAxes, **right_bot_bo)
        axhp.text(1, 0.98, phtext, transform=axhp.transAxes, **right_top_it)
        aux_ti = tini - ref_time    
        aux_te = tend - ref_time    
        norm_trr = rr.times() + (rr.stats.starttime - ref_time)
        norm_tzzno = zzno.times() + (zzno.stats.starttime - ref_time)
        norm_tzz = zz.times() + (zz.stats.starttime-ref_time)
        ### Plot unshifted radial and vertical componens
        for a_ax in (ax[0], axraw[inr][inc]):
            a_ax.grid(**gridd)
            a_ax.axvspan(aux_ti, aux_te, **colene)
            a_ax.plot(norm_tzzno, zzno.data, label='Z', **signalZ)
            a_ax.plot(norm_trr, rr.data, label='R', **signalR)
        ax[0].set_title(ttitle)
        ax[0].set_ylabel('Disp [m]')
        ax[0].legend(loc='best', fontsize=10)
        a_ax.text(1, 0, ttext, transform=axraw[inr][inc].transAxes, **right_bot_bo)
        a_ax.set_yticks([])
        ### Plot unshifted radial and shifted vertical component
        for b_ax in (ax[1], axsh[inr][inc]):
            b_ax.grid(**gridd)
            b_ax.axvspan(aux_ti, aux_te, **colene)
            b_ax.plot(norm_tzz, zz.data, label='Z shifted', **signalZ)
            b_ax.plot(norm_trr, rr.data, label='R', **signalR)
            b_ax.text(1, 0.98, phtext, transform=b_ax.transAxes, **right_top_it)
        ax[1].set_ylabel('Disp [m]')
        ax[1].legend(loc='lower right', fontsize=10)
        b_ax.text(1, 0, ttext, transform=axsh[inr][inc].transAxes, **right_bot_bo)
        b_ax.set_yticks([])
        ### Plot cross-correlation and normalized envelope, cc(t) and nu(t)
        for c_ax in (ax[2], axcc[inr][inc]):
            c_ax.grid(**gridd)
            c_ax.axvspan(aux_ti, aux_te, **colene)
            c_ax.plot(t0win, corrs, label=r'$\nu$(t)', **signalZ)
            c_ax.plot(norm_trr, norm_env, label=r'$\eta$(t)', **envZR)       
            c_ax.set_ylim(-1.05, 1.05)
        ax[2].set_ylabel('[ad]')
        ax[2].legend(loc='best', fontsize=10)
        c_ax.text(1, 0, ttext, transform=axcc[inr][inc].transAxes, **right_bot_bo)
        c_ax.text(1, 0.98, phtext, transform=axcc[inr][inc].transAxes, **right_top_it)
        #### Plot characteristic function chi(t)
        for chf_ax in (ax[3], axch[inr][inc]):
            chf_ax.grid(**gridd)
            chf_ax.set_ylim(-1.05, 1.05)
            chf_ax.axvspan(aux_ti, aux_te, **colene)
            chf_ax.axhline(threscf, color='k', linestyle='--', lw=0.8)
            chf_ax.plot(time_arr, char_fun, label='r$\chi$(t)', **envZR)       
        ax[3].set_ylabel('[ad]')
        ax[3].legend(loc='best', fontsize=10)
        chf_ax.text(1, 0, ttext, transform=axch[inr][inc].transAxes, **right_bot_bo)
        chf_ax.text(1, 0.98, phtext, transform=axch[inr][inc].transAxes, **right_top_it)
        ### Plot H/V(t) curve
        ax[4].set_ylabel('Ellipticity [R/Z]')
        ax[4].set_xlabel(f'Seconds after {label_ref}')
        for hv_ax in (ax[4], axhv[inr][inc]):
            hv_ax.grid(**gridd)
            hv_ax.axvspan(aux_ti, aux_te, **colene)
            hv_ax.plot(time_arr, hv)
            hv_ax.set_yscale('log')
            hv_ax.set_ylim(0.1, 10)
            hv_ax.yaxis.set_major_formatter(logfmt)        
        ## Choose the right time window where to compute the ellipticity
        if sum(choose_win)>0:
            mint = time_arr[choose_win][0]
            auxhv = []
            auxcc = []
            auxchf = []
            # Merge the adyacent time windows when needed. 
            # Otherwise, keep different time windows
            for it, tau in enumerate(time_arr[choose_win][:-1]):
                diff = time_arr[choose_win][it+1] - tau
                auxhv.append(hv[choose_win][it])
                auxcc.append(intp_cor[choose_win][it])
                auxchf.append(char_fun[choose_win][it])
                if round(diff, 2)==1/newsamp:
                    if it==len(time_arr[choose_win])-2:
                        maxt = time_arr[choose_win][it+1]
                        indmax = np.argmax(auxchf)
                        tmaxR = ref_time + mint + indmax/newsamp
                        for axx in ax:
                            axx.axvspan(mint, maxt, **colwin)
                        axraw[inr][inc].axvspan(mint, maxt, **colwin)
                        axsh[inr][inc].axvspan(mint, maxt, **colwin)
                        axcc[inr][inc].axvspan(mint, maxt, **colwin)
                        axch[inr][inc].axvspan(mint, maxt, **colwin)
                        axhv[inr][inc].axvspan(mint, maxt, **colwin)
                        ax[3].axvline(mint + indmax/newsamp, **colmax)
                        axch[inr][inc].axvline(mint + indmax/newsamp, **colmax)
                        ## Save the data
                        dicdata['ts'].append((ref_time+mint, ref_time+maxt))
                        dicdata['tmax'].append(tmaxR)
                        dicdata['hv'].append(np.percentile(auxhv, percs))
                        dicdata['cc'].append(np.percentile(auxcc, percs))
                        dicdata['chf'].append(np.max(auxchf))
                        auxhv = []
                        auxcc = []
                        auxchf = []
                    else:
                        continue
                else:
                    maxt = tau
                    indmax = np.argmax(auxchf)
                    tmaxR = ref_time + mint + indmax/newsamp
                    for axx in ax:
                        axx.axvspan(mint, maxt, **colwin)
                    axraw[inr][inc].axvspan(mint, maxt, **colwin)
                    axsh[inr][inc].axvspan(mint, maxt, **colwin)
                    axcc[inr][inc].axvspan(mint, maxt, **colwin)
                    axch[inr][inc].axvspan(mint, maxt, **colwin)
                    axhv[inr][inc].axvspan(mint, maxt, **colwin)
                    ax[3].axvline(mint + indmax/newsamp, **colmax)
                    axch[inr][inc].axvline(mint + indmax/newsamp, **colmax)
                    ## Save the data
                    dicdata['ts'].append((ref_time+mint, ref_time+maxt))
                    dicdata['tmax'].append(tmaxR)
                    dicdata['hv'].append(np.percentile(auxhv, percs))
                    dicdata['cc'].append(np.percentile(auxcc, percs))
                    dicdata['chf'].append(np.max(auxchf))
                    auxhv = []
                    auxcc = []
                    auxchf = []
                    mint = time_arr[choose_win][it+1]
        else:
            dicdata['ts'].append((np.nan, np.nan))
            dicdata['tmax'].append(np.nan)
            dicdata['cc'].append([np.nan, np.nan, np.nan])
            dicdata['chf'].append(np.nan)
            dicdata['hv'].append([np.nan, np.nan, np.nan])
        fname = f'{save_dir}/{pp:03d}s_{evname}_bw{width:02d}.png'
        mint_plot = np.nanmin(dicdata['ts'])
        maxt_plot = np.nanmax(dicdata['ts'])
        if type(mint_plot)==UTCDateTime and type(maxt_plot)==UTCDateTime:
            ax[-1].set_xlim(mint_plot-ref_time-10*pp, maxt_plot-ref_time+10*pp)
        fig.tight_layout()
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    ## Tuning the plots
    p_xlab = (0.5, 0.03)    # Position of x-label
    p_ylab = (0.02, 0.5)    # Position of y-label
    textx = dict(va='top', ha='center', fontweight='bold', fontsize=12)
    texty = dict(va='center', ha='right', fontweight='bold', fontsize=12, rotation=90)
    splots_dict = dict(top=0.97, bottom=0.08, left=0.05, 
                       right=0.985, hspace=0.2, wspace=0.15)
    
    figraw.subplots_adjust(**splots_dict)
    figraw.text(p_xlab[0], p_xlab[1], s=f'Time after {label_ref} [s]', **textx)
    figraw.text(p_ylab[0], p_ylab[1], s='Displacement [m]', **texty)
    figraw.suptitle(f'R (red) and Z (blue) unshifted comps - {evname}', y=1, va='top')
    figraw.savefig(f'{save_dir}/{evname}_unshiftedRZ_bw{width:02d}.png', dpi=dpi)
    plt.close(figraw)
    
    figsh.subplots_adjust(**splots_dict)
    figsh.text(p_xlab[0], p_xlab[1], s=f'Time after {label_ref} [s]', **textx)
    figsh.text(p_ylab[0], p_ylab[1], s='Displacement [m]', **texty)
    figsh.suptitle(f'Aligned R (red) and Z (blue) - {evname}', y=1, va='top')
    figsh.savefig(f'{save_dir}/{evname}_shiftedRZ_bw{width:02d}.png', dpi=dpi)
    plt.close(figsh)
    
    figcc.subplots_adjust(**splots_dict)
    figcc.text(p_xlab[0], p_xlab[1], s=f'Time after {label_ref} [s]', **textx)
    figcc.text(p_ylab[0], p_ylab[1], s='Cross-corr | Norm envelope Z*R', **texty)
    figcc.suptitle(f'R-Zs cross-correlation and normalized Z*R envelope - {evname}', y=1, va='top')
    figcc.savefig(f'{save_dir}/{evname}_cc_env_bw{width:02d}.png', dpi=dpi)
    plt.close(figcc)
    
    figch.subplots_adjust(**splots_dict)
    figch.text(p_xlab[0], p_xlab[1], s=f'Time after {label_ref} [s]', **textx)
    figch.text(p_ylab[0], p_ylab[1], s='Characteristic function [ad]', **texty)
    figch.suptitle(f'Characteristic function (>{threscf}) - {evname}', y=1, va='top')
    figch.savefig(f'{save_dir}/{evname}_charf_bw{width:02d}.png', dpi=dpi)
    plt.close(figch)
    
    fighv.subplots_adjust(**splots_dict)
    fighv.text(p_xlab[0], p_xlab[1], s='Time after {label_ref} [s]', **textx)
    fighv.text(p_ylab[0], p_ylab[1], s='Ellipticity [R/Z]', **texty)
    fighv.suptitle(f'Rayleigh wave ellipticity - {evname}', y=1, va='top')
    fighv.savefig(f'{save_dir}/{evname}_ellipRZ_bw{width:02d}.png', dpi=dpi)
    plt.close(fighv)
    
    figho.subplots_adjust(**splots_dict)
    figho.text(p_xlab[0], p_xlab[1], s='Radial [R]', **textx)
    figho.text(p_ylab[0], p_ylab[1], s='Vertical [Z]', **texty)
    ## Time colorbar for hodograms
    axins = inset_axes(axho[-1][-2], width="150%", height="7%", 
                       loc='lower left', borderpad=-1)
    cbar = figho.colorbar(mapp, cax=axins, orientation="horizontal")
    cbar.set_label('Time')
    cbar.set_ticks([])
    figho.savefig(f'{save_dir}/{evname}_hodo_bw{width:02d}.png', dpi=dpi)
    plt.close(figho)
    
    ## Final H/V curve, based on computations
    maphv = plt.get_cmap('binary')
    figsum, axsum = plt.subplots(figsize=(8,5))
    figphase, axphase = plt.subplots(figsize=(8,5))
    figts, axts = plt.subplots(figsize=(8,5))
    axphase.fill_between(periods, 70, 110, color='red', 
                         linewidth=0, alpha=0.2)
    
    ## For each period, get the ellipticity curve
    for per, dicp in data_dict.items():
        ## ... as long as there was an actual optimal phase shift.
        if not np.isnan(dicp['phsh']):
            axphase.plot(per, dicp['phsh'], marker='o')
            if len(dicp['tmax'])>=1:
                if type(dicp['tmax'][0])==UTCDateTime:
                    axts.errorbar(x=per, y=dicp['tmax'][0]-ref_time, 
                                  yerr=[[dicp['tmax'][0]-dicp['ts'][0][0]], 
                                        [dicp['ts'][0][1]-dicp['tmax'][0]]], 
                                  fmt='o', color='k', markersize=10)
            aux_chf = sorted(dicp['chf'])
            new_hv = [ dicp['hv'][np.where(ach==dicp['chf'])[0][0]] for ach in aux_chf if not np.isnan(ach)]
            if len(new_hv)>0:
                for j, valchf in enumerate(aux_chf):
                    minhv, medhv, maxhv = new_hv[j]
                    axsum.errorbar(x=per, y=medhv, yerr=[[medhv-minhv], [maxhv-medhv]], 
                                   fmt='o', color=maphv(valchf), markersize=10)
    
    axsum.set_yscale('log')
    axsum.set_xscale('linear')
    axsum.set_ylim(0.1, 10)
    axsum.yaxis.set_major_formatter(logfmt)
    axsum.xaxis.set_major_formatter(logfmt)
    axsum.grid(**gridd)
    axsum.set_xlabel('Period [s]', fontweight='bold')
    axsum.set_ylabel(f'Rayleigh ellipticity [R/Z] - {evname}', fontweight='bold')
    ## 
    axphase.set_xscale('linear')
    axphase.set_ylim(0, 180)
    axphase.xaxis.set_major_formatter(logfmt)
    axphase.xaxis.set_minor_formatter(logfmt)
    axphase.grid(**gridd)
    axphase.set_xlabel('Period [s]', fontweight='bold')
    axphase.set_ylabel(f'Phase angle RZ [$\phi$] - {evname}', fontweight='bold')
    ##
    axts.set_xscale('linear')
    axts.xaxis.set_major_formatter(logfmt)
    axts.xaxis.set_minor_formatter(logfmt)
    axts.grid(**gridd)
    axts.set_xlabel('Period [s]', fontweight='bold')
    axts.set_ylabel(f'Rayleigh time w.r.t {label_ref} [s]', fontweight='bold')
    ## Tight the figures and save them
    figphase.tight_layout()
    figphase.savefig(f'{save_dir}/{evname}_Phase_bw{width:02d}.png', dpi=dpi)
    plt.close(figphase)
    ##
    figsum.tight_layout()
    figsum.savefig(f'{save_dir}/{evname}_Ellip_bw{width:02d}.png', dpi=dpi)
    plt.close(figsum)
    ##
    figts.tight_layout()
    figts.savefig(f'{save_dir}/{evname}_Times_bw{width:02d}.png', dpi=dpi)
    plt.close(figts)
        
    if savepkl:
        print('Saving waveform data')
        with open(f'{save_dir}/{evname}_waves.pkl', 'wb') as fpickle:
            pickle.dump(ZRT_dict, fpickle, pickle.HIGHEST_PROTOCOL)
        
        print('Saving results')
        with open(f'{save_dir}/{evname}.pkl', 'wb') as fpickle:
            pickle.dump(data_dict, fpickle, pickle.HIGHEST_PROTOCOL)
    
    return ZRT_dict, data_dict


def st_adjust(in_st):
    """
    Synchronize and cut stream to use the same number of samples
    It assumes there are three traces, either ZNE or UVW
    
    Parameters
    ----------
    in_st : Stream
        Input raw stream to synchronize (should have three traces)
    
    Return (operation is in-place)
    ------
    cti : UTCDateTime object
        Initial commong time
    """
    ntrs = len([ tr.id[-1] for tr in in_st ])
    if ntrs!=3:
        return np.nan                   # Return nan if gappy or missing comps
    cti = max([tr.stats.starttime for tr in in_st]) # Larger init
    ctf = min([tr.stats.endtime for tr in in_st])
    sr = in_st[0].stats.sampling_rate   # All have the same sampling rate
    nsamps = int(sr*(ctf-cti))          # Number of total samples
    newt = np.arange(nsamps)/sr         # New array of times
    for tr in in_st:
        dt = tr.stats.starttime - cti   # Diff between current trace and common start time
        auxt0 = tr.times() + dt         # Add the delay
        auxdata = tr.data               # Get the old data
        newdata = np.interp(newt, auxt0, auxdata)   # Interpolate 
        tr.stats.starttime = cti        # Replace starting time
        tr.data = newdata               # Replace data
    return cti


def getrot_insight(client, stime, etime, loccha, 
                   out='DISP', mode='UVW-ZNE', tap = 0.05,
                   pre_filt=[0.005, 0.01, 50.0, 60.0]):
    """
    Function to rotate InSight data from a given waveform Client
    (originally SDS archive)
    
    Parameters
    ----------
    client : Client  (Obspy object)
        Client from where the data will be retrieved
    stime  : UTCDateTime (Obspy object)
        Starting time of the waveform data to be cut
    etime  : UTCDateTime (Obspy object)
        Ending time of the waveform data to be cut
    loccha : string
        Station ID WITHOUT the component (e.g., BQ.DREG..HH, GR.AHRW..HH)
    out    : string
        Ground-motion units. Either from DISP, VEL, ACC
    mode   : string
        Components of the old and new system, separated by - [UVW-ZNE]
    tap    : float
        Taper percentage [0.05 = 5%]
    pre_filt : list
        List of 4 values to pass to the bandpass filter.
        
    Return
    ------
    znest   : Stream
        Stream with the rotated traces (into ZNE components)
    """
    net, sta, loc, cha = loccha.split('.')
    st = client.get_waveforms(network=net, station=sta, 
                              location=loc, channel=f'{cha}*', 
                              starttime=stime, endtime=etime, merge=None)
    ## Synchronize and cut waveforms to use the same number of samples    
    ct0 = st_adjust(st)           # Synchronize traces
    if not type(ct0)==UTCDateTime:
        print('Gappy data or missing components! Returning empty stream...')
        return Stream()
    
    st.detrend('linear')
    st.detrend('demean')
    st.taper(tap)
    st.remove_response(inventory=inventory, output=out, zero_mean=True, 
                       taper=True, taper_fraction=tap, pre_filt=pre_filt)
    
    in1, in2, in3 = mode.split('-')[0]
    out1, out2, out3 = mode.split('-')[1]
    tra1 = st.select(channel=f'??{in1}')[0]
    dataux_1 = tra1.data
    dipaux_1 = inventory.get_orientation(tra1.id, datetime=ct0)['dip']
    aziaux_1 = inventory.get_orientation(tra1.id, datetime=ct0)['azimuth']
    ## Get information of channel V
    tra2 = st.select(channel=f'??{in2}')[0]
    dataux_2 = tra2.data
    dipaux_2 = inventory.get_orientation(tra2.id, datetime=ct0)['dip']
    aziaux_2 = inventory.get_orientation(tra2.id, datetime=ct0)['azimuth']
    
    ## Get information of channel W
    tra3 = st.select(channel=f'??{in3}')[0]
    dataux_3 = tra3.data
    dipaux_3 = inventory.get_orientation(tra3.id, datetime=ct0)['dip']
    aziaux_3 = inventory.get_orientation(tra3.id, datetime=ct0)['azimuth']
    
    ZNE = rotate2zne(dataux_1, aziaux_1, dipaux_1,
                     dataux_2, aziaux_2, dipaux_2,
                     dataux_3, aziaux_3, dipaux_3)
    znest = st.copy()
    ## Replace old data with the rotated data
    for ix, com in mode.split('-')[0]:
        chan = f'??{com}'
        znest.select(channel=chan)[0].data = ZNE[ix]
        znest.select(channel=chan)[0].stats.channel = znest.select(channel=chan)[0].stats.channel[:2] + mode.split('-')[1][ix]
    
    return znest

if __name__=="__main__":
    ## Custom parameters
    tP = UTCDateTime('2022-05-04T23:27:45')     # P-wave arrival for S1222a
    refT = 'P-wave'                             # Whether referred to P-wave arrival time or quake OT
    tmin = 400                                  # Minimum time delay after refT
    tmax = 800                                  # Maximum time delay after refT
    evname = 'S1222a'                           # Event reference name
    midbaz = 129                                # Back-azimuth
    archi_seis = '/mnt/Projects/InSight/archive'    # SDS archive where the data is stored
    inv_pkl = '/home/seismo/GDrive/PhD/InSight/Metadata/dataless.XB.ELYSE.seed.pkl' # Inventory as a pickle file
    out_dir = '.'                               # Output directory
    dictf = dict(corners=2, zerophase=True)     # Filter dictionary
    client = sdsClient(archi_seis)              # Loading the client
    print('Reading Inventory...')
    with open(inv_pkl, 'rb') as fpk:
        inventory = pickle.load(fpk)
       
    mintR = tP + tmin                           # Minimum time of the suspected window
    maxtR = tP + tmax                           # Maximum time of the suspected window

    ## Cut little bit longer time windows
    stime = tP + 0.9*tmin
    etime = tP + 1.1*tmax
    
    stid = 'XB.ELYSE.02.BH'                     # Station ID (without component)

    ## Still need to implement getting the times 
    ## from dictionary with picked min and max arrival times
    # rtimesl = dict_curves['min']['time']        # 
    # rtimesh = dict_curves['max']['time']        # 
    # tfreqsl = dict_curves['min']['freq']        # 
    # tfreqsh = dict_curves['max']['freq']        # 
    # auxtl = np.array([ auxt - rtimesl[0] for auxt in rtimesl ])
    # auxth = np.array([ auxt - rtimesh[0] for auxt in rtimesh ])
    # mintt = rtimesl[0] + np.interp(1/cp, tfreqsl, auxtl)
    # maxtt = rtimesh[0] + np.interp(1/cp, tfreqsh, auxth)

    ## Get the data
    zne_st = getrot_insight(client, stime, etime, stid)
    ## Finally get the ellipticity of the event at different periods
    get_ellipticity(zne_st, midbaz, inventory, 
                    mintR, maxtR, 
                    evname, tP, refT, savedir=out_dir)
