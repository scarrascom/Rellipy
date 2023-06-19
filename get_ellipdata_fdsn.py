#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author  : Sebastian Carrasco
@e-mail  : acarrasc@uni-koeln.de
@purpose : Implementation of Rellipy (calculation of Rayleigh-wave ellipticity and phase shift)
           for data through FDSN web services (or stored locally)
"""

import os
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

from Rellipy import get_ellipticity

if __name__=="__main__":
    client_evts = Client("IRIS")    # Client from where to get the events catalogue
    client_data = Client('BGR')     # Client from where to get the waveform data and metadata
    
    starttime = '2023-02-06'        # Starting date of the catalogue, YYYY-MM-DD
    endtime = '2023-02-08'          # Ending date of the catalogue, YYYY-MM-DD
    
    minmag = 7.0                    # Minimum magnitude of the events to analyze
    maxmag = 8.2                    # Maximum magnitude of the events to analyze
    
    mindist = 10                    # Minimum distance of the events, in degrees
    maxdist = 120                   # Maximum distance of the events, in degrees
    
    maxdep = 100                    # Maximum depth, in km
    stid = 'BQ.DREG..HHZ'           # Full station ID (component is not used)
    # Whether restitution should be applied before processing (True) or not (False)
    # It might save some total seconds/minutes for long waveforms and many events
    rem_resp = True
    m2km = 1000
    
    net, sta, loc, cha = stid.split('.')
    t0 = UTCDateTime(starttime)
    tf = UTCDateTime(endtime)
    
    periodsT = np.hstack([np.arange(8, 35, 2), np.arange(36, 60, 4), np.arange(60, 110, 5)])
    ## Get inventory from FDSN services
    dl = client_data.get_stations(network=net, station=sta, level='response')
    ### If desired, you can also get the data from a local SDS archive and local inventory
    #from obspy.clients.filesystem.sds import Client as sdsClient
    #from obspy import read_inventory
    #path_sds = '/mnt/SC-Share/seiscomp/var/lib/archive'
    #path_inv = '/mnt/Station/Inventory/LOCAL/STATIONXML/FULL_BENS.xml'
    #client_data = sdsClient(path_sds)
    #dl = read_inventory(path_inv)
    thisdl = dl.select(station=sta, time=tf)
    lat_rec = thisdl.get_coordinates(stid)['latitude']
    lon_rec = thisdl.get_coordinates(stid)['longitude']
    ele_rec = thisdl.get_coordinates(stid)['elevation']
    
    ## Get catalogue via FDSN services
    print('Getting catalogue...')
    cat = client_evts.get_events(starttime=t0, endtime=tf, minmagnitude=minmag, 
                                 maxmagnitude=maxmag, longitude=lon_rec, latitude=lat_rec, 
                                 minradius=mindist, maxradius=maxdist, maxdepth=maxdep,
                                 includearrivals=False, catalog='GCMT')
    
    print(f'There are {len(cat)} events!')
    
    longitudes = []
    latitudes = []
    epi_distances = []
    bazs = []
    depths = []
    mags = []
    evnames = [] 
    ori_times = []
    
    pwd = os.getcwd()     # Output directory where per-event subdirectories will be created
    for event in cat:
        prefor = event.origins[0]
        if not prefor.depth:
            print('Event skipped!')
            continue
        ## Getting the event parameters
        orig_time = prefor.time         # As UTCDateTime object
        depth = prefor.depth/m2km       # Convert to km
        lat = prefor.latitude
        lon = prefor.longitude
        # If different magnitudes are available, get the median
        mag = np.median([ magevt.mag for magevt in event.magnitudes ])
        # Distance and back-azimuth between source and station
        distm, baz, _ = gps2dist_azimuth(lat_rec, lon_rec, lat, lon)
        distkm = distm/m2km     # Convert to km
        ## The event name is given by the origin time - A subdirectory is created
        evname = orig_time.strftime('%Y%m%d_%H%M%S')
        full_path = os.path.join(pwd, evname)
        str_time = orig_time.strftime('%Y-%m-%dT%H:%M:%S')
        print('##################################################')
        print(f'Event located at {lat}, {lon} (lat/lon)')
        print(f'Epicentral distance: {distkm:.2f} km')
        print(f'Back-azimuth from station: {baz:.1f} deg')
        print(f'Depth = {depth} km')
        print(f'Magnitude = {mag}')
        print(f'Origin time = {str_time} UTC')
        ## Getting the right inventory at the time of the 
        thisdl = dl.select(station=sta, time=orig_time)
        try:
            ## Get the times for the time window
            mintR = orig_time + distkm/4.5
            maxtR = orig_time + distkm/2.5
            auxdt = (maxtR - mintR)*1.1
            zne_st = client_data.get_waveforms(net, sta, loc, f'{cha[:-1]}*', 
                                               mintR - auxdt, mintR + auxdt)
            if len(zne_st)==0:
                print('No data!!')
                continue
            elif len(zne_st)>3:
                print('Gappy data!!')
                continue
            if rem_resp:
                zne_st.remove_response(thisdl, output='DISP')
            os.makedirs(full_path, exist_ok=True)
            get_ellipticity(znest=zne_st, 
                            midbaz=baz, 
                            inventory=thisdl, 
                            mint=mintR, maxt=maxtR, 
                            evname=evname,          # String object
                            ref_time=orig_time,     # UTCDateTime object
                            label_ref='OT',         # Reference time is origin time (OT) | string object
                            save_dir=full_path,     # Save the outputs in the corresponding directory
                            periods=periodsT)       # For the pre-defined periods
        except:
            print('Oops, there was a problem! Skipping this event')
            continue
        ## If successful, appending parameters to lists and save
        longitudes.append(lon)
        latitudes.append(lat)
        epi_distances.append(distkm)
        bazs.append(baz)
        depths.append(depth)
        mags.append(mag)
        ori_times.append(str_time)
        evnames.append(evname)
        np.savetxt(f'{full_path}/parameters.txt', 
                   np.transpose([[str_time], [lat], [lon], [distkm], [baz], [depth], [mag]]),
                   delimiter=',', fmt='%s')
    ## Saving a full catalogue
    np.savetxt(f'Full_catalogue_{sta}.txt', 
               np.transpose([ori_times, latitudes, longitudes, epi_distances, bazs, depths, mags]),
               delimiter=',', fmt='%s')
