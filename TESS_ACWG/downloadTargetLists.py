import requests, os, time, pdb
import pandas as pd
import numpy as np
from datetime import datetime

def targetsWithPublishedConfirmation( forceDownload=False ):
    """
    Confirmed Planets from NASA Exoplanet Archive.    

    """
    ##date = str(datetime.date(datetime.now()))
    date = '2022-11-03'
    path = 'PS_'+date+'.csv'
    confirmedFpath = path
    if not forceDownload:
        if os.path.exists(f'{os.getcwd()}/{path}'):
            return confirmedFpath
    
    default_query = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps+where+tran_flag+=+1&format=csv"
    
    add_few_elements = ','.join(
        ['pl_name','default_flag','discoverymethod','disc_year','disc_facility','tran_flag',\
         'pl_orbper','pl_orbpererr1','pl_orbpererr2',\
         'pl_orbperlim','pl_orbsmax','pl_orbsmaxerr1','pl_orbsmaxerr2','pl_orbsmaxlim','pl_rade',\
         'pl_radeerr1','pl_radeerr2','pl_radelim','pl_masse','pl_masseerr1','pl_masseerr2','pl_masselim',\
         'pl_bmassprov','pl_orbeccen','pl_orbeccenerr1','pl_orbeccenerr2','pl_orbeccenlim','pl_insol',\
         'pl_insolerr1','pl_insolerr2','pl_insollim','pl_orbincl','pl_orbinclerr1','pl_orbinclerr2',\
         'pl_orbincllim','pl_imppar','pl_impparerr1','pl_impparerr2','pl_impparlim','pl_trandur',\
         'pl_trandurerr1','pl_trandurerr2','pl_trandurlim','pl_ratdor','pl_ratdorerr1','pl_ratdorerr2',\
         'pl_ratdorlim','pl_ratror','pl_ratrorerr1','pl_ratrorerr2','pl_ratrorlim','st_teff','st_tefferr1',\
         'st_tefferr2','st_tefflim','st_rad','st_raderr1','st_raderr2','st_radlim','st_mass','st_masserr1',\
         'st_masserr2','st_masslim','st_logg','st_loggerr1','st_loggerr2','st_logglim','rastr','ra',\
         'decstr','dec','sy_dist','sy_disterr1','sy_disterr2','sy_vmag','sy_vmagerr1','sy_vmagerr2',\
         'sy_jmag','sy_jmagerr1','sy_jmagerr2','sy_hmag','sy_hmagerr1','sy_hmagerr2','sy_kmag',\
         'sy_kmagerr1','sy_kmagerr2'])

    all_planets =  requests.get(default_query.split('*')[0] + add_few_elements + default_query.split('*')[1]) 
    
    all_planets = all_planets.text.split('\n')
    
    planets_df = pd.DataFrame(columns=all_planets[0].split(','), 
                              data = [i.split(',') for i in all_planets[1:-1]])
    planets_df = planets_df.replace(to_replace='', value=np.nan)

    
    for i in planets_df:
        planets_df[i] = planets_df[i].str.replace('"', "")
    planets_df.head()
    planets_df.to_csv(confirmedFpath, index=False)

    return confirmedFpath

def targetsConfirmedTESS( forceDownload=False ):
    """
    Planets confirmed by TESS from NASA Planet Archive.
    
    """
    
    ##date = str(datetime.date(datetime.now()))
    date = '2022-11-03'
    path = 'PS_TESS_'+date+'.csv'
    confirmedFpath = path
    if not forceDownload:
        if os.path.exists(f'{os.getcwd()}/{path}'):
            return confirmedFpath
    
    default_query = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps+where+disc_facility+=+%27Transiting%20Exoplanet%20Survey%20Satellite%20(TESS)%27&format=csv"
    
    add_few_elements = ','.join(['tic_id'])
        
    all_planets =  requests.get(default_query.split('*')[0] + add_few_elements + default_query.split('*')[1]) 
    
    all_planets = all_planets.text.split('\n')
    
    planets_df = pd.DataFrame(columns=all_planets[0].split(','), 
                              data = [i.split(',') for i in all_planets[1:-1]])
    planets_df = planets_df.replace(to_replace='', value=np.nan)

    
    for i in planets_df:
        planets_df[i] = planets_df[i].str.replace('"', "")
    planets_df.head()
    planets_df.to_csv(confirmedFpath, index=False)

    return confirmedFpath

def targetsUnpublishedTOIs( csvIpath='TOI_YYYY-MM-DD.csv', forceDownload=False ):
    """
    TESS Project Candidates from NASA Exoplanet Archive.
    
    """
    ##date = str(datetime.date(datetime.now()))
    date = '2022-11-03'
    path = 'TOI_'+date+'.csv'
    toiFpath = path
    if not forceDownload:
        if os.path.exists(f'{os.getcwd()}/{path}'):
            return toiFpath
    
    default_query = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=TOI&format=csv"
    
    add_few_elements = '&select='+','.join(
    ['toi','tid','tfopwg_disp','rastr','ra','decstr','dec','pl_orbper','pl_orbpererr1','pl_orbpererr2',\
     'pl_orbperlim','pl_trandurh','pl_trandurherr1','pl_trandurherr2','pl_trandurhlim','pl_rade',\
     'pl_radeerr1','pl_radeerr2','pl_radelim','pl_insol','pl_insolerr1','pl_insolerr2','pl_insollim',\
     'pl_eqt','pl_eqterr1','pl_eqterr2','pl_eqtlim','st_tmag','st_tmagerr1','st_tmagerr2','st_tmaglim',\
     'st_teff','st_tefferr1','st_tefferr2','st_tefflim','st_logg','st_loggerr1','st_loggerr2','st_logglim',\
     'st_rad','st_raderr1','st_raderr2','st_radlim&'])

    all_planets =  requests.get(default_query.split('&')[0] + add_few_elements + default_query.split('&')[1]) 
    
    all_planets = all_planets.text.split('\n')
    
    planets_df = pd.DataFrame(columns=all_planets[0].split(','), 
                              data = [i.split(',') for i in all_planets[1:-1]])
    planets_df = planets_df.replace(to_replace='', value=np.nan)
        
    planets_df.head()
    planets_df.to_csv(toiFpath, index=False)
    
    return toiFpath

def ExoFOP( forceDownload=False ):
    """
    Download TFOP information. 
    """
    ##date = str(datetime.date(datetime.now()))
    date = '2022-11-03'
    path = 'ExoFOP_'+date+'.csv'
    toiFpath = path
    if not forceDownload:
        if os.path.exists(f'{os.getcwd()}/{path}'):
            return toiFpath
    print( '\nDownloading list from ExoFOP website...' )
    query = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    print( 'Done.' )
    planets = requests.get( query ).text
    ofile = open( toiFpath, 'w' )
    ofile.write( planets )
    ofile.close()
    print( 'Saved:\n{0}\n'.format( toiFpath ) )
    return toiFpath

