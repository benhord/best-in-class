import pdb, sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from . import Utils, downloadTargetLists, surveyGrids



"""
Module for pre-processing ASCII/csv files into pickle files that are 
then used as input by other modules such as survey.py.

The top-level routines are:
- Confirmed()
- candidateTESS()
- predictedTESS()
Make sure IPATH variables are set correctly then run the above routines.
      
For instructions on the process to follow when downloading from the 
NASA Exoplanet Archive, see the docstrings for the following routines:
- readRawConfirmedNExScI()

"""

IPATH_BARCLAY2018_V2 = 'datafileBarclayTESS_v2.txt'
IPATH_BARCLAY2018_V1 = 'detected_planets.csv'

TEFFK_SUN = 5800

def Confirmed( csvIpath='', pklOdir='', readExisting=False ):
    zAll, zMissing, dateStr = readConfirmedNExScI( csvIpath, readExisting=readExisting )
    zOut = { 'missingProperties':zMissing, 'allVals':zAll, 'dateStr':dateStr }
    oname = 'confirmedProperties.pkl'
    odir = os.getcwd()
    opath = os.path.join( odir, oname )
    #if not forceDownload:
    if readExisting==True:
        if os.path.exists( opath ):
            pklAge = os.path.getmtime(opath)
            if (time.time() - pklAge)/3600 < 24:
                ostr = '\n{0}\nFile exists and has been updated within last 24hrs'\
                       .format( opath )
                #ostr += ' (Set forceDownload=True to force update)\n'
                ostr += ' (Set readExisting=False to force update)\n'
                print( ostr )
                return opath
    ofile = open( opath, 'wb' )
    pickle.dump( zOut, ofile )
    ofile.close()
    print( '\nSaved:\n{0}\n'.format( opath ) )
    return opath

def TOIs( csvIpath='', pklOdir='', forceDownload=False ):
    z, zMissing, dateStr = readTOIsNExScI( csvIpath, forceDownload=forceDownload )
    zAll = checkTOIsTESSCP( z, forceDownload = forceDownload )
    #zAll = z
    zOut = { 'missingProperties':zMissing, 'allVals':zAll, 'dateStr':dateStr }
    oname = 'toiProperties.pkl'
    opath = os.path.join( pklOdir, oname )
    if not forceDownload:
        if os.path.exists(opath):
            pklAge = os.path.getmtime(opath)
            if (time.time() - pklAge)/3600 < 24:
                ostr = '\n{0}\nFile exists and has been updated within last 24hrs'\
                       .format( opath )
                ostr += ' (Set forceDownload=True to force update)\n'
                print( ostr )
                return opath
    ofile = open( opath, 'wb' )
    pickle.dump( zOut, ofile )
    ofile.close()
    print( '\nSaved:\n{0}\n'.format( opath ) )

    return opath

def predictedTESS():
    """
    There are two versions of the Barclay predictions.
    1. Published.
       https://figshare.com/articles/dataset/...
       ...TESS_Extended_Mission_Yield_Simulations/11775081
       - Table 2 from Barclay, Pepper, Quintana (2018).

    2. Updated 
       https://figshare.com/articles/dataset/...
       ...TESS_Extended_Mission_Yield_Simulations/11775081
       - Includes extended mission.
       - Uses a 'conservative' detection threshold.
       - Uses Petigura 2018 for FGK occurrence rate.
       - Uses TIC 8 (based on Gaia)
    """
    z = readRawBarclayLines_v2()
    n = len( z['RpValRE'] )
 
    z['MpValME'] = Utils.planetMassFromRadius( z['RpValRE'], \
                                               whichRelation='ExoArchive' )
    z = addTeq( z )
    z['TSM'] = Utils.computeTSM( z['RpValRE'], z['MpValME'], z['RsRS'], \
                                 z['TeqK'], z['Jmag'] )
    z['ESM'] = Utils.computeESM( z['TeqK'], z['RpRs'], z['TstarK'], z['Kmag'] )
    odir = os.path.dirname( __file__ )
    oname = 'predictedProperties_v2.pkl'
    opath = os.path.join( odir, oname )
    ofile = open( opath, 'wb' )
    pickle.dump( z, ofile )
    ofile.close()
    print( '\nSaved:\n{0}\n'.format( opath ) )
    return opath

#################################################################################

def readRawBarclayLines_v2():
    """
    Reads the Barclay predicted planet data and returns as a dictionary.
    All detections satisfy the conservative detection criteria.
    """
    # updated version incl. extended mission (2020)
    idir = os.path.dirname( __file__ )
    ipath = os.path.join( idir, 'datafileBarclayTESS_v2.txt' )
    ifile = open( ipath, 'r')

    z = {}
    z['RAdeg'] = []
    z['Decdeg'] = []
    z['cad2min'] = []
    z['Vmag'] = []
    z['Kmag'] = []
    z['Jmag'] = []
    z['RsRS'] = []
    z['MsMS'] = []
    z['TstarK'] = []
    z['subGiants'] = []
    z['Pday'] = []
    z['RpValRE'] = []
    z['aRs'] = []
    z['RpRs'] = []
    z['b'] = []
    z['T14hr'] = []
    z['Insol'] = []
    z['conservativeDetected'] = []

    n=0
    for line in ifile:
        l = line.split()
        if l[0] != '#' and l[0][0] != '#':

            #Check correct formatting/number of attributes
            n += 1
            if len(l) != 32 and len(l) != 33:
                raise Exception('Predicted planet {0} has incorrect number of attributes'.format(str(n))) 
            
            z['RAdeg'] += [l[1]]
            z['Decdeg'] += [l[2]]
            z['cad2min'] += [l[6]]
            z['Vmag'] += [l[10]]
            z['Kmag'] += [l[11]]
            z['Jmag'] += [l[12]]
            z['RsRS'] += [l[14]]
            z['MsMS'] += [l[15]]
            z['TstarK'] += [l[16]]
            z['subGiants'] += [l[18-33]]
            z['Pday'] += [l[21-33]]
            z['RpValRE'] += [l[22-33]]
            z['aRs'] += [l[24-33]]
            z['RpRs'] += [l[26-33]]
            z['b'] += [l[27-33]]
            z['T14hr'] += [l[28-33]]
            z['Insol'] += [l[-3]]
            z['conservativeDetected'] += [int(l[-13])] 

    ifile.close()

    #Correct the data types
    for key in z:
        for i in range(len(z[key])):
            if key not in ['cad2min', 'subGiants','conservativeDetected']:
                z[key][i] = float(z[key][i])
            else:
                z[key][i] = int(z[key][i])

    # UPDATE: Just store conservative detections as an additional array
    #Cut out undetected according to the Barclay et al 'conservative model'
    # See Section 4.2 of Barclay, Pepper, Quintana (2020).
    #ixs = []
    #for i in range(len(detected)):
    #    if int(detected[i]):
    #        ixs.append(i)
    #z1 = {}
    #for key in z:
    #    z1[key] = np.array(z[key])[ixs]
    z1 = {}
    for key in z:
        z1[key] = np.array(z[key])

    return z1

def convertStrToFloat( string ):
    if string!='':
        outp = float( string )
    else:
        outp = np.nan
    return outp

def getDateStr( fpath, whichList='Confirmed' ):
    fname = os.path.basename( fpath )
    if whichList=='Confirmed':
        prefix = 'PS_'
    elif whichList=='TOIs':
        prefix = 'TOI_'
    elif whichList == 'Predicted':
        prefix = 'Predicted'
    n = len( prefix )
    ix0 = n+fname.rfind( prefix )
    ix1 = ix0+10
    dateStr = fname[ix0:ix1].replace( '.', '/' )
    return dateStr

def readConfirmedNExScI( fpath, readExisting=False ):
    dateStr = getDateStr( fpath, whichList='Confirmed' )
    zRaw = readRawConfirmedNExScI( fpath, readExisting=readExisting )
    #if not forceDownload:
    if readExisting==True:
        if 'allVals' in zRaw:
            z = pickle.load(open('confirmedProperties.pkl', 'rb'))
            zMissing, zAll, dateStr = [i for i in z.values()]
            return zAll, zMissing, dateStr
    
    zAll, zMissing = processRawConfirmed( zRaw )
    zAll = addMissingInsol( zAll )
    zAll = addUnits( zAll )
    zAll = addGravPl( zAll )
    zAll = addTeq( zAll )
    zAll['TSM'] = Utils.computeTSM( zAll['RpValRE'], zAll['MpValME'], \
                                    zAll['RsRS'], zAll['TeqK'], zAll['Jmag'] )
    zAll['ESM'] = Utils.computeESM( zAll['TeqK'], zAll['RpRs'], \
                                    zAll['TstarK'], zAll['Kmag'] )
    zAll['eclipseDepth'] = Utils.computeEclipseDepth( zAll['TeqK'], \
                                                      zAll['RpRs'], \
                                                      zAll['TstarK'] )
    zAll['transitSignal'] = Utils.computeTransSignal( zAll['RpRs'], \
                                                      zAll['RpValRE'], \
                                                      zAll['TeqK'], \
                                                      zAll['MpValME'] )
    zAll['Kamp'] = Utils.computeRVSemiAmp( zAll['Pday'], zAll['MpValME'], zAll['MsMS'] )
    return zAll, zMissing, dateStr

def readTOIsNExScI( fpath, forceDownload=False ):
    dateStr = getDateStr( fpath, whichList='TOIs' )
    zRaw = readRawTOIsNExScI( fpath, forceDownload=forceDownload )
    zAll = zRaw

    #print(zAll)
    
    if not forceDownload:
        if 'allVals' in zAll:
            z = pickle.load(open('toiProperties.pkl', 'rb'))
            zMissing, zAll, dateStr = [i for i in z.values()]
            return zAll, zMissing, dateStr
    
    zAll['MpValME'] = Utils.planetMassFromRadius( zAll['RpValRE'], \
                                                  whichRelation='ExoArchive' )
    
    zAll['Jmag'] = Utils.JHKVmags(zAll['TICID'])['Jmag']
    zAll['Hmag'] = Utils.JHKVmags(zAll['TICID'])['Hmag']
    zAll['Kmag'] = Utils.JHKVmags(zAll['TICID'])['Kmag']
    zAll['Vmag'] = Utils.JHKVmags(zAll['TICID'])['Vmag']
    zAll['Imag'] = Utils.JHKVmags(zAll['TICID'])['Imag']

    zAll['MsMS'] = Utils.computeStellarMass( zAll['RsRS'], zAll['loggstarCGS'])

    zAll['TeqK'], zAll['aRs'] = Utils.TeqK_Kempton( zAll['Pday'], zAll['MsMS'], \
                                                    zAll['TstarK'], zAll['RsRS'])
    #TeqK computed as in Kempton. Pulled TeqK values stored in TeqK_exofop

    zAll['TSM'] = Utils.computeTSM( zAll['RpValRE'], zAll['MpValME'], \
                                    zAll['RsRS'], zAll['TeqK'], zAll['Jmag'] )
    zAll['ESM'] = Utils.computeESM( zAll['TeqK'], zAll['RpRs'], \
                                    zAll['TstarK'], zAll['Kmag'] )
    zAll['eclipseDepth'] = Utils.computeEclipseDepth( zAll['TeqK'], \
                                                      zAll['RpRs'], \
                                                      zAll['TstarK'] )
    zAll['transitSignal'] = Utils.computeTransSignal( zAll['RpRs'], \
                                                      zAll['RpValRE'], \
                                                      zAll['TeqK'], \
                                                      zAll['MpValME'] )
    
    zAll['Kamp'] = Utils.computeRVSemiAmp( zAll['Pday'], zAll['MpValME'], zAll['MsMS'] )
    
    zMissing = {}
    for k in ['TSM','ESM']:
        zMissing[k] = zAll['planetName'][np.isfinite( zAll[k] )==False]
    return zAll, zMissing, dateStr

def addMissingInsol( z ):
    """
    Relations taken from this page:
    https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html
    """
    TsK = z['TstarK']
    RsRS = z['RsRS']
    LsLS = ( RsRS**2. )*( ( TsK/TEFFK_SUN )**4. )
    Insol = LsLS*( ( 1./z['aAU'] )**2. )
    ixs = ( np.isfinite( z['Insol'] )==False )
    z['Insol'][ixs] = Insol[ixs]

    return z
    
def addUnits( z ):
    
    # Stellar masses and radii:
    z['MsSI'] = z['MsMS']*Utils.MSUN_SI
    z['RsSI'] = z['RsRS']*Utils.RSUN_SI
    
    # Planet radii:
    z['RpValSI'] = z['RpValRE']*Utils.REARTH_SI
    z['RpLowErrSI'] = z['RpLowErrRE']*Utils.REARTH_SI
    z['RpUppErrSI'] = z['RpUppErrRE']*Utils.REARTH_SI
    z['RpValRJ'] = z['RpValSI']/Utils.RJUP_SI
    z['RpLowErrRJ'] = z['RpLowErrSI']/Utils.RJUP_SI
    z['RpUppErrRJ'] = z['RpUppErrSI']/Utils.RJUP_SI

    # Planet masses:
    z['MpValSI'] = z['MpValME']*Utils.MEARTH_SI
    z['MpLowErrSI'] = z['MpLowErrME']*Utils.MEARTH_SI
    z['MpUppErrSI'] = z['MpUppErrME']*Utils.MEARTH_SI
    z['MpValMJ'] = z['MpValSI']/Utils.MJUP_SI
    z['MpLowErrMJ'] = z['MpLowErrSI']/Utils.MJUP_SI
    z['MpUppErrMJ'] = z['MpUppErrSI']/Utils.MJUP_SI

    return z


def addGravPl( z ):
    z['gpSI'] = Utils.GRAV_SI*z['MpValSI']/( z['RpValSI']**2 )
    return z
    
def addTeq( z ):
    # Equation 3 from Kempton et al (2018):
    z['TeqK'] = Utils.calcTeqK( z['TstarK'], z['aRs'] )
    return z



def processRawConfirmed( zRaw ):
    # First check the number of unique planets:
    p = np.unique( zRaw['planetName'] )
    n = len( p )
    print( '\n{0:.0f} unique planets identified.'.format( n ) )
    
    # Loop over each planet:
    z = {}
    for i in range( n ):
        zi = extractProperties( zRaw, p[i] )
        if i==0:
            z['planetName'] = [ p[i] ]
            properties = list( zi.keys() )
            for k in properties:
                z[k] = [ zi[k] ]
        else:
            z['planetName'] += [ p[i] ]
            for k in properties:
                z[k] += [ zi[k] ]
    for k in properties:
        z[k] = np.array( z[k] )
    z['planetName'] = np.array( z['planetName'], dtype=str )
    z['RA'] = np.array( z['RA'], dtype=str )
    z['Dec'] = np.array( z['Dec'], dtype=str )
    z = correctaRs( z )
    z = correctRpRs( z )
    z = correctImpact( z )
    # Assume circular orbits when eccentricity unknown:
    z['ecc'][np.isfinite( z['ecc'] )==False] = 0
    z = correctT14hr( z )
    
    zMissing = {}
    for k in ['b','aAU','RpRs','TstarK','Vmag','Jmag','RpValRE']:
        nTotal = len( z[k] )
        nMissing = nTotal-np.sum( np.isfinite( z[k] ) )
        ixs = np.isfinite( z[k] )==False
        zMissing[k] = p[ixs]
        print( '\n\n{0} --> missing {1:.0f}'.format( k, nMissing ) )
        for i in p[ixs]:
            print( i )
    return z, zMissing


def correctaRs( z ):
    # Missing aRs values:
    ixs = ( np.isfinite( z['aRs'] )==False )
    aSI = z['aAU'][ixs]*Utils.AU_SI
    RsSI = z['RsRS'][ixs]*Utils.RSUN_SI
    z['aRs'][ixs] = aSI/RsSI
    # Missing aAU values:
    ixs = ( np.isfinite( z['aAU'] )==False )*( np.isfinite( z['aRs'] )==True )
    aSI = z['aRs'][ixs]*( z['RsRS'][ixs]*Utils.RSUN_SI )
    z['aAU'][ixs] = aSI/Utils.AU_SI
    return z

def correctRpRs( z ):
    ixs = ( np.isfinite( z['RpRs'] )==False )
    RpSI = z['RpValRE'][ixs]*Utils.REARTH_SI
    RsSI = z['RsRS'][ixs]*Utils.RSUN_SI
    z['RpRs'][ixs] = RpSI/RsSI
    z = Utils.fixAnomalousRpRs( z )
    return z
    
def correctT14hr( z ):

    ixs1 = ( np.isfinite( z['T14hr'] )==False )
    ixs2 = ( z['b']-z['RpRs']<1 ) # transiting
    ixs = ixs1*ixs2
    PSI = z['Pday'][ixs]*24*60*60
    b = z['b'][ixs]
    sini = np.sin( np.deg2rad( z['inclDeg'][ixs] ) )
    RpRs = z['RpRs'][ixs]
    aRs = z['aRs'][ixs]
    x = np.sqrt( ( ( 1+RpRs )**2.)-( b**2. ) )/( aRs*sini )
    T14SI = (PSI/np.pi)*np.arcsin( x )
    z['T14hr'][ixs] = T14SI/( 60*60 )
    return z


def correctImpact( z ):
    # Missing b values:
    ixs = ( np.isfinite( z['b'] )==False )
    aSI = z['aAU'][ixs]*Utils.AU_SI
    RsSI = z['RsRS'][ixs]*Utils.RSUN_SI
    inclRad = np.deg2rad( z['inclDeg'][ixs] )
    z['b'][ixs] = ( aSI/RsSI )*np.cos( inclRad )
    # Missing inclination values:
    ixs = ( np.isfinite( z['inclDeg'] )==False )*( np.isfinite( z['b'] )==True )\
          *( np.isfinite( z['aRs'] )==True )
    cosi = z['b'][ixs]/z['aRs'][ixs]
    z['inclDeg'][ixs] = np.rad2deg( np.arccos( cosi ) )
    return z


def extractProperties( zRaw, planetName ):
    """
    This routine is applied to one planet at a time, which
    may have multiple property entries on Exoplanet Archive.
    """
    
    
    # Properties without uncertainties are props1:
    props1 = [ 'Pday', 'aAU', 'ecc', 'inclDeg', 'b', 'distParsec', \
               'Insol', 'T14hr', 'aRs', 'TstarK', 'MsMS', \
               'Vmag', 'Jmag', 'Hmag', 'Kmag', \
               'discoveryMethod', 'discoveryFacility', \
               'discoveryYear', 'discoveredByTESS' ]
    
    # Properties with uncertainties are props2:
    props2 = [ 'RpValRE', 'RpUppErrRE', 'RpLowErrRE', \
               'MpValME', 'MpUppErrME', 'MpLowErrME', \
               'RpRs', 'RpRsUppErr', 'RpRsLowErr', \
               'TstarK', 'TstarKUppErr', 'TstarKLowErr', \
               'RsRS', 'RsRSUppErr', 'RsRSLowErr' ]
    nAll = len( zRaw['planetName'] )
    ixs = np.arange( nAll )[zRaw['planetName']==planetName]
    nPlanet = len( ixs )
    zPlanet = {}
    for k in list( zRaw.keys() ):
        zPlanet[k] = zRaw[k][ixs]

    # Fill properties with default values:
    ixDefault = ( zPlanet['defaultFlag']==1 )
    zOut = {}
    for k in props1:
        if ( k=='discoveryYear' )+( k=='discoveredByTESS' ):
            zOut[k] = int( zPlanet[k][ixDefault] )
        elif ( k=='discoveryMethod' )+( k=='discoveryFacility' ):
            zOut[k] = str( zPlanet[k][ixDefault][0] )
        else:
            zOut[k] = np.abs( float( zPlanet[k][ixDefault] ) )
    for k in props2:
        zOut[k] = np.abs( float( zPlanet[k][ixDefault] ) )
        #if planetName=='EPIC 211945201 b':
        #    print( k, zOut[k], zPlanet[k][ixDefault] )
    #if planetName=='EPIC 211945201 b':
    #    pdb.set_trace()
    zOut['RA_deg'] = float( zPlanet['RA_deg'][ixDefault][0] )
    zOut['Dec_deg'] = float( zPlanet['Dec_deg'][ixDefault][0] )
    zOut['RA'] = str( zPlanet['RA'][ixDefault][0] )
    zOut['Dec'] = str( zPlanet['Dec'][ixDefault][0] )
    ixOthers = np.arange( nPlanet )[zPlanet['defaultFlag']==0]
    nOthers = len( ixOthers )
    if nOthers>0:
        # For the properties without uncertainties (props1), if the default
        # parameter set does not include a value, try to insert a value from
        # a non-default parameter set instead:
        for k in props1:
            if type( zOut[k] )==str: # np.isfinite() fails if string passed as input
                continue
            elif np.isfinite( zOut[k] )==False:
                for i in range( nOthers ):
                    if np.isfinite( zPlanet[k][ixOthers[i]] ):
                        zOut[k] = float( zPlanet[k][ixOthers[i]] )
                        break
        # Do similar for the properties that require uncertainties (props2):
       
        z = [ [ 'RpValRE', 'RpUppErrRE', 'RpLowErrRE' ], \
              [ 'MpValME', 'MpUppErrME', 'MpLowErrME' ], \
              [ 'RpRs', 'RpRsUppErr', 'RpRsLowErr' ], \
              [ 'TstarK', 'TstarKUppErr', 'TstarKLowErr' ] ] #change?
        if ( zOut['MpLowErrME']<0 ):
            print( '\nA wtf' )
            pdb.set_trace()
        for k in z:
            zOut = fixValuesWithUncertainties( zOut, zPlanet, k, ixOthers, \
                                               planetName, mostPreciseAlways=True )
    return zOut

def fixValuesWithUncertainties( zAll, zPlanet, k, ixOthers, planetName, \
                                mostPreciseAlways=True ):
    """
    Subroutine for identifying planets with mass and radius values not included
    in the default parameter set, then checking to see if these values can be
    taken from a non-default parameter set instead.
    """
    nOthers = len( ixOthers )
    cMed = np.isfinite( zAll[k[0]] )==False
    cUpp = np.isfinite( zAll[k[1]] )==False
    cLow = np.isfinite( zAll[k[2]] )==False
    if ( cMed+cUpp+cLow ):
        # Case 1. Resort to non-default values if default value is NaN.
        medVal = np.abs( zPlanet[k[0]][ixOthers] )
        uncsUpp = np.abs( zPlanet[k[1]][ixOthers] )
        uncsLow = np.abs( zPlanet[k[2]][ixOthers] )
        uncs = np.mean( np.column_stack( [ uncsLow, uncsUpp ] ), axis=1 )
        ixs = np.isfinite( uncs )
        n = int( ixs.sum() )
        if n>0:
            ixPrecise = np.arange( n )[np.argmin(uncs[ixs])]
            zAll[k[0]] = float( medVal[ixs][ixPrecise] )
            zAll[k[1]] = float( uncsUpp[ixs][ixPrecise] )
            zAll[k[2]] = float( uncsLow[ixs][ixPrecise] )
            #zAll[k[0]] = float( zPlanet[k[0]][ixOthers[ixs][ixPrecise]] )
            #zAll[k[1]] = float( zPlanet[k[1]][ixOthers[ixs][ixPrecise]] )
            #zAll[k[2]] = float( zPlanet[k[2]][ixOthers[ixs][ixPrecise]] )
    elif mostPreciseAlways:
        # Case 2. Default value is not NaN, but priority is most precise value.
        medVal = np.abs( np.concatenate( [ [zAll[k[0]]], zPlanet[k[0]][ixOthers] ] ) )
        uncsUpp = np.abs( np.concatenate( [ [zAll[k[1]]], zPlanet[k[1]][ixOthers] ] ) )
        uncsLow = np.abs( np.concatenate( [ [zAll[k[2]]], zPlanet[k[2]][ixOthers] ] ) )
        uncs = np.mean( np.column_stack( [ uncsLow, uncsUpp ] ), axis=1 )
        ixs = np.isfinite( medVal )*np.isfinite( uncs )
        n = int( ixs.sum() )
        if n>0:
            ixPrecise = np.arange( n )[np.argmin(uncs[ixs])]
            zAll[k[0]] = float( medVal[ixs][ixPrecise] )
            zAll[k[1]] = float( uncsUpp[ixs][ixPrecise] )
            zAll[k[2]] = float( uncsLow[ixs][ixPrecise] )
    else:
        # Case 3. Priority is default value, which is not NaN, so no need to change.
        pass

    return zAll
        
        
def readRawConfirmedNExScI( csvIpath, readExisting=False ):
    """
    Instructions for downloading table from NASA Exoplanet Archive:
    1. Remove condition 'Default parameter set = 1'.
    2. Remove columns: 
         - Number of stars
         - Number of planets
         - Discovery method
         - Solution type
         - Controversial flag
         - Planet parameter reference
         - Equilibrium temperature
         - Data show TTVs
         - Stellar parameter reference
         - Spectral type
         - Stellar metallicity
         - Stellar metallicity ratio
         - Stellar surface gravity
         - System parameter reference
         - RA (sexagesimal)
         - Dec (sexagesimal)
         - Gaia magnitude
         - Date of last update
         - Planet parameter reference publication
         - Release date
    3. Add columns:
         - Detected by transits
         - Inclination
         - Impact parameter
         - Transit duration
         - Ratio of semi-major axis to stellar radius
         - Ratio of planet to stellar radius
         - RA (deg)
         - Dec (deg)
         - J magnitude
         - H magnitude
    4. Set condition 'Detected by transits = 1'.
    5. Download table as CSV. Make sure 'Values only' is *not* checked.
    """
    if readExisting==True:
        fpath = f'{os.getcwd()}/confirmedProperties.pkl'
        if os.path.exists(fpath):
            pklAge = os.path.getmtime(fpath)
            if (time.time() - pklAge)/3600 < 24:
                z = pickle.load(open('confirmedProperties.pkl', 'rb'))
                return z
    
    print( '\nReading NExScI table of confirmed planets:\n{0}'.format( csvIpath ) )
    t = np.genfromtxt( csvIpath, dtype=str, delimiter=',', invalid_raise=False )
    print( '\nMessage from readRawConfirmedNExScI() routine:' )
    print( 'NOTE: Some rows may have formatting issues and will not be read.' )
    print( 'These would be flagged here. No solution to this currently.\n\n' )
    cols = t[0,:]
    
    z = {}
    
    z['planetName'] = t[1:,cols=='pl_name'].flatten()
    z['RA'] = t[1:,cols=='rastr'].flatten()
    z['Dec'] = t[1:,cols=='decstr'].flatten()
    z['RA_deg'] = t[1:,cols=='ra'].flatten()
    z['Dec_deg'] = t[1:,cols=='dec'].flatten()

    z['discoveryMethod'] = t[1:,cols=='discoverymethod'].flatten()
    z['discoveryFacility'] = t[1:,cols=='disc_facility'].flatten()
    z['discoveryYear'] = t[1:,cols=='disc_year'].flatten()
    z['defaultFlag'] = t[1:,cols=='default_flag'].flatten()
    z['Pday'] = t[1:,cols=='pl_orbper'].flatten()
    z['aAU'] = t[1:,cols=='pl_orbsmax'].flatten()
    z['distParsec'] = t[1:,cols=='sy_dist'].flatten()

    z['RpValRE'] = t[1:,cols=='pl_rade'].flatten()
    z['RpUppErrRE'] = t[1:,cols=='pl_radeerr1'].flatten()
    z['RpLowErrRE'] = t[1:,cols=='pl_radeerr2'].flatten()
    
    z['MpValME'] = t[1:,cols=='pl_masse'].flatten()
    z['MpUppErrME'] = t[1:,cols=='pl_masseerr1'].flatten()
    z['MpLowErrME'] = t[1:,cols=='pl_masseerr2'].flatten()
    z['MpProvenance'] = t[1:,cols=='pl_bmassprov'].flatten()

    z['ecc'] = t[1:,cols=='pl_orbeccen'].flatten()
    z['inclDeg'] = t[1:,cols=='pl_orbincl'].flatten()
    z['b'] = t[1:,cols=='pl_imppar'].flatten()
    z['T14hr'] = t[1:,cols=='pl_trandur'].flatten()
    z['aRs'] = t[1:,cols=='pl_ratdor'].flatten()
    z['RpRs'] = t[1:,cols=='pl_ratror'].flatten()
    z['RpRsUppErr'] = t[1:,cols=='pl_ratrorerr1'].flatten()
    z['RpRsLowErr'] = t[1:,cols=='pl_ratrorerr2'].flatten()
    z['Insol'] = t[1:,cols=='pl_insol'].flatten()
    
    z['TstarK'] = t[1:,cols=='st_teff'].flatten()
    z['TstarKUppErr'] = t[1:,cols=='st_tefferr1'].flatten()
    z['TstarKLowErr'] = t[1:,cols=='st_tefferr2'].flatten()
    z['RsRS'] = t[1:,cols=='st_rad'].flatten()
    z['RsRSUppErr'] = t[1:,cols=='st_raderr1'].flatten()
    z['RsRSLowErr'] = t[1:,cols=='st_raderr2'].flatten()
    z['MsMS'] = t[1:,cols=='st_mass'].flatten()
    
    z['Vmag'] = t[1:,cols=='sy_vmag'].flatten()
    z['Jmag'] = t[1:,cols=='sy_jmag'].flatten()
    z['Hmag'] = t[1:,cols=='sy_hmag'].flatten()
    z['Kmag'] = t[1:,cols=='sy_kmag'].flatten()

    def convertMissing( zarr ):
        # Old version:
        #zarrOut = np.ones( len( zarr ) )
        #ixs = ( zarr!='' )
        #zarrOut[ixs] = np.abs( np.array( zarr[ixs], dtype=float ) )
        #ixs = ( zarr=='' )
        #zarrOut[ixs] = np.nan
        # New version from TOI routine:
        zarrOut = np.ones( len( zarr ) )
        ixs = ( zarr!='' )
        zarrOut[ixs] = np.array( zarr[ixs], dtype=float ) 
        ixs = ( zarr=='' )
        zarrOut[ixs] = np.nan
        return zarrOut

    # Add a convenient binary flag for TESS discoveries:
    n = len( z['discoveryFacility'] )
    z['discoveredByTESS'] = np.zeros( n )
    strTESS = 'Transiting Exoplanet Survey Satellite (TESS)'
    ixs = ( z['discoveryFacility']==strTESS )
    z['discoveredByTESS'][ixs] = 1
    for k in list( z.keys() ):
        if ( k=='planetName' )+( k=='MpProvenance' )\
           +( k=='discoveryFacility' )+( k=='discoveryMethod' ):
            continue
        elif ( k=='RA' )+( k=='Dec' ):
            z[k] = np.array( z[k], dtype=str )
        elif ( k=='defaultFlag' )+( k=='discoveryYear' )+( k=='discoveredByTESS' ):
            z[k] = np.array( z[k], dtype=int )
        else:
            z[k] = convertMissing( z[k] )
            z[k] = np.array( z[k], dtype=float )
    
    return z


def readRawTOIsNExScI( fpath, forceDownload=False ):
    """
    """
    if not forceDownload:
        ipath = f'{os.getcwd()}/toiProperties.pkl'
        if os.path.exists(ipath):
            pklAge = os.path.getmtime(ipath)
            if (time.time() - pklAge)/3600 < 24:
                z = pickle.load(open('toiProperties.pkl', 'rb'))
                return z
    
    print( '\nReading NExScI table of TOIs:\n{0}'.format( fpath ) )
    t = np.genfromtxt( fpath, dtype=str, delimiter=',', invalid_raise=False )
    print( '\nMessage from readRawTOIsNExScI() routine:' )
    print( 'NOTE: Some rows may have formatting issues and will not be read.' )
    print( 'These would be flagged here. No solution to this currently.\n\n' )
    cols = t[0,:]
    
    z = {}
    z['planetName'] = np.array( t[1:,cols=='toi'].flatten(), dtype='<U20' )
    z['TICID'] = np.array( t[1:,cols=='tid'].flatten(), dtype='<U20' )
    z['RA_deg'] = t[1:,cols=='ra'].flatten()
    z['RA'] = np.array( t[1:,cols=='rastr'].flatten(), dtype='<U20' )
    z['Dec_deg'] = t[1:,cols=='dec'].flatten()
    z['Dec'] = np.array( t[1:,cols=='decstr'].flatten(), dtype='<U20' )
    z['Insol'] = t[1:,cols=='pl_insol'].flatten()
    z['Pday'] = t[1:,cols=='pl_orbper'].flatten()
    z['TeqK_exofop'] = t[1:,cols=='pl_eqt'].flatten() #TeqK taken from Exoplanet Archive rather than calculated as in Kempton et al.
    z['RpValRE'] = t[1:,cols=='pl_rade'].flatten()
    z['RpUppErrRE'] = t[1:,cols=='pl_radeerr1'].flatten()
    z['RpLowErrRE'] = t[1:,cols=='pl_radeerr2'].flatten()
    z['T14hr'] = t[1:,cols=='pl_trandurh'].flatten()
    z['TstarK'] = t[1:,cols=='st_teff'].flatten()
    z['loggstarCGS'] = t[1:,cols=='st_logg'].flatten()
    z['RsRS'] = t[1:,cols=='st_rad'].flatten()
    z['RsUppErrRS'] = t[1:,cols=='st_raderr1'].flatten()
    z['RsLowErrRS'] = t[1:,cols=='st_raderr2'].flatten()
    z['Tmag'] = t[1:,cols=='st_tmag'].flatten()

    # TODO = Request JHK mags are added.
    TFOP = t[1:,cols=='tfopwg_disp'].flatten()
    #this is super hacked to include original targets that have since been
    #  shown to be false positive and confirmed BH 6/22/23
    #print(len(TFOP))
    #ixs = (TFOP!='AA')#( TFOP!='KP' )*( TFOP!='FP' )*( TFOP!='FA' )
    ixs = ( TFOP!='KP' )*( TFOP!='FP' )*( TFOP!='FA' )
    for k in list( z.keys() ):
        z[k] = z[k][ixs]
    TFOP = TFOP[ixs]
    #print(len(TFOP))
    #blah
    n = len( z['planetName'] )
    for i in range( n ):
        if TFOP[i]=='':
            TFOP[i] = 'TFOP?'
        z['planetName'][i] = 'TOI-{0}({1})'.format( z['planetName'][i], TFOP[i] ) #BH 11/2/22
        #z['planetName'][i] = 'TOI-{0}'.format( z['planetName'][i])
        #print('HERE')
    def convertMissing( zarr ):
        zarrOut = np.ones( len( zarr ) )
        ixs = ( zarr!='' )
        zarrOut[ixs] = np.array( zarr[ixs], dtype=float ) 
        ixs = ( zarr=='' )
        zarrOut[ixs] = np.nan
        return zarrOut

    for k in list( z.keys() ):
        if ( k=='planetName' or k =='TICID' or k =='RA' or k == 'Dec' ):
            continue
        else:
            z[k] = convertMissing( z[k] )
            z[k] = np.array( z[k], dtype=float )

    z['RpValRJ'] = z['RpValRE']*( Utils.REARTH_SI/Utils.RJUP_SI )
    z['RpUppErrRJ'] = z['RpUppErrRE']*( Utils.REARTH_SI/Utils.RJUP_SI )
    z['RpLowErrRJ'] = z['RpLowErrRE']*( Utils.REARTH_SI/Utils.RJUP_SI )
    z['RpRs'] = ( z['RpValRE']*Utils.REARTH_SI )/( z['RsRS']*Utils.RSUN_SI )
    
    return z

def checkTOIsTESSCP ( zIN, forceDownload ):
               
    TOI_TICID = zIN['TICID']
    
    CP_TICIDpath = downloadTargetLists.targetsConfirmedTESS( forceDownload = forceDownload ) 
    ADIR = os.getcwd()
    ipath = os.path.join( ADIR, CP_TICIDpath )
    print('\nSaved:\n{0}\n'.format( ipath ))

    if not os.path.isfile(ipath):
        raise Exception("TESS CP TICID file not found")

    t = np.genfromtxt( ipath, dtype=str, delimiter=',', invalid_raise=False )

    CP_TICID = [CP[4::] for CP in t[1:]]
    
    ixs = []
    for i in range(len(TOI_TICID)):
        inCP = False
        for CP in CP_TICID:
            if TOI_TICID[i] == CP:
                inCP = True
                break
        if not inCP:
            ixs.append(i)
    
    zOut = {}
    for key in zIN:
        inList = zIN[key]
        zOut[key] = np.array([inList[n] for n in range(len(inList)) if n in ixs])
    print('Reading {0}/{1} TOIs not listed as a TESS Confirmed Planet'.format(len(zOut['TICID']), len(TOI_TICID)))

    return zOut     
