import os, sys, pdb
import pickle
import numpy as np
from TESS_ACWG import downloadTargetLists, processTargetLists
from TESS_ACWG import surveyGrids, Utils
from TESS_ACWG import massRadiusFigure 
from TESS_ACWG import surveySetup
ADIR = os.getcwd()

########################################################################
# Step 1: Download the target list from NASA Exoplanet Archive.

csvFpath = downloadTargetLists.targetsWithPublishedConfirmation( forceDownload=False )
CSV_FNAME = csvFpath
CSV_FPATH = os.path.join( ADIR, CSV_FNAME )

########################################################################
# Step 2: Process the csv file downloaded from NASA Exoplanet Archive.

confirmedPickle = processTargetLists.Confirmed( csvIpath=csvFpath, pklOdir=ADIR, \
                                                readExisting=False )

########################################################################
# Hack: Estimate masses for those with radius measurements but no mass
#       measurement. From this point on, confirmedProperties_MissingMassesAdded.pkl
#       contains all the information required for making the figure and ASCII
#       output. The 'obsSample' property is used to distinguish between those
#       planets with measured and estimated masses, via the corresponding
#       subroutine defined within surveySetup.py. Specifically, planets with
#       estimated masses are identified by those with NaNs for mass uncertainties.
#       These are also lumped in with those that have mass uncertainties but
#       a precision of <3sigma. 
ipath = 'confirmedProperties.pkl'
opath = 'confirmedProperties_MissingMassesAdded.pkl'
ifile = open( ipath, 'rb' )
z = pickle.load( ifile )
ifile.close()
ixs = np.isnan( z['allVals']['MpValME'] )*( np.isnan( z['allVals']['RpValRE'] )==False )
z['allVals']['MpValME'][ixs] = Utils.planetMassFromRadius( z['allVals']['RpValRE'][ixs], \
                                                           whichRelation='ExoArchive' )
                                                           #'Chen&Kipping2017' )
z['allVals']['TSM'][ixs] = Utils.computeTSM( z['allVals']['RpValRE'][ixs], z['allVals']['MpValME'][ixs], z['allVals']['RsRS'][ixs], z['allVals']['TeqK'][ixs], z['allVals']['Jmag'][ixs] )
z['allVals']['transitSignal'][ixs] = Utils.computeTransSignal( z['allVals']['RpRs'][ixs], z['allVals']['RpValSI'][ixs], z['allVals']['TeqK'][ixs], z['allVals']['MpValSI'][ixs] )
z['allVals']['Kamp'][ixs] = Utils.computeRVSemiAmp( z['allVals']['Pday'][ixs], z['allVals']['MpValME'][ixs], z['allVals']['MsMS'][ixs] )

ofile = open( opath, 'wb' )
pickle.dump( z, ofile )
ofile.close()
confirmedPickle = opath


########################################################################
# Step 3: Make the figures using the processed pickle file as input.
survey = { 'surveyName':'ACWG', 'obsSample':'NeedMasses', 'framework':'ACWG', \
           'gridEdges':surveySetup.gridEdges, 'thresholdTSM':surveySetup.thresholdTSM, \
           'thresholdESM':surveySetup.thresholdESM, \
           'thresholdTS':surveySetup.thresholdTranSignal, \
           'thresholdSE':surveySetup.thresholdSecEclipse, \
           'thresholdJmag':surveySetup.thresholdJmag, \
           'thresholdKmag':surveySetup.thresholdKmag, \
           'preCuts':surveySetup.preCutsConfirmed }

for obsSample in ['PublishedMassesOnly','NeedMasses']:
    survey['obsSample'] = obsSample
    for SMFlag in [ 'TSM', 'ESM' ]:  
        figFpaths = surveyGrids.Confirmed( ipath=confirmedPickle, survey=survey, \
                                           SMFlag=SMFlag, HeatMap=False )
        ASCII1, ASCII2 = surveyGrids.CreateASCII_Confirmed( ipath=confirmedPickle, \
                                                            survey=survey, SMFlag=SMFlag )

########################################################################
# Step 4: Make a plot comparing the assumed mass-radius relation to
# the measured masses and radii of the confirmed planets.
if 1:
    massRadiusFpaths = massRadiusFigure.Confirmed( ipath=confirmedPickle, weighting='None' )
    massRadiusFpaths = massRadiusFigure.Confirmed( ipath=confirmedPickle, weighting='TSM' )
    massRadiusFpaths = massRadiusFigure.Confirmed( ipath=confirmedPickle, weighting='ESM' )

