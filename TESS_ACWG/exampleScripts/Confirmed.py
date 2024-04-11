import os, sys, pdb
from TESS_ACWG import downloadTargetLists, processTargetLists
from TESS_ACWG import surveyGrids, surveySetup
from TESS_ACWG import massRadiusFigure

ADIR = os.getcwd()

########################################################################
# Step 1: Download the target list from NASA Exoplanet Archive.

csvFpath = downloadTargetLists.targetsWithPublishedConfirmation( forceDownload=False )
CSV_FNAME = csvFpath
CSV_FPATH = os.path.join( ADIR, CSV_FNAME )

########################################################################
# Step 2: Process the csv file downloaded from NASA Exoplanet Archive.
if 1:
    confirmedPickle = processTargetLists.Confirmed( csvIpath=csvFpath, pklOdir=ADIR, \
                                                   forceDownload=False )
else:
    confirmedPickle = 'confirmedProperties.pkl'
    
########################################################################
# Step 3: Make the figures using the processed pickle file as input.
#figFpaths = surveyGrids.Confirmed( ipath=confirmedPickle, surveyGrid='ACWG', \
#                                   obsSample='PublishedMassesOnly', thresholdTSM='ACWG' )
survey = { 'surveyName':'ACWG', 'obsSample':'PublishedMassesOnly', 'framework':'ACWG', \
           'gridEdges':surveySetup.gridEdges, 'thresholdTSM':surveySetup.thresholdTSM, \
           'thresholdESM':surveySetup.thresholdESM, 'preCuts':surveySetup.preCutsConfirmed }
figFpaths = surveyGrids.Confirmed( ipath=confirmedPickle, survey=survey, SMFlag = 'ESM', HeatMap = False )

########################################################################
# Step 4: Make a plot comparing the assumed mass-radius relation to
# the measured masses and radii of the confirmed planets.
massRadiusFpath = massRadiusFigure.Confirmed( ipath=confirmedPickle )

