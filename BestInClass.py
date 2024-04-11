import os, sys, pdb
import pickle
import numpy as np
from TESS_ACWG import downloadTargetLists, processTargetLists
from TESS_ACWG import surveyGrids, Utils
from TESS_ACWG import massRadiusFigure 
from TESS_ACWG import surveySetup
ADIR = os.getcwd()

#############################################################################
# Step 1: Assume that Confirmed.py and TOIs.py have already been run. Specify
#         the file paths to the output generated from those scripts here.

confirmedPickle = 'confirmedProperties_MissingMassesAdded.pkl'
toiPickle = 'toiProperties.pkl'


########################################################################
# Step 2: Make the figures using the Confirmed and TOI pickle files as input.
survey = { 'surveyName':'ACWG', 'obsSample':'BestInClass', 'framework':'BestInClass', \
           'gridEdges':surveySetup.gridEdges, 'thresholdTSM':surveySetup.thresholdTSM, \
           'thresholdESM':surveySetup.thresholdESM, \
           'thresholdTS':surveySetup.thresholdTranSignal, \
           'thresholdSE':surveySetup.thresholdSecEclipse, \
           'thresholdJmag':surveySetup.thresholdJmag, \
           'thresholdKmag':surveySetup.thresholdKmag }
# Define pre-cuts for Confirmed and TOI samples:
survey['preCuts'] = { 'Confirmed':surveySetup.preCutsConfirmed, \
                      'TOIs':surveySetup.preCutsTOIs }
ipaths = { 'Confirmed':confirmedPickle, 'TOIs':toiPickle }

for SMFlag in [ 'TSM', 'ESM' ]:
    figFpaths = surveyGrids.BestInClass( ipaths=ipaths, survey=survey, \
                                         SMFlag=SMFlag, HeatMap=False )

    #out = surveyGrids.gridASCIIBestInClass(ipaths=ipaths, survey=survey, \
    #                                       SMFlag=SMFlag )
    #### Un-comment above two lines for normal run to get CSV file ####
    #edit line ~710 in surveyGrids.py to alter which params are in CSV file

    
    #print(out)
    
    # TODO...
    #ASCII1, ASCII2 = surveyGrids.CreateASCII_BestInClass( ipaths=ipaths, survey=survey, \
    #                                                      SMFlag=SMFlag )


