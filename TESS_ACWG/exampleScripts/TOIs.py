import os, sys, pdb
from TESS_ACWG import downloadTargetLists, processTargetLists
from TESS_ACWG import surveyGrids, surveySetup

ADIR = os.getcwd()

########################################################################
# Step 1: Download the target list from NASA Exoplanet Archive.

toiFpath = downloadTargetLists.targetsUnpublishedTOIs( forceDownload=False )

########################################################################
# Step 2: Process the csv file downloaded from NASA Exoplanet Archive.

toiPickle = processTargetLists.TOIs( csvIpath=toiFpath, pklOdir=ADIR, forceDownload=False )
    
########################################################################
# Step 3: Make the figures using the processed pickle file as input.
survey = { 'surveyName':'ACWG', 'framework':'ACWG', \
           'gridEdges':surveySetup.gridEdges, 'preCuts':surveySetup.preCutsTOIs, \
           'thresholdTSM':surveySetup.thresholdTSM, 'thresholdESM':surveySetup.thresholdESM }
RARanges = 'completeSet'
onlyPCs = 0 # 1 = True, 0 = False
# figFpaths = surveyGrids.TOIs( ipath=toiPickle, survey=survey, RARanges=RARanges, \
#                               SMFlag = 'ESM', onlyPCs = onlyPCs, HeatMap = True )
ASCII = surveyGrids.CreateASCII( survey=survey, SMFlag='TSM', onlyPCs = onlyPCs, \
                                multTIC=True )