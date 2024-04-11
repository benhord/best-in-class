import os, sys, pdb
from TESS_ACWG import downloadTargetLists, processTargetLists
from TESS_ACWG import surveyGrids
from TESS_ACWG import surveySetup

ADIR = os.getcwd()

########################################################################
# Step 1: Download the target list from NASA Exoplanet Archive.

csvIpath = 'TOI_2022-04-25.csv'
toiFpath = downloadTargetLists.targetsUnpublishedTOIs( csvIpath=csvIpath, forceDownload=False )

########################################################################
# Step 2: Process the csv file downloaded from NASA Exoplanet Archive.

toiPickle = processTargetLists.TOIs( csvIpath=toiFpath, pklOdir=ADIR, forceDownload=False )
    
########################################################################
# Step 3: Make the figures using the processed pickle file as input.
survey = { 'surveyName':'ACWG', 'obsSample':'TOIs', 'framework':'ACWG', \
           'gridEdges':surveySetup.gridEdges, 'preCuts':surveySetup.preCutsTOIs, \
           'thresholdTSM':surveySetup.thresholdTSM, 'thresholdESM':surveySetup.thresholdESM,
           'thresholdTS':surveySetup.thresholdTranSignal, \
           'thresholdSE':surveySetup.thresholdSecEclipse, \
           'thresholdJmag':surveySetup.thresholdJmag, \
           'thresholdKmag':surveySetup.thresholdKmag }
RARanges = 'completeSet'
topFivePredicted = False
for onlyPCs in [0, 1]:
    for SMFlag in [ 'TSM', 'ESM' ]:
        figFpaths = surveyGrids.TOIs( ipath=toiPickle, survey=survey, RARanges=RARanges, \
                                      SMFlag=SMFlag, onlyPCs=onlyPCs, HeatMap=False )
        ASCII = surveyGrids.CreateASCII_TOIs( survey=survey, SMFlag=SMFlag, onlyPCs=onlyPCs, \
                                              topFivePredicted=topFivePredicted, \
                                              forceDownloadExoFOP=False )
