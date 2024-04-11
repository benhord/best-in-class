import numpy as np
import pdb
from TESS_ACWG import Utils

"""
Routines to define the parameters of the survey, including:
- preCuts to be applied to the target sample.
- threshold TSMs as a function of planet radius
- radius and temperature values defining the survey grid
"""

def preCutsConfirmed( z, obsSample ):
    """
    Pre-cuts to apply to sample of confirmed planets. 
    Edit titleStr and cutStr as appropriate.
    """
    n0 = len( z['planetName'] )
    ixs = np.arange( n0 )
    # Exclude very big and small stars:
    ixs = ixs[( z['RsRS'][ixs]>=0.05 )*( z['RsRS'][ixs]<10 )]
    # Removing grazing transits:
    ixs = ixs[z['b'][ixs]-z['RpRs'][ixs]<1]
    
    # TODO = Add option to apply a bright limit?
    # Optional string to be added to figure describing cuts.
    # Could use kwarg values to help specify.
    cutStr = ''
    if obsSample=='AllConfirmed':
        titleStr = 'Confirmed planets (empirical mass-radius relation used where necessary)'
    elif obsSample=='PublishedMassesOnly':
        titleStr = 'Planets with peer-reviewed published ($>5\\sigma$) masses'
        ixs = ixs[np.isfinite( z['MpValME'][ixs] )*np.isfinite( z['MpLsigME'][ixs] )]
        ixs = ixs[( z['MpValME'][ixs]/z['MpLsigME'][ixs]>=5 )]
        ixs = ixs[z['b'][ixs] < 1]
    elif obsSample=='NeedMasses':
        titleStr = 'Confirmed planets lacking published $>5\\sigma$ mass measurements'
        ixs = ixs[( z['MpValME'][ixs]/z['MpLsigME'][ixs]<5 )]
        ixs = ixs[z['b'][ixs] < 1]
    elif obsSample=='BestInClass':
        titleStr = 'Best-in-class Sample'
        #ixs = ixs[np.isfinite( z['MpValME'][ixs] )*np.isfinite( z['MpLsigME'][ixs] )]
        #ixs = ixs[( z['MpValME'][ixs]/z['MpLsigME'][ixs]>=5 )]
        #ixs = ixs[np.isfinite( z['MpValME'][ixs] )]
        ixs = ixs[z['b'][ixs] < 1]
        #ixs = ixs[z['RpRs'][ixs] < 1]
    else:
        pdb.set_trace()
    return ixs, cutStr, titleStr


def preCutsTOIs( z ):
    """
    Pre-cuts to apply to the sample of TOI candidates.
    Edit titleStr and cutStr as appropriate.
    """
    titleStr = 'TOIs not yet listed as confirmed on NASA Exoplanet Archive'    
    n0 = len( z['planetName'] )
    ixs = np.arange( n0 )
    # Exclude very big and small stars:
    ixs = ixs[( z['RsRS']>=0.05 )*( z['RsRS']<10 )]
    # Exclude grazing transits
    #ixs = ixs[z['b'][ixs] < 1]
    cutStr = ''
    return ixs, cutStr, titleStr


def thresholdTSM( RpRE, framework='ACWG' ):
    if framework=='ACWG':
        TSMstr = 'Kempton et al. (2018) TSM cuts have been applied'
        if RpRE<1.50: # 1. Terrestrials
            TSM = 0#10
        elif ( RpRE>=1.50 )*( RpRE<2.75 ): # 2. Small sub-Neptunes
            TSM = 0#90
        elif ( RpRE>=2.75 )*( RpRE<4.00 ): # 3. Large sub-Neptunes
            TSM = 0#90
        elif ( RpRE>=4.00 )*( RpRE<10.00 ): # 4. Sub-Jovians
            TSM = 0#90
        elif ( RpRE>=10.00 ): # 5. Gas giants
            TSM = 0#100
    elif framework=='TOIs':
        TSMstr = 'Only targets with TSM>50 (Rp>1.5RE) or TSM>10 (Rp<1.5RE) shown'
        if RpRE<1.50: # 1. Terrestrials
            TSM = 0#10
        else:
            TSM = 0#50 # 2. Everything else
    elif framework=='BestInClass':
        TSMstr = 'No TSM cuts applied'
        if RpRE<1.50: # 1. Terrestrials
            TSM = 0
        else:
            TSM = 0 # 2. Everything else
    return TSM, TSMstr


def thresholdESM( RpRE, framework='ACWG' ):
    ESMstr = '* Kempton et al. (2018) ESM cuts applied'
    ESM = 3#0

    return ESM, ESMstr


def thresholdTranSignal( RpRE, framework='ACWG' ):
    TSstr = 'Transmission signal size cuts made'
    TS = 2.0e-5#3.0e-5
        
    return TS, TSstr


def thresholdSecEclipse( RpRE, framework='ACWG' ):
    EDstr = 'Secondary eclipse depth cuts made'
    ED = 6.0e-5
        
    return ED, EDstr


def thresholdJmag( RpRE, framework='ACWG' ):
    if framework=='ACWG':
        Jstr = 'Jmag brightness cuts made'
        JCut = 6.0
    elif framework=='BestInClass':
        Jstr = 'Jmag brightness cuts made'
        JCut = 6.0

    return JCut, Jstr


def thresholdKmag( RpRE, framework='ACWG' ):
    Kstr = 'Kmag brightness cuts made'
    KCut = 6.4

    return KCut, Kstr


def gridEdges( surveyName='ACWG' ):
    if surveyName=='ACWG':
        TeqK = np.array( [ 100, 350, 800, 1250, 1750, 2250, 3000 ] )
        RpRE = np.array( [ 0.3, 1.50, 2.75, 4.00, 10.00, 25 ] )
    return TeqK, RpRE

