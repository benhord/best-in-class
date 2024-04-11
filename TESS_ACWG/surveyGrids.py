import pdb, sys, os, time
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from . import Utils, processTargetLists, downloadTargetLists
from . import surveySetup
#from astropy.io import ascii
from forecaster import mr_forecast # for estimating mass uncertainties
from astropy.table import Table
try:
    import pysynphot
    pysynphotImport = True
except:
    pysynphotImport = False

FIGDIR = os.path.join( os.getcwd(), 'Figures' )
    

def quickCycle1():
    def gridEdgesFunc( surveyName='ACWG' ):
        if surveyName=='ACWG':
            TeqK = np.array( [ 100, 350, 800, 1250, 1750, 2250, 3000 ] )
            RpRE = np.array( [ 0.3, 1.50, 2.75, 4.00, 10.00, 25 ] )
        return TeqK, RpRE
    
    d = np.loadtxt( 'cycle1.csv', delimiter=',', dtype=str )
    pl = d[:,0]
    RpRE = np.array(d[:,2],dtype=float)
    TeqK = np.array(d[:,3],dtype=float)
    obsType = np.array(d[:,4],dtype=int)
    n = len( RpRE )
    ixs0 = np.arange( n )
    ixs1 = ixs0[( obsType==1 )] # transits
    ixs2 = ixs0[( obsType==2 )] # eclipses
    ixs3 = ixs0[( obsType==3 )] # phC

    titleStr0 = 'JWST Cycle 1 targets that will be observed GTO + GO'
    titleStr0 += '\nNo distinction for different instrument modes '
    titleStr1 = '{0}\ntransits + eclipses + phase curves'.format( titleStr0 )
    titleStr2a = '{0}\ntransmission (i.e. transits + phase curves)'.format( titleStr0 )
    titleStr2b = '{0}\nemission (i.e. eclipses + phase curves)'.format( titleStr0 )
    c1 = 'Orange'
    c2 = 'Cyan'
    c3 = 'Magenta'
    l1 = 'Transits'
    l2 = 'Eclipses'
    l3 = 'Phase curves'
    ms0 = 10
    fp = 1.5
    z = [ [ titleStr1,[ [ixs1,'o',c1,l1,ms0], [ixs2,'d',c2,l2,ms0], \
                        [ixs3,'*',c3,l3,fp*ms0] ] ], \
          [ titleStr2b,[ [ixs2,'d',c2,l2,ms0], [ixs3,'*',c3,l3,fp*ms0] ] ], \
          [ titleStr2a,[ [ixs1,'o',c1,l1,ms0], [ixs3,'*',c3,l3,fp*ms0] ] ] ]


    for i in z:
        fig, ax, ax2 = generateAxisScatter( wideFormat=True, titleStr='', showLegend=False )
        title_fs = 18
        toplineY = 0.98
        fig.text( 0.02, toplineY-0.02, i[0], fontsize=title_fs, weight='heavy', \
                  rotation=0, horizontalalignment='left', verticalalignment='top' )

        survey = { 'surveyName':'ACWG', 'gridEdges':gridEdgesFunc }
        Tgrid, Rgrid = drawGrid( ax, survey=survey )
        for j in i[1]:
            print( j[1] )
            ax.plot( TeqK[j[0]], RpRE[j[0]], j[1], mfc=j[2], mec='none', \
                     alpha=0.8, ms=j[4], label='' )
            ax.plot( TeqK[j[0]], RpRE[j[0]], j[1], mfc='none', mec='Black', \
                     alpha=1, ms=j[4], label='' )
            ax.plot( [-TeqK[j[0]][0]], [-RpRE[j[0]][0]], j[1], mfc=j[2], mec='Black', \
                     alpha=1, ms=j[4], label=j[3] )
        ax.legend( ncol=len( i[1] ), loc='lower right', bbox_to_anchor=[0.8,1], fontsize=16 )
        
    # pdb.set_trace()


def Confirmed( ipath='confirmedProperties.pkl', survey={}, SMFlag='TSM', HeatMap=False ):
    """
    """
    wideFormat = True
    showNeptuneRadius = False
    showJupiterRadius = False
    addSignature = False
    figPaths = gridConfirmed( ipath=ipath, wideFormat=wideFormat, \
                              showNeptuneRadius=showNeptuneRadius, \
                              showJupiterRadius=showJupiterRadius, \
                              survey=survey, addSignature=addSignature, \
                              SMFlag=SMFlag, HeatMap=HeatMap )
    for f in figPaths: # PDFs and PNGs
        for k in list( f.keys() ):
            fnew = f[k].replace( 'Confirmed_', 'Confirmed_{0}_'\
                                 .format( survey['obsSample'] ) )
            if os.path.isfile( fnew ):
                os.remove( fnew )
            os.rename( f[k], fnew )
            plt.close( 'all' )
    return None


def TOIs( ipath='toiProperties.pkl', survey={}, RARanges='all', SMFlag='TSM', \
          onlyPCs=False, HeatMap=False ):
    """
    """
    wideFormat = True
    addSignature = False
    DecRestrictions = [ ['DecAll',None,None], ['DecNth',-20,None], ['DecSth',None,20] ]
    z = readTOIProperties( ipath=ipath, SMFlag=SMFlag)[0]
    n0 = len( z['planetName'] )
    ixs = np.isfinite( z['TeqK'] )*np.isfinite( z['SM'] )*np.isfinite( z['RpValRE'] )
    print( '\nReading in {0:.0f} TOIs total.'.format( n0 ) )
    print( 'Returning {0:.0f} TOIs with radii, {1}, and Teq values.'\
           .format( ixs.sum(), SMFlag ) ) 
    print( '\nSaved:' )
    RARestrictions = Utils.getRARanges()
    if RARanges=='completeSet':
        RARestrictions += [ [ 0, 24 ] ]
    opaths = {}
    for i in DecRestrictions:
        opaths[i[0]] = {}
        for RA in RARestrictions:
            if RARanges=='all':
                r = 'RAall'
            else:
                r = 'RA{0:.0f}-{1:.0f}h'.format( RA[0], RA[1] )
            figPaths = gridTOIs( ipath=ipath, wideFormat=wideFormat, \
                                 addSignature=addSignature, survey=survey, \
                                 RAMin_hr=RA[0], RAMax_hr=RA[1], \
                                 DecMin_deg=i[1], DecMax_deg=i[2],
                                 SMFlag=SMFlag, onlyPCs=onlyPCs, \
                                 HeatMap=HeatMap )
            opaths[i[0]][r] = []
            for f in figPaths: # PDFs and PNGs
                for k in list( f.keys() ):
                    opath = f[k]
                    if f[k].find( '.pdf' )>0:
                        fnew = f[k].replace( '.pdf', '_{0}_{1}.pdf'.format( i[0], r ) )
                    elif opath.find( '.png' )>0:
                        fnew = f[k].replace( '.png', '_{0}_{1}.png'.format( i[0], r ) )
                    if os.path.isfile( fnew ):
                        os.remove( fnew )

                    os.rename( f[k], fnew )
                    opaths[i[0]][r] += [ fnew ]
            plt.close( 'all' )
    return opaths


def BestInClass( ipaths={ 'Confirmed':'', 'TOIs':'' }, survey={}, \
                 SMFlag='TSM', HeatMap=False ):
    """
    ipaths['Confirmed'] should probably be the version with missing masses added.
    """
    # Only option for now is for no RA/Dec limits for best-in-class:
    RAMin_hr = RAMax_hr = DecMin_deg = DecMax_deg = None
    wideFormat = False#True
    addSignature = False
    figPaths = gridBestInClass( ipaths=ipaths, wideFormat=wideFormat, \
                                survey=survey, addSignature=addSignature, \
                                SMFlag=SMFlag, HeatMap=HeatMap, \
                                RAMin_hr=RAMin_hr,  RAMax_hr=RAMax_hr, \
                                DecMin_deg=DecMin_deg,  DecMax_deg=DecMax_deg )
    for f in figPaths: # PDFs and PNGs
        for k in list( f.keys() ):
            fnew = f[k].replace( 'BestInClass_', 'BestInClass_{0}_'\
                                 .format( survey['obsSample'] ) )
            if os.path.isfile( fnew ):
                os.remove( fnew )
            os.rename( f[k], fnew )
            plt.close( 'all' )
    return None


#############################################################################
# Transmission/Emission spectroscopy survey:


def gridBestInClass( ipaths={ 'Confirmed':'', 'TOIs':''}, \
                     wideFormat=True, addSignature=False, survey={}, \
                     RAMin_hr=None, RAMax_hr=None, \
                     DecMin_deg=None, DecMax_deg=None, \
                     SMFlag='TSM', onlyPCs=True, ASCII=False, HeatMap=False ):
    """
    Top 5 in each cell are 'Best in class' drawn from Confirmed and TOIs.
    """
    showGrid = True
    zT, dateStr = readTOIProperties( ipath=ipaths['TOIs'], SMFlag=SMFlag )
    zC, dateStr = readConfirmedProperties( ipath=ipaths['Confirmed'], SMFlag=SMFlag )
    ostr = 'BestInClass'
    #if onlyPCs == True:
    #    ostr = 'TOIs_onlyPCs'

    tmp_df = pd.DataFrame.from_dict(zT)
    #print(tmp_df.planetName)
    #print('TOI-4856.01(PC)' in tmp_df.planetName.values)
    #print('TOI-1022.01(PC)' in tmp_df.planetName.values)
    #print(tmp_df.RpRs[tmp_df.planetName == 'TOI-1347.01(PC)'])
    #print([idx for idx, s in enumerate(list(tmp_df.planetName.values)) if '4856' in s][0])
    #print(tmp_df.columns)
    #blah
    print(len(tmp_df))
    print(tmp_df)
    
    preCutsC = survey['preCuts']['Confirmed']
    preCutsT = survey['preCuts']['TOIs']
    obsSample = survey['obsSample']

    # temporary hack:
    if np.any( [ RAMin_hr, RAMax_hr, DecMin_deg, DecMax_deg ] ) is not None:
        print( 'For best-in-class, RA/Dec limits not yet implemented' )
        print( '... need to sort out consistently for Confirmed and TOIs first (TODO).' )
        pdb.set_trace() 
    limitsRA_hr = [ RAMin_hr, RAMax_hr ]
    limitsDec_deg = [ DecMin_deg, DecMax_deg ]
    zT, cutStrT, titleStrT, RADecStrT = Utils.applyPreCutsTOIs( zT, preCutsT, obsSample, \
                                                                limitsRA_hr, \
                                                                limitsDec_deg, \
                                                                onlyPCs=True )
    # Hack to add an array zT['thresholdPasses'] of 0s and 1s specifying
    # if a given TOI with estimated mass passes the ACWG (i.e. Kempton et al)
    # TSM/ESM threshold metric for its given Rp-Teq box; note that the survey
    # dictionary contains the Rp-Teq divisions, but we must force 'ACWG' as
    # the framework argument here, because framework=='BestInClass' in the
    # survey dictionary:
    zT = Utils.addThresholdPasses( zT, survey, SMFlag, 'ACWG' )
    zC, cutStrC, titleStrC, RADecStrC = Utils.applyPreCutsConfirmed( zC, preCutsC, \
                                                                     obsSample, \
                                                                     limitsRA_hr, \
                                                                     limitsDec_deg )

    #print(zT)
    #print(zT['planetName'])
    #print(zT.keys())

    # filtering tois based off of those in sample_tois.txt file
    #tmp_df = pd.DataFrame.from_dict(zT)
    ###sample_tois = np.loadtxt('sample_tois.txt', dtype=str, delimiter='\n')

    ###tmp_df = tmp_df[tmp_df.planetName.isin(sample_tois)].reset_index().drop(columns='index')

    ###zComb = {}
    ###for i in range(len(tmp_df.columns)):
    ###    k = tmp_df.columns[i]
    ###    zComb[k] = np.array(tmp_df[k])
        
    ###zT = zComb
    
    # Combine the Confirmed and TOI samples:
    z = Utils.combineConfirmedAndTOIs( zC, zT )
    
    # Restrict grids only to Nov 3, 2022 best-in-class sample
    tmp_df = pd.DataFrame.from_dict(z)
    print(tmp_df.planetName)
    tmp_df.to_csv('all_pls_out_' + str(SMFlag) + '.csv', index=False)

    nov_sample = np.loadtxt('sample_11032022.txt', dtype=str, delimiter='\n')
    ###tmp_df = tmp_df[tmp_df.planetName.isin(nov_sample)].reset_index().drop(columns='index') #Comment back in for paper runs
    tmp_df.to_csv(SMFlag + '_sample_pls_out.csv', index=False)

    #print(tmp_df[tmp_df.planetName == 'TOI-3235.01(PC)'])
    #blah
    
    #print(len(nov_sample))
    #print(list(set(nov_sample) - set(tmp_df.planetName)))
    #print(len(tmp_df))
    #print(tmp_df)
    #print(tmp_df.columns)
    #blah

    ###validated_pls = np.loadtxt('')
    ###tmp_df['statusMass'][tmp_df['planetName'].isin(validated_pls)] = 5
    
    # Filter out bad planets from text file ##after TOI-906.01 is hacked bits
    bad_pls = np.loadtxt('bad_pls.txt', dtype=str, delimiter='\n')
    #bad_pls = np.loadtxt('bad_pls_alt.txt', dtype=str, delimiter='\n')

    ####tmp_df = tmp_df[~tmp_df.planetName.isin(bad_pls)].reset_index().drop(columns='index')
    tmp_df['statusMass'][tmp_df['planetName'].isin(bad_pls)] = 4

    #tmp_df['planetName'][tmp_df['statusMass'] == 3] = tmp_df['planetName'][tmp_df['statusMass'] == 3].isin(sample_tois)
    
    #print(tmp_df.columns)
    tmp_df['planetName'][tmp_df['statusMass'] >= 3] = tmp_df['planetName'][tmp_df['statusMass'] >= 3].str.split('(').str[0]
    #print(tmp_df.planetName)
    #blah

    validated_pls = np.loadtxt('tois_validated.txt', dtype=str,
                               delimiter='\n')
    confirmed_pls = np.loadtxt('confirmed_pls.txt', dtype=str, delimiter='\n')
    inconclusive_pls = np.loadtxt('inconclusive_pls.txt', dtype=str,
                               delimiter='\n')
    ## COMMENTED OUT FOR AAS TALK
    tmp_df['statusMass'][tmp_df['planetName'].isin(validated_pls)] = 5
    tmp_df['statusMass'][tmp_df['planetName'].isin(confirmed_pls)] = 2
    tmp_df['statusMass'][tmp_df['planetName'].isin(inconclusive_pls)] = 6

    #Comment these back in for paper runs
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-836.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-969.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-1099.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-1853.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-2134.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-4463.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-4641.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-238.01'].reset_index().drop(columns='index')
    #tmp_df = tmp_df[tmp_df.planetName != 'TOI-1347.01'].reset_index().drop(columns='index')

    #print(tmp_df[tmp_df.planetName == 'TOI-1347.01'])
    #print(tmp_df[tmp_df.planetName == 'TOI-2134.01']['TeqK'])
    #print(SMFlag)
    
    #print(tmp_df['planetName'][tmp_df['planetName'].isin(validated_pls)])
    #blah
    
    zComb = {}
    for i in range(len(tmp_df.columns)):
        k = tmp_df.columns[i]
        zComb[k] = np.array(tmp_df[k])
        
    z = zComb
    
    #Kepler-230 b
    #Kepler-24 e
    #TOI-1856.01(PC)
    ##test_pl = 'K2-106b'
    ##print(test_pl)
    #print(z['TeqK'][z['planetName'] == test_pl])
    #print(z['RpRs'][z['planetName'] == test_pl])
    #print(z['TstarK'][z['planetName'] == test_pl])
    #print(z['Kmag'][z['planetName'] == test_pl])
    ##print(SMFlag)
    ##print(z['SM'][z['planetName'] == test_pl])
    #print(z['RpValRE'][z['planetName'] == test_pl])
    #print(z['MpValME'][z['planetName'] == test_pl])
    #print(z['TstarK'][z['planetName'] == test_pl])
    #print(z.keys())
    tmp_df = pd.DataFrame.from_dict(z)
    print(tmp_df)
    tmp_df.to_csv(str(SMFlag) + '2-22-24' + '.csv', index=False)
     
    ##if SMFlag == 'TSM':
    ##    print(Utils.computeTSM(z['RpValRE'][z['planetName'] == test_pl], z['MpValME'][z['planetName'] == test_pl], z['RsRS'][z['planetName'] == test_pl], z['TeqK'][z['planetName'] == test_pl], z['Jmag'][z['planetName'] == test_pl]))
    ##elif SMFlag == 'ESM':
    ##    print(Utils.computeESM(z['TeqK'][z['planetName'] == test_pl], z['RpRs'][z['planetName'] == test_pl], z['TstarK'][z['planetName'] == test_pl], z['Kmag'][z['planetName'] == test_pl]))
    
    titleStr = ''#'Best-in-class sample'
    RADecStr = ''
    onames = {}

    # Radius-temperature grid plot listing the top-ranked planets in each cell:
    if ASCII:
        plList = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, \
                                dateStr=dateStr, survey=survey, \
                                RADecStr=RADecStr, ASCII=ASCII, \
                                HeatMap=HeatMap, \
                                TOIGrid=False, bestInClass=True )
        return plList
    fig2, ax2 = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, \
                               dateStr=dateStr, \
                               survey=survey, RADecStr=RADecStr, \
                               HeatMap=HeatMap, \
                               TOIGrid=False, bestInClass=True )
    onames['2'] = '{0}_gridTop{1}s.pdf'.format( ostr, SMFlag )

    #toiNote = 'TOIs with "PC" TFOPWG Disposition shown in darker font\n'
    #if onlyPCs == True:
    #    toiNote = 'Only TOIs with "PC" TFOPWG Disposition are displayed\n'
    #toiNote += 'Masses estimated from empirical relation (adapted from Chen & Kipping 2017)'
    #if SMFlag=='TSM':
    #    toiNote = 'When measured mass not available, adapted Chen & Kipping (2017) relation is assumed for TSM.'
    #    fig2.text( 0.08, 0.91-0.10, toiNote, \
    #            c='black', fontsize=14, horizontalalignment='left', \
    #            verticalalignment='bottom' )

    if addSignature==True:
        for ax in [ax2]:
            addSignatureToAxis( ax )

    figs = { '2':fig2 }

    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat/BestInClass'.format( survey['surveyName'] )
    else:
        odirExt = 'survey{0}/narrowFormat/BestInClass'.format( survey['surveyName'] )
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opathsPDF = {}
    opathsPNG = {}
    sourceStr = 'Source: NASA Exoplanet Archive ({0})'.format( dateStr )
    for k in ['2']:
        figs[k].text( 0.97, 0.01, sourceStr, fontsize=10, \
                      horizontalalignment='right', verticalalignment='bottom' )
        if addSignature==True:
            onames[k] = onames[k].replace( '.pdf', '_wSignature.pdf' )
        
        opathk = os.path.join( odir, onames[k] )
        figs[k].savefig( opathk )
        opathk_png = opathk.replace( '.pdf', '.png' )
        figs[k].savefig( opathk_png )
        opathsPDF[k] = opathk
        opathsPNG[k] = opathk_png
        print( '{0}\n{1}'.format( opathk, opathk_png ) )
        print( 'RADecStr = {0}'.format( RADecStr ) )
    
    return opathsPDF, opathsPNG


def gridTOIs( ipath='toiProperties.pkl', wideFormat=True, \
              addSignature=False, survey={}, \
              RAMin_hr=None, RAMax_hr=None, \
              DecMin_deg=None, DecMax_deg=None, \
              SMFlag='TSM', onlyPCs=False, ASCII=False, HeatMap=False ):
    """
    TOIs that have not been confirmed.
    """
    
    showGrid = True
    z, dateStr = readTOIProperties( ipath=ipath, SMFlag=SMFlag )
    ostr = 'TOIs'
    if onlyPCs == True:
        ostr = 'TOIs_onlyPCs'
    limitsRA_hr = [ RAMin_hr, RAMax_hr ]
    limitsDec_deg = [ DecMin_deg, DecMax_deg ]
    z, cutStr, titleStr, RADecStr = Utils.applyPreCutsTOIs( z, survey['preCuts'], \
                                                            survey['obsSample'], \
                                                            limitsRA_hr, limitsDec_deg, \
                                                            onlyPCs=onlyPCs )
    onames = {}

    # Filter out bad planets from text file
    #tmp_df = pd.DataFrame.from_dict(z)
    #tmp_df.to_csv('all_pls_out.csv', index=False)
    #bad_pls = np.loadtxt('bad_pls.txt', dtype=str, delimiter='\n')

    #tmp_df = tmp_df[~tmp_df.planetName.isin(bad_pls)].reset_index().drop(columns='index')

    #tmp_df['planetName'] = tmp_df['planetName'].str.split('(').str[0]
    
    #zComb = {}
    #for i in range(len(tmp_df.columns)):
    #    k = tmp_df.columns[i]
    #    zComb[k] = np.array(tmp_df[k])
        
    #z = zComb
    
    # Radius-temperature grid plot listing the top-ranked planets in each cell:
    if ASCII:
        plList = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, dateStr=dateStr, \
                                survey=survey, RADecStr=RADecStr, ASCII=ASCII, \
                                HeatMap=HeatMap, TOIGrid=True )
        return plList
    fig2, ax2 = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, dateStr=dateStr, TOIGrid=True, \
                               survey=survey, RADecStr=RADecStr, HeatMap=HeatMap )
    onames['2'] = '{0}_gridTop{1}s.pdf'.format( ostr, SMFlag )

    #toiNote = 'TOIs with "PC" TFOPWG Disposition shown in darker font\n'
    #if onlyPCs == True:
    #    toiNote = 'Only TOIs with "PC" TFOPWG Disposition are displayed\n'
    #toiNote += 'Masses estimated from empirical relation (adapted from Chen & Kipping 2017)'
    if SMFlag=='TSM':
        toiNote = 'For TSM calculations, empirical relation adapted from Chen & Kipping (2017) assumed for masses.'
        fig2.text( 0.08, 0.91-0.10, toiNote, \
                c='black', fontsize=14, horizontalalignment='left', \
                verticalalignment='bottom' )

    if addSignature==True:
        for ax in [ax2]:
            addSignatureToAxis( ax )

    figs = { '2':fig2 }

    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat/TOIs/{1}'.format( survey['surveyName'], \
                                                         SMFlag)
        if onlyPCs:
            odirExt = odirExt+'/onlyPCs'
    else:
        odirExt = 'survey{0}/narrowFormat/TOIs/{1}'.format( survey['surveyName'], \
                                                           SMFlag)
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opathsPDF = {}
    opathsPNG = {}
    sourceStr = 'Source: NASA Exoplanet Archive ({0})'.format( dateStr )
    for k in ['2']:
        figs[k].text( 0.97, 0.01, sourceStr, fontsize=10, \
                      horizontalalignment='right', verticalalignment='bottom' )
        if addSignature==True:
            onames[k] = onames[k].replace( '.pdf', '_wSignature.pdf' )
        
        opathk = os.path.join( odir, onames[k] )
        figs[k].savefig( opathk )
        opathk_png = opathk.replace( '.pdf', '.png' )
        figs[k].savefig( opathk_png )
        opathsPDF[k] = opathk
        opathsPNG[k] = opathk_png
        print( '{0}\n{1}'.format( opathk, opathk_png ) )
        print( 'RADecStr = {0}'.format( RADecStr ) )
    
    return opathsPDF, opathsPNG
    
    
def gridTESS( publishedMasses=True, wideFormat=True, addSignature=False, SMFlag = 'TSM' ):
    """
    Confirmed TESS planets without published mass.
    Currently unused, may be out of date
    """

    surveyName = '' #This variable is used below but undefined, this is added as a placeholder definition
    showGrid = True
    z = readConfirmedTESSProperties( publishedMasses=publishedMasses, SMFlag=SMFlag )
    if publishedMasses==True:
        ostr = 'ConfirmedWithMassTESS'
        titleStr = 'Confirmed TESS planets with peer-reviewed published masses'
    else:
        ostr = 'ConfirmedNoMassTESS'
        titleStr = 'Confirmed TESS planets without peer-reviewed published masses'
    
    n0 = len( z['planetName'] )
    # Exclude very big and small stars:
    ixs0 = ( z['RsRS']>=0.05 )*( z['RsRS']<10 )
    # TODO = Add option to apply a bright limit?
    ixs = np.arange( n0 )[ixs0]#[ixs2][ixs3]
    print( '{0:.0f} planets have >5-sigma mass measurements.'.format( len( ixs ) ) )
    pl = z['planetName'][ixs]
    SM = z['SM'][ixs]
    Teq = z['TeqK'][ixs]
    Ts = z['TstarK'][ixs]
    MpVal = z['MpValME'][ixs]
    RpVal = z['RpValRE'][ixs]
    RpLsig = z['RpLsigRE'][ixs]
    RpUsig = z['RpUsigRE'][ixs]

    onames = {}

    # Radius-temperature grid plot listing the top-ranked planets in each cell:
    fig2, ax2 = plotTeqRpGrid( Teq, RpVal, Ts, (SMFlag, SM), pl, titleStr=titleStr )
    onames['2'] = '{0}_gridTop{1}s.pdf'.format( ostr, SMFlag )

    if publishedMasses==False:
        fig2.text( 0.10, 0.905-0.025, 'Masses estimated from empirical relation', \
                   c='black', fontsize=14, horizontalalignment='left', \
                   verticalalignment='bottom' )
    
    if addSignature==True:
        for ax in [ax2]:
            addSignatureToAxis( ax )
    
    figs = { '2':fig2 }

    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat'.format( surveyName ) #surveyName is undefined
    else:
        odirExt = 'survey{0}/narrowFormat'.format( surveyName )
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    for k in ['2']:
        if addSignature==True:
            onames[k] = onames[k].replace( '.pdf', '_wSignature.pdf' )
        opath = os.path.join( odir, onames[k] )
        figs[k].savefig( opath )
        print( opath )

    return None

    
def gridConfirmed( ipath='confirmedProperties.pkl', wideFormat=True, \
                   survey={}, addSignature=False, showGrid=True, \
                   showNeptuneRadius=False, showJupiterRadius=False, \
                   SMFlag='TSM', HeatMap=False, ASCII=False ):
    """
    
    """
    z, dateStr = readConfirmedProperties( ipath=ipath, SMFlag=SMFlag )
    ostr = 'Confirmed'

    # Not applying Dec restrictions to Confirmed planets for now:
    #DecStr, DecMin_deg, DecMax_deg = processDecRestriction( None, None )
    #RADecStr = ''
    
    limitsRA_hr = [ None, None ] # not implemented currently
    limitsDec_deg = [ None, None ] # not implemented currently
    z, cutStr, titleStr, RADecStr = Utils.applyPreCutsConfirmed( z, survey['preCuts'], \
                                                                 survey['obsSample'], \
                                                                 limitsRA_hr, \
                                                                 limitsDec_deg )    
    onames = {}
    
    # Radius-temperature plot for all planets with well-measured mass:
    if not ASCII:
        fig1a, ax1a = plotTeqRpScatter( z, SMFlag, \
                                        applySMcuts=False, \
                                        wideFormat=wideFormat, survey=survey, \
                                        showGrid=showGrid, titleStr=titleStr, \
                                        indicateTESS=False, dateStr=dateStr, \
                                        showNeptuneRadius=showNeptuneRadius, \
                                        showJupiterRadius=showJupiterRadius )
        onames['1a'] = '{0}_allPlanets.pdf'.format( ostr )

        # Radius-temperature plot for all planets with well-measured mass
        # and SM cuts applied:
        fig1b, ax1b = plotTeqRpScatter( z, SMFlag, \
                                        applySMcuts=True, \
                                        wideFormat=wideFormat, survey=survey, \
                                        showGrid=showGrid, titleStr=titleStr, \
                                        indicateTESS=False, dateStr=dateStr, \
                                        showNeptuneRadius=showNeptuneRadius, \
                                        showJupiterRadius=showJupiterRadius )

        onames['1b'] = '{0}_{1}cutsApplied.pdf'.format( ostr, SMFlag )


        # Radius-temperature grid plot listing the top-ranked planets in each cell:
        extraNotes = '\nTESS discoveries shown in bold font'
        fig2, ax2 = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, dateStr=dateStr, \
                                   extraNotes=extraNotes, survey=survey, TOIGrid=False, \
                                   RADecStr=RADecStr, HeatMap=HeatMap, \
                                   confirmed=True  )
        fig2.text( 0.10, 0.995, cutStr, c='black', fontsize=12, \
                   horizontalalignment='left', verticalalignment='top' )
    else:
        plList, dateStr = plotTeqRpGrid( z, SMFlag, titleStr=titleStr, dateStr=dateStr, \
                                         survey=survey, ASCII=ASCII, RADecStr=RADecStr, \
                                         HeatMap=HeatMap, TOIGrid=False  )
        return plList, dateStr
    
    onames['2'] = '{0}_gridTop{1}s.pdf'.format( ostr, SMFlag )

    # Scatter plots without the grid:
    fig3a, ax3a = plotTeqRpScatter( z, SMFlag, \
                                    applySMcuts=False, \
                                    wideFormat=wideFormat, survey=survey, \
                                    showGrid=showGrid, titleStr=titleStr, \
                                    indicateTESS=True, dateStr=dateStr, \
                                    showNeptuneRadius=showNeptuneRadius, \
                                    showJupiterRadius=showJupiterRadius )
    onames['3a'] = '{0}_allPlanets_showsTESS.pdf'.format( ostr )
    fig3b, ax3b = plotTeqRpScatter( z, SMFlag, \
                                    applySMcuts=True, \
                                    wideFormat=wideFormat, survey=survey, \
                                    showGrid=showGrid, titleStr=titleStr,
                                    indicateTESS=True, dateStr=dateStr, \
                                    showNeptuneRadius=showNeptuneRadius, \
                                    showJupiterRadius=showJupiterRadius )
    
    onames['3b'] = '{0}_{1}cutsApplied_showsTESS.pdf'.format( ostr, SMFlag )

    if addSignature==True:
        for ax in [ax1a,ax1b,ax2,ax3a,ax3b]:
            addSignatureToAxis( ax )
    
    figs = { '1a':fig1a, '1b':fig1b, '2':fig2, '3a':fig3a, '3b':fig3b }
    print( '\nSaved:' )
    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat/Confirmed/{1}'.format( survey['surveyName'], \
                                                               SMFlag )
    else:
        odirExt = 'survey{0}/narrowFormat/Confirmed/{1}'.format( survey['surveyName'], \
                                                            SMFlag )
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opathsPDF = {}
    opathsPNG = {}
    sourceStr = 'Source: NASA Exoplanet Archive ({0})'.format( dateStr )
    for k in ['1a','1b','2','3a','3b']:
        figs[k].text( 0.97, 0.01, sourceStr, fontsize=10, \
                      horizontalalignment='right', verticalalignment='bottom' )
        if addSignature==True:
            onames[k] = onames[k].replace( '.pdf', '_wSignature.pdf' )
        opathk = os.path.join( odir, onames[k] )
        figs[k].savefig( opathk )
        opathk_png = opathk.replace( '.pdf', '.png' )
        figs[k].savefig( opathk_png )
        opathsPDF[k] = opathk
        opathsPNG[k] = opathk_png
        print( '{0}\n{1}'.format( opathk, opathk_png ) )
    return opathsPDF, opathsPNG


def gridASCIIBestInClass(ipaths={ 'Confirmed':'', 'TOIs':''}, \
                         addSignature=False, survey={}, \
                         RAMin_hr=None, RAMax_hr=None, \
                         DecMin_deg=None, DecMax_deg=None, \
                         SMFlag='TSM', onlyPCs=True ):

    bestPlanets = gridBestInClass( ipaths=ipaths, survey=survey, SMFlag=SMFlag, \
                                   HeatMap=False, wideFormat=False, ASCII=True )
    
    showGrid = True
    zT, dateStr = readTOIProperties( ipath=ipaths['TOIs'], SMFlag=SMFlag )
    zC, dateStr = readConfirmedProperties( ipath=ipaths['Confirmed'], SMFlag=SMFlag )
    ostr = 'BestInClass'
    #if onlyPCs == True:
    #    ostr = 'TOIs_onlyPCs'

    preCutsC = survey['preCuts']['Confirmed']
    preCutsT = survey['preCuts']['TOIs']
    obsSample = survey['obsSample']

    # temporary hack:
    if np.any( [ RAMin_hr, RAMax_hr, DecMin_deg, DecMax_deg ] ) is not None:
        print( 'For best-in-class, RA/Dec limits not yet implemented' )
        print( '... need to sort out consistently for Confirmed and TOIs first (TODO).' )
        pdb.set_trace() 
    limitsRA_hr = [ RAMin_hr, RAMax_hr ]
    limitsDec_deg = [ DecMin_deg, DecMax_deg ]
    zT, cutStrT, titleStrT, RADecStrT = Utils.applyPreCutsTOIs( zT, preCutsT, obsSample, \
                                                                limitsRA_hr, \
                                                                limitsDec_deg, \
                                                                onlyPCs=True )
    # Hack to add an array zT['thresholdPasses'] of 0s and 1s specifying
    # if a given TOI with estimated mass passes the ACWG (i.e. Kempton et al)
    # TSM/ESM threshold metric for its given Rp-Teq box; note that the survey
    # dictionary contains the Rp-Teq divisions, but we must force 'ACWG' as
    # the framework argument here, because framework=='BestInClass' in the
    # survey dictionary:
    zT = Utils.addThresholdPasses( zT, survey, SMFlag, 'ACWG' )
    zC, cutStrC, titleStrC, RADecStrC = Utils.applyPreCutsConfirmed( zC, preCutsC, \
                                                                     obsSample, \
                                                                     limitsRA_hr, \
                                                                     limitsDec_deg )

    tmp_df = pd.DataFrame.from_dict(zT)
    sample_tois = np.loadtxt('sample_tois.txt', dtype=str, delimiter='\n')

    tmp_df = tmp_df[tmp_df.planetName.isin(sample_tois)].reset_index().drop(columns='index')

    zComb = {}
    for i in range(len(tmp_df.columns)):
        k = tmp_df.columns[i]
        zComb[k] = np.array(tmp_df[k])
        
    zT = zComb
    
    # Combine the Confirmed and TOI samples:
    z = Utils.combineConfirmedAndTOIs( zC, zT )

    tmp_df = pd.DataFrame.from_dict(z)
    #tmp_df.to_csv('all_pls_out.csv', index=False)
    #bad_pls = np.loadtxt('bad_pls.txt', dtype=str, delimiter='\n')

    #tmp_df = tmp_df[~tmp_df.planetName.isin(bad_pls)].reset_index().drop(columns='index')

    #tmp_df['planetName'][tmp_df['statusMass'] == 3] = tmp_df['planetName'][tmp_df['statusMass'] == 3].str.split('(').str[0]

    zComb = {}
    for i in range(len(tmp_df.columns)):
        k = tmp_df.columns[i]
        zComb[k] = np.array(tmp_df[k])
        
    z = zComb
    
    bestPlanets = [x.replace("*", "") for x in bestPlanets[0]]
    z['planetName'] = [x.replace(" ", "") for x in z['planetName']]
    ixs = [list(z['planetName']).index(i) for i in bestPlanets]
    bestPlanets = [x.replace("(PC)", "") for x in bestPlanets]

    if not os.path.isfile( 'bestInClassOutputParams.csv' ):
        outParams = pd.DataFrame( {'planetName' : bestPlanets, \
                                   'disposition' : z['statusMass'][ixs], \
                                   'Teq' : z['TeqK'][ixs], 'Rad' : z['RpValRE'][ixs], \
                                   SMFlag : z['SM'][ixs]} )

        outParams.to_csv('bestInClassOutputParams.csv', index=False)
        
    else:
        inParams = pd.read_csv('bestInClassOutputParams.csv', header=0, comment='#')

        newParams = pd.DataFrame( {'planetName' : bestPlanets, \
                                   'disposition' : z['statusMass'][ixs], \
                                   'Teq' : z['TeqK'][ixs], 'Rad' : z['RpValRE'][ixs], \
                                   SMFlag : z['SM'][ixs]} )
        outParams = pd.merge(inParams, newParams, how='outer', \
                             on=['planetName', 'disposition', 'Teq', 'Rad'])

        outParams.to_csv('bestInClassOutputParams.csv', index=False)
        
    

    print(outParams)
    
    #remove (PC) also

    #save csv file
    
    return( outParams )


def transmissionPredictedTESS( showSolarSystem=True, wideFormat=False, \
                               surveyModule='ACWG', showGrid=True, \
                               showStellarTrack=True, showNeptuneRadius=True, \
                               showJupiterRadius=True, crossHair=None, \
                               addSignature=False, SMFlag='TSM' ):
    """
    Focusing on TESS predicted yield, not worrying about survey grid.
    Currently unused, out of date. In the middle of being repurposed 
    to create plots of predicted planets
    """

    z = readPredictedProperties(SMFlag = SMFlag)
    ostr = 'predictedTESS'
    
    survey = { 'surveyName':'ACWG', 'obsSample':'PublishedMassesOnly', \
               'framework':'ACWG', 'gridEdges':surveySetup.gridEdges, \
               'thresholdTSM':surveySetup.thresholdTSM, \
               'thresholdESM':surveySetup.thresholdESM, \
               'preCuts':surveySetup.preCutsConfirmed }

    n0 = len( z['RsRS'] )

    #Exclude very big and small stars:
    ixs = ( z['RsRS']>=0.05 )*( z['RsRS']<10 )
    
    SM = z['SM'][ixs]
    Teq = z['TeqK'][ixs]
    Ts = z['TstarK'][ixs]
    MpVal = z['MpValME'][ixs]
    RpVal = z['RpValRE'][ixs]
    TESS = [0 for i in range(len(ixs))]
    pl = ['PredictedPlanet-{0}'.format(str(i)) for i in range(len(ixs))]

    dateStr = processTargetLists.getDateStr( 'predictedProperties_v2.pkl', \
                                             whichList='Predicted' )

    # Radius-temperature plot for all planets with well-measured mass:    
    titleStr = 'Top ranked predicted planets'
    
    fig0a, ax0a = plotTeqRpScatter( pl, Teq[ixs], RpVal[ixs], Ts[ixs], \
                                    ('SMFlag', SM[ixs]), \
                                    TESS[ixs], applyTSMcuts=False, ms=6, alpha=1, \
                                    starColors=True, showSolarSystem=showSolarSystem, \
                                    showStellarTrack=showStellarTrack, \
                                    wideFormat=wideFormat, titleStr=titleStr, \
                                    surveyModule=surveyModule, showGrid=showGrid, \
                                    indicateTESS=False, dateStr=dateStr, \
                                    showNeptuneRadius=showNeptuneRadius, \
                                    showJupiterRadius=showJupiterRadius, survey=survey )
    if crossHair is not None:
        ix = ( pl==crossHair )
        ax0a.axvline( Teq[ix], c='HotPink' )
        ax0a.axhline( RpVal[ix], c='HotPink' )
        # pdb.set_trace()
    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat'.format( surveyModule )
    else:
        odirExt = 'survey{0}/narrowFormat'.format( surveyModule )
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    onames = {}
    for k in ['0a','0b','1a','1b','2a','2b','3']:
        onames[k] = '{0}_{1}.pdf'.format( ostr, k )
    if showSolarSystem==True:
        for k in ['0a','0b','1a','1b','2a','2b','3']:
            onames[k] = onames[k].replace( '.pdf', '_wSS.pdf' )
    if showStellarTrack==True:
        for k in ['0a','0b','1a','1b','2a','2b','3']:
            onames[k] = onames[k].replace( '.pdf', '_wST.pdf' )
    opaths = {}
    for k in ['0a','0b','1a','1b','2a','2b','3']:
        opaths[k] = os.path.join( odir, onames[k] )
        
    fig0a.savefig( opaths['0a'] )
    
    fig0b, ax0b = plotTeqRpScatter( pl[ixs], Teq[ixs], RpVal[ixs], Ts[ixs], \
                                    ('SMFlag', SM[ixs]), \
                                    TESS[ixs], applyTSMcuts=False, ms=6, alpha=1, \
                                    starColors=True, showSolarSystem=showSolarSystem, \
                                    showStellarTrack=showStellarTrack, \
                                    wideFormat=wideFormat, titleStr=titleStr, \
                                    surveyModule=surveyModule, showGrid=showGrid, \
                                    indicateTESS=False, dateStr=dateStr, \
                                    showNeptuneRadius=showNeptuneRadius, \
                                    showJupiterRadius=showJupiterRadius, survey=survey )
   
    plotTeqRpTESS( ax0b, SMFlag = SMFlag )
    
    fig0b.savefig( opaths['0b'] )
    
 # Same as previous but with low alpha value for known planets:
    titleStr = 'Predicted TESS planets' 
    
    fig1, ax1 = plotTeqRpScatter( pl[ixs], Teq[ixs], RpVal[ixs], Ts[ixs], \
                                  ('SMFlag', SM[ixs]), \
                                  TESS[ixs], applyTSMcuts=False, ms=6, alpha=0.3, \
                                  starColors=True, showSolarSystem=showSolarSystem, \
                                  showStellarTrack=showStellarTrack, \
                                  wideFormat=wideFormat, titleStr=titleStr, \
                                  surveyModule=surveyModule, showGrid=showGrid, \
                                  indicateTESS=False, dateStr=dateStr, \
                                  showNeptuneRadius=showNeptuneRadius, \
                                  showJupiterRadius=showJupiterRadius, survey=survey )
    
    fig1.savefig( opaths['1a'] )

    # Add the bright predicted TESS planets:
    titleStr = 'Predicted TESS planets'
    
    plotTeqRpTESS( ax1, showSolarSystem=False, showNeptuneRadius=False, \
                   showJupiterRadius=False, SMFlag = SMFlag, z=z )
    
    fig1.savefig( opaths['1b'] )
    
    # Radius-temperature plot for all planets with well-measured mass:
    fig2, ax2 = plotTeqRpScatter( pl[ixs], Teq[ixs], RpVal[ixs], Ts[ixs], \
                                  ('SMFlag', SM[ixs]), \
                                  TESS[ixs], applyTSMcuts=False, ms=3, alpha=1, \
                                  starColors=False, showSolarSystem=showSolarSystem, \
                                  showStellarTrack=showStellarTrack, \
                                  wideFormat=wideFormat, titleStr=titleStr, \
                                  surveyModule=surveyModule, showGrid=showGrid, \
                                  indicateTESS=False, dateStr=dateStr, \
                                  showNeptuneRadius=showNeptuneRadius, \
                                  showJupiterRadius=showJupiterRadius, survey=survey )
    if wideFormat==True:
        odirExt = 'survey{0}/wideFormat/{1}'.format( surveyModule, SMFlag )
    else:
        odirExt = 'survey{0}/narrowFormat/{1}'.format( surveyModule, SMFlag )
    odir = os.path.join( FIGDIR, odirExt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    
    fig2.savefig( opaths['2a'] )

    # Add the bright predicted TESS planets:
    plotTeqRpTESS( ax2, showSolarSystem=False, showNeptuneRadius=False, \
                   showJupiterRadius=False, SMFlag = SMFlag, z=z )
    
    fig2.savefig( opaths['2b'] )

    # Make a plot with the TESS planets only:
    # titleStr = 'TESS predicted planets (Barclay et al., 2018)'
    # fig3, ax3, ax3Legend = generateBasicAxis( wideFormat=wideFormat, titleStr=titleStr )
    # plotTeqRpTESS( ax3, titleStr=titleStr, showSolarSystem=showSolarSystem, \
    #                showNeptuneRadius=showNeptuneRadius, \
    #                showJupiterRadius=showJupiterRadius, SMFlag = SMFlag, z=z )
    # if showStellarTrack==True:
    #     ax3 = addStellarTrack( ax3 )
    # #opath3 = os.path.join( FIGDIR, onames['3'] )
    # fig3.savefig( opaths['3'] )

    print( '\nSaved:' )
    for k in ['0a','0b','1a','1b','2a','2b','3']:
        print( onames[k] )
  
    return None


#############################################################################
# Utility routines:

def printTopPredictedSubNeptunes( z, onlySubNeptunes=True ):
    ixs = ( z['cad2min']==1 )*( z['RpValRE']>1.5 )*( z['RpValRE']<4 )
    brightLim = 6
    ixs *= ( z['Vmag']>brightLim )*( z['Jmag']>brightLim )*( z['Kmag']>brightLim )
    TSM = z['TSM'][ixs]
    RpRE = z['RpValRE'][ixs]
    MpME = z['MpValME'][ixs]
    aRs = z['aRs'][ixs]
    ideg = z['ideg'][ixs]
    T14hr = z['T14hr'][ixs]
    Pday = z['Pday'][ixs]
    Teq = z['TeqK'][ixs]
    RsRS = z['RsRS'][ixs]
    Ts = z['TstarK'][ixs]
    V = z['Vmag'][ixs]
    J = z['Jmag'][ixs]
    K = z['Kmag'][ixs]
    Tedges = np.arange( 200, 1100, 100 )
    nbins = len( Tedges )-1
    ntop = 5
    if onlySubNeptunes==True:
        print( '\nTop predicted **sub-Neptunes** cooler than 1000K in 100K bins:' )
    else:
        print( '\nTop predicted planets cooler than 1000K in 100K bins:' )
    for i in range( nbins ):
        Tl = Tedges[i]
        Tu = Tedges[i+1]
        ixs1 = ( Teq>=Tl )*( Teq<Tu )
        print( '\n>>>> Between {0:.0f}-{1:.0f}K:'.format( Tl, Tu ) )
        hstr = '         TSM  Mp(ME)  Rp(RE)  Rs(RS)   Ts(K)   Tp(K)     '
        hstr += 'P(d)     aRs   i(deg)  T14(h)      V       J       K'
        print( hstr )
        print( '{0}'.format( 110*'-' ) )
        m = int( ixs1.sum() )
        ixs2 = np.arange( m )[np.argsort( TSM[ixs1] )][::-1]
        n = int( min( [ len( ixs2 ), ntop ] ) )
        for j in range( ntop ):
            ostr = '{0:.0f}. '.format( j+1 ).rjust( 5 )
            ostr += '{0:.1f} '.format( TSM[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.2f} '.format( MpME[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.2f} '.format( RpRE[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.2f} '.format( RsRS[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.0f} '.format( Ts[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.0f} '.format( Teq[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.2f} '.format( Pday[ixs1][ixs2][j] ).rjust( 9 )
            ostr += '{0:.2f} '.format( aRs[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.1f} '.format( ideg[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.1f} '.format( T14hr[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.1f} '.format( V[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.1f} '.format( J[ixs1][ixs2][j] ).rjust( 8 )
            ostr += '{0:.1f} '.format( K[ixs1][ixs2][j] ).rjust( 8 )
            print( ostr )

    return None


def plotTeqRpTESS( ax, showSolarSystem=True, showNeptuneRadius=True, \
                   showJupiterRadius=True, SMFlag = 'TSM', z={}):

    m = 10
    ixs = ( z['cad2min']==1 )*( ( z['Vmag']<m )+( z['Jmag']<m )+( z['Kmag']<m ) )
    SM = z['SM'][ixs]
    Teq = z['TeqK'][ixs]
    RpVal = z['RpValRE'][ixs]
    printTopPredictedSubNeptunes( z )
    Ts = z['TstarK'][ixs]
    n = len( Teq )
    alpha = 1
    ms = 6
    z0 = 1000
    applySMcuts = False
    for i in range( n ):
        c = Utils.getStarColor( Ts[i] )
        # TODO: prior to calling this routine, it might be good
        # to instead create a boolean array for each target specifying
        # whether or not it passes the TSM/ESM threshold for its cell...
        if SMFlag == 'TSM':
            thresholdSM = surveySetup.thresholdTSM( RpVal[i], Teq[i], framework='ACWG' )
        elif SMFlag == 'ESM':
            thresholdSM = surveySetup.thresholdESM(RpVal[i], Teq[i], framework='ACWG')
        if applySMcuts==False: # plotting everything regardless of TSM
            ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                     mfc=c, mec=c, zorder=z0+i )
        elif SM[i]>thresholdSM: # if TSM cuts applied, this one is high enough
            ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                     mfc=c, mec=c, zorder=z0+i )
        else: # otherwise plot as a smaller background point
            
            ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=0.5*ms, alpha=alpha, \
                     mfc=c, mec=c, zorder=0 )
    ax = addSolarSystem( ax, showSolarSystem=showSolarSystem, \
                         showNeptuneRadius=showNeptuneRadius, \
                         showJupiterRadius=showJupiterRadius )
    return ax
    
    
def drawGrid( ax, cgrid=None, zorder=0, survey={} ):
    if cgrid is None:
        cgrid = np.array( [ 201, 148, 199 ] )/256.
    
    Tgrid, Rgrid = survey['gridEdges']( survey['surveyName'] )

    # Number of cells along each axis:
    nT = len( Tgrid )
    nR = len( Rgrid )
    
    for i in range( nT ):
        ax.plot( [Tgrid[i],Tgrid[i]], [Rgrid.min(),Rgrid.max()], '-', \
                 c=cgrid, zorder=zorder )
    for i in range( nR ):
        ax.plot( [Tgrid.min(),Tgrid.max()], [Rgrid[i],Rgrid[i]], '-', \
                 c=cgrid, zorder=zorder )
    return Tgrid, Rgrid


def addSignatureToAxis( ax ):
    c = 0.2*np.ones( 3 )
    ax.text( 0.02, 1, 'Figure by T. Mikal-Evans', fontsize=12, color=c, \
             verticalalignment='top', transform=ax.transAxes, zorder=0 )
    return None


def plotTeqRpGrid( plDict, SMFlag, cgrid=None, titleStr='', \
                   RADecStr='', dateStr='', wideFormat=True, survey={}, \
                   TOIGrid=False, bestInClass=False, \
                   ASCII=False, HeatMap=True, extraNotes=None, \
                   confirmed=False ):
    """
    Plots grid of planets and TOIs by TeqK and RpRE
    SM: (TSM or ESM, list of float)
    pl contains the full planet names that will ultimately be printed in the ASCII output.
    """

    pl = plDict['planetName']
    if bestInClass==True:
    #    # 0's for confirmed planets *not* discovered by TESS, plus TOIs
    #    # 1's only for *confirmed* planets discovered by TESS
        ixsTESS = ( plDict['confirmedTESS']==1 )
        plTESS = list( plDict['planetName'][ixsTESS] )
    #    for i in range(len(plTESS)):
    #        plTESS[i] = plTESS[i].replace(' ', '')
    elif TOIGrid==False:
    #    # If it's not the TOIs, then identify those that were discovered
    #    # by TESS for highlighting in bold on the plots:
    #    TESS = np.array( plDict['TESS'], dtype=int )
    #    plTESS = pl[TESS>0]
        plTESS = []#    plTESS = list(plTESS)
    #    for i in range(len(plTESS)):
    #        plTESS[i] = plTESS[i].replace(' ', '')
    else:
        plTESS = None

    if cgrid is None:
        cgrid = np.array( [ 201, 148, 199 ] )/256.
        
    if not ASCII:
        if HeatMap:
            fig, ax, ax2, axc = generateAxisGrid( wideFormat=wideFormat, \
                                                  titleStr=titleStr, \
                                                  RADecStr=RADecStr, HeatMap=HeatMap )
        else:
            fig, ax, ax2 = generateAxisGrid( wideFormat=wideFormat, titleStr=titleStr, \
                                            RADecStr=RADecStr, HeatMap=HeatMap )
    else:
        ax = None
    Tgrid, Rgrid = survey['gridEdges']( survey['surveyName'] )
    
    nT = len( Tgrid )
    nR = len( Rgrid )
    xLines = np.arange( 0.5, nT+0.5 )
    yLines = np.arange( 0.5, nR+0.5 )
    
    if ASCII:
        plList = addTopSMs( ax, plDict, SMFlag, Tgrid, Rgrid, xLines, yLines, \
                            TOIGrid=TOIGrid, bestInClass=bestInClass, \
                            plTESS=plTESS, survey=survey, ASCII=True )
        return plList, dateStr
    ax, SMstr = addTopSMs(  ax, plDict, SMFlag, Tgrid, Rgrid, xLines, yLines, \
                            TOIGrid=TOIGrid, bestInClass=bestInClass, \
                            plTESS=plTESS, survey=survey, confirmed=confirmed )
    for i in range( nT ):
        ax.plot( [xLines[i],xLines[i]], [yLines.min(),yLines.max()], '-', \
                 c=cgrid, zorder=1 )
    for i in range( nR ):
        ax.plot( [xLines.min(),xLines.max()], [yLines[i],yLines[i]], '-', \
                 c=cgrid, zorder=1 )
    
    xtxt = 0.04
    if HeatMap:
        # Creates a new custom colormap
        cdict = { 'red':[    [0, 1, 204/255],
                             [1/6, 204/255, 153/255],
                             [1/2, 153/255, 1],
                             [1, 1, 1]],                
                  'green':[  [0, 1, 153/255],
                             [1/6, 153/255, 204/255],
                             [2/6, 204/255, 1],
                             [4/6, 1, 204/255],
                             [5/6, 204/255, 153/255],
                             [0.999, 153/255, 1],
                             [1, 1, 1]],                    
                  'blue':[    [0, 1, 1],
                              [2/6, 1, 153/255],
                              [0.999, 153/255, 1],
                              [1, 1, 1]] }
        cmap = matplotlib.colors.LinearSegmentedColormap( 'testCmap', \
                                                          segmentdata=cdict, N=256 )
        print( '\n\naddHeatMap() routine must be updated to take dictionary as input\n\n' )
        pdb.set_trace()
        ax, val = addHeatMap(ax, xLines, yLines, TeqK, RpRE, Tgrid, Rgrid, cmap)
        addColorBar(axc, val, cmap)
    
    formatAxisTicks( ax )
    ax.xaxis.set_ticks( xLines, minor=False )
    ax.yaxis.set_ticks( yLines, minor=False )
    ax.set_xticklabels( Tgrid )
    ax.set_yticklabels( Rgrid )
    if wideFormat==False:
        subtitleY = 0.94
        dySubTitle = 0.01
    else:
        subtitleY = 0.925
        dySubTitle = 0.015
    #fig.text( xtxt, subtitleY, SMstr, c='green', fontsize=14, \
    #          horizontalalignment='left', verticalalignment='bottom' )
   
    #otherNotes = '{0} values are listed in square brackets.'.format( SMFlag )
    #if bestInClass==False: # bestInClass don't print TSM/ESM uncertainties
    #    otherNotes += ' NOTE: Quoted $1\\sigma$ uncertainties'
    #    otherNotes += ' are approximate\nand only account for the uncertainty in the stellar and planetary radii.'
    #otherNotes += '\nAsterisks indicate potential top-5 in box, based on Kunimoto et al. (2022) predicted yield for Y1-7.'
    #if bestInClass==True:
    #    otherNotes += '\nFor reference, values listed on right in purple are top-5 {0} values predicted for that box.'.format( SMFlag )
    #if extraNotes is not None:
        #otherNotes += '\n{0}'.format( extraNotes )
    #    otherNotes += '{0}'.format( extraNotes )
    #fig.text( xtxt, subtitleY-dySubTitle, otherNotes, c='black', \
    #          fontsize=14, horizontalalignment='left', verticalalignment='top' )
    if bestInClass==True:
        print('legend commented out line 1159 in surveyGrids.py')
        Utils.legendBestInClass( fig )
    dx = 0.02*( xLines.max()-xLines.min() )
    dy = 0.03*( yLines.max()-yLines.min() )
    ax.set_xlim( [ xLines.min()-dx, xLines.max()+dx ] )
    ax.set_ylim( [ yLines.min()-dy, yLines.max()+dy ] )

    return fig, ax




def formatAxisTicks( ax ):
    tick_fs = 14
    tl = 10
    tlm = 5
    tw = 2
    ax.spines['bottom'].set_linewidth( tw )
    ax.spines['left'].set_linewidth( tw )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params( labelsize=tick_fs )
    ax.xaxis.set_tick_params( length=tl, width=tw, which='major' )
    ax.xaxis.set_tick_params( length=tlm, width=tw, which='minor' )
    ax.yaxis.set_tick_params( length=tl, width=tw, which='major' )
    ax.yaxis.set_tick_params( length=tlm, width=tw, which='minor' )
    return ax


def addTopSMs( ax, plDict, SMFlag, Tgrid, Rgrid, xLines, yLines, bestInClass=False, \
               TOIGrid=False, plTESS=None, survey={}, ASCII=False, confirmed=False ):

    """
    Supplementary routine to graph planets and TOIs with top SM values in each grid section

    Parameters:
    ax: figure axes from other supplementary routines
    plDict: Dictionary of arrays containing the planet+star properties
    pl, TeqK, RpRE, TstarK: planet values 
    SM: (TSM or ESM, list of values)
    Tgrid, Rgrid: values of TeqK and RpRE grid section values 
    xLines, yLines: x and y values for graph 
    survey: dictionary of survey values
    """

    # Unpack dictionary into necessary arrays:
    pl = plDict['planetName']
    TeqK = plDict['TeqK']
    TstarK = plDict['TstarK']
    RpRE = plDict['RpValRE']
    SMVals = plDict['SM']
    AMVals = plDict['AM']
    MagVals = plDict['Mag']
    if bestInClass==True:
        sMass = plDict['statusMass']
    else:
        sMass = None
               
    plNames = []    
    framework = survey['framework']
    nx = len( xLines )-1 # Number of lines on x axis
    ny = len( yLines )-1 # Number of lines on y axis
    n = len( pl ) # Number of planets
    ixs0 = np.arange( n )
    if nx>4:
        text_fs = 11
        #ndx = 27
    elif nx<=4:
        text_fs = 16
        #ndx = 20
    nList = 5
    top5Predicted = Utils.medianSimulation_Kunimoto2022( survey['surveyName'], SMFlag )
    for i in range( nx ): # loop over temperature columns
        ixsi = ( TeqK>=Tgrid[i] )*( TeqK<Tgrid[i+1] ) # Show if within the temperature range
        xsymb = xLines[i] + 0.05*( xLines[i+1]-xLines[i] )
        dx = ( xLines[i+1]-xLines[i] )
        xtxt = xLines[i] + 0.12*dx
        xtxtR = xLines[i+1] - 0.03*dx

        TeqKi = 0.5*( Tgrid[i]+Tgrid[i+1] )
        for j in range( ny ): # loop over radius rows
            # Indices inside box above the TSM threshold:
            RpREj = 0.5*( Rgrid[j]+Rgrid[j+1] )

            #Find the threshold SM for the cell
            if SMFlag == 'TSM':
                SMj, SMstr = survey['thresholdTSM']( RpREj, framework=framework )
                AMj, AMstr = survey['thresholdTS']( RpREj, framework=framework )
                Magj, Magstr = survey['thresholdJmag']( RpREj, framework=framework )
            elif SMFlag == 'ESM':
                SMj, SMstr = survey['thresholdESM']( RpREj, framework=framework )
                AMj, AMstr = survey['thresholdSE']( RpREj, framework=framework )
                Magj, Magstr = survey['thresholdKmag']( RpREj, framework=framework )

            ixsj = ( RpRE>=Rgrid[j] )*( RpRE<Rgrid[j+1] ) # Show if within the radius range
            # Show if in the cell and SM higher than threshold:
            ixsij = ixs0[ixsi*ixsj*( SMVals>SMj )*( AMVals>AMj )*( MagVals>Magj )] 
            nij = len( ixsij ) # Number in cell higher than threshold
            if nij>0:
                # Order by decreasing SM:
                ixso = np.argsort( SMVals[ixsij] ) 
                ixs = ixsij[ixso][::-1] 
                
                nwrite = min( [ nij, nList ] ) # number above threshold in cell or 5
                dy = ( yLines[j+1]-yLines[j] )/float(nList+0.5)
                y0 = yLines[j]+4.8*dy

                # Old implementation by Trey Burga using Barclay et al. (2018) predictions:
                #predSM = getFifthPredicted( SMFlag, Rgrid[j+1], Rgrid[j], \
                #                            Tgrid[i+1], Tgrid[i] )
                top5ij = top5Predicted[:,j,i]
                predSM = top5ij[-1]
                #print( top5Predicted[:,j,i] )
                #pdb.set_trace()
                ixs = ixs[:nwrite] # TESTING 2022-07-28, I think this should work fine...
                # 2022-07-28: 
                # This is a VERY hacked up routine with a stupid-long argument list,
                # created to handle the BestInClass plots, because it does an additional
                # sorting operation within each grid cell according to the mass status
                # of the planet. In the future, this should be cleaned up...
                # 2022-08-30:
                # Been requested (by Eliza) not to sort according to mass status, so maybe
                # this input argument list doesn't have to be so long? Can't remember what
                # the hacking comment above referred to though.
                #print(plNames)
                #barf
                plNames = Utils.writeToGridCell( ax, plNames, nwrite, y0, dy, ixs, pl, \
                                                 plTESS, TstarK, TOIGrid, SMFlag, plDict, \
                                                 SMVals, top5ij, ASCII, bestInClass, \
                                                 sMass, xtxt, xtxtR, text_fs, xsymb, \
                                                 confirmed)
    #print(plNames)            
    if ASCII:
        return plNames
    else:
        return ax, SMstr
    

def generateAxisScatter( xlim=[0,3100], ylim=[0,26], wideFormat=False, \
                         whichType='RpTeq', titleStr='', DecStr='', \
                         showLegend=True ):
    fig, ax, axLegend, axc = generateAxes( wideFormat=wideFormat, whichType=whichType, \
                                           showLegend=showLegend )
    title_fs = 18
    toplineY = 0.98
    fig.text( 0.02, toplineY-0.02, titleStr, fontsize=title_fs, weight='heavy', \
              rotation=0, horizontalalignment='left', verticalalignment='bottom' )
    return fig, ax, axLegend


def generateAxisGrid( xlim=[0,3100], ylim=[0,26], wideFormat=False, whichType='RpTeq', \
                      RADecStr='', titleStr='', showLegend=True, HeatMap=False ):
    if HeatMap:
        fig, ax, axLegend, axc = generateAxes( wideFormat=wideFormat, whichType=whichType, \
                                               showLegend=showLegend, HeatMap=HeatMap )
    else:
        fig, ax, axLegend, axc = generateAxes( wideFormat=wideFormat, whichType=whichType, \
                                               showLegend=showLegend, HeatMap=HeatMap )
    title_fs = 18
    toplineY = 0.98
    fig.text( 0.02, toplineY-0.02, titleStr, fontsize=title_fs, weight='heavy', \
              rotation=0, horizontalalignment='left', verticalalignment='bottom' )
    subtitle_fs = 14
    if HeatMap:
        label_fs = 14
        cb_text = 'Fraction of TOIs vs Fraction of Predicted'
        axc.text( 0, 2, cb_text, fontsize=label_fs, \
                horizontalalignment='left', verticalalignment='center', \
                rotation=0, transform=axc.transAxes ) # Creates the color bar label
        return fig, ax, axLegend, axc
    else:
        fig.text( 0.98, toplineY, RADecStr, fontsize=subtitle_fs, weight='normal', \
                rotation=0, horizontalalignment='right', verticalalignment='top' )
        return fig, ax, axLegend


def generateAxes( wideFormat=True, whichType='RpTeq', showLegend=True, HeatMap=False ):
    if wideFormat==False:
        fig = plt.figure( figsize=[11,9] )
        xlow = 0.09
        ylow = 0.08
        axh = 0.8
        axw = 0.93
        dxl = 0.06
        xlow2 = xlow+0.5*axw
        ylow2 = ylow+axh+0.005
        axw2 = 0.5*axw
        subtitleY = 0.94
        dyNewLine = 0.01
    else:
        fig = plt.figure( figsize=[12.5,9] )#18,9] )
        xlow = 0.07#0.055#0.064
        ylow = 0.085
        axh = 0.715
        axw = 0.9#0.94#0.93
        dxl = 0.036
        xlow2 = xlow+0.7*axw
        ylow2 = ylow+axh+0.02
        axw2 = 0.25*axw
        subtitleY = 0.925
        dySubTitle = 0.015
    ax = fig.add_axes( [ xlow, ylow, axw, axh ] )
    if showLegend==True:
        axLegend = fig.add_axes( [ xlow2, ylow2, axw2, 0.09*axh ] )
        addStellarSpectralTypeLegend( axLegend, ms=8, text_fs=10 )
    else:
        axLegend = None

    if HeatMap:
        axc = fig.add_axes([xlow2, ylow2+0.125, axw2, 0.015*axh]) #Colorbar axis

    ax = formatAxes( ax, whichType=whichType )
    label_fs = 16
    if whichType=='RpTeq':
        fig.text( xlow-dxl, ylow+0.5*axh, '$R_p$ ($R_E$)', fontsize=label_fs, \
                  rotation=90, horizontalalignment='right', verticalalignment='center' )
        fig.text( xlow+0.5*axw, 0.001, '$T_{\\rm{eq}}$ (K)', fontsize=label_fs, \
                  rotation=0, horizontalalignment='center', verticalalignment='bottom' )
    elif ( whichType=='RpInsolLog' )+( whichType=='RpInsol' ):
        fig.text( xlow-dxl, ylow+0.5*axh, '$R_p$ ($R_E$)', fontsize=label_fs, \
                  rotation=90, horizontalalignment='right', verticalalignment='center' )
        fig.text( xlow+0.5*axw, 0.001, 'Insolation relative to Earth', rotation=0,
                  fontsize=label_fs, horizontalalignment='center', \
                  verticalalignment='bottom' )
    if showLegend==True:
        subtitle_fs = 14
        subtitleStr = 'Circles indicate host star spectral type'
        if wideFormat==True:
            sptY = 0.87
        else:
            sptY = 0.90
        fig.text( xlow2+0.5*axw2, sptY, subtitleStr, fontsize=subtitle_fs, \
                  horizontalalignment='center', verticalalignment='bottom', \
                  weight='normal', rotation=0 )
    if HeatMap:
        return fig, ax, axLegend, axc
    else:
        return fig, ax, axLegend, None


def formatAxes( ax, whichType='RpTeq', xlim='default', ylim='default', \
                xticksMajor='default', yticksMajor='default', \
                xticksMinor='default', yticksMinor='default' ):
                
    tick_fs = 14
    tl = 10
    tlm = 5
    tw = 2
    if whichType=='RpTeq':
        if xlim is 'default':
            xlim = [ 0, 3100 ]
        if ylim is 'default':
            ylim = [ 0, 26 ]            
        if xticksMinor is 'default':
            xticksMinor = np.arange( 0, 4000, 100 )
        if xticksMajor is 'default':
            xticksMajor = np.arange( 0, 4000, 500 )
        if yticksMinor is 'default':
            yticksMinor = np.arange( 0, 30, 1 )
        if yticksMajor is 'default':
            yticksMajor = np.arange( 0, 30, 2 )
    elif whichType=='RpInsol':
        if xlim is 'default':
            xlim = [ 0, 11e3 ]
        if ylim is 'default':
            ylim = [ 0, 26 ]            
        if xticksMinor is 'default':
            xticksMinor = np.arange( 0, 10500, 100 )
        if xticksMajor is 'default':
            xticksMajor = np.arange( 0, 10500, 1000 )            
        if yticksMinor is 'default':
            yticksMinor = np.arange( 0, 30, 1 )
        if yticksMajor is 'default':
            yticksMajor = np.arange( 0, 30, 2 )
    elif whichType=='RpInsolLog':
        ax.set_xscale( 'log' )
        ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter( tickLogFormat ) )
        if xlim is 'default':
            xlim = [ 0.1, 11e3 ]
        if ylim is 'default':
            ylim = [ 0, 26 ]            
        if xticksMinor is 'default':
            
            xticksMinor = np.arange( 0.1, 1, 0.1 )
            xticksMinor = np.concatenate( [ xticksMinor, np.arange( 1, 10, 1 ) ] )
            xticksMinor = np.concatenate( [ xticksMinor, np.arange( 10, 100, 10 ) ] )
            xticksMinor = np.concatenate( [ xticksMinor, np.arange( 100, 1000, 100 ) ] )
            xticksMinor = np.concatenate( [ xticksMinor, np.arange( 1000, 10000, 1000 ) ] )
        if xticksMajor is 'default':
            
            xticksMajor = np.logspace( -1, 4, 6 )
        if yticksMinor is 'default':
            yticksMinor = np.arange( 0, 30, 1 )
        if yticksMajor is 'default':
            yticksMajor = np.arange( 0, 30, 2 )
    else:
        pdb.set_trace()

    ax.xaxis.set_ticks( xticksMinor, minor=True )
    ax.xaxis.set_ticks( xticksMajor, minor=False )
    ax.yaxis.set_ticks( yticksMinor, minor=True )
    ax.yaxis.set_ticks( yticksMajor, minor=False )
        
    ax.tick_params( labelsize=tick_fs )
    ax.xaxis.set_tick_params( length=tl, width=tw, which='major' )
    ax.xaxis.set_tick_params( length=tlm, width=tw, which='minor' )
    ax.yaxis.set_tick_params( length=tl, width=tw, which='major' )
    ax.yaxis.set_tick_params( length=tlm, width=tw, which='minor' )

    ax.spines['bottom'].set_linewidth( tw )
    ax.spines['left'].set_linewidth( tw )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim( ylim )
    ax.set_xlim( xlim )
    return ax

def tickLogFormat( y, pos ):
    # Find the number of decimal places required
    decimalplaces = int(np.ceil(np.maximum(-np.log10(y),0))) # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


    

def plotTeqRpScatter( plDict, SMFlag, ms=8, alpha=1, \
                      starColors=True, applySMcuts=False, survey={}, \
                      showGrid=True, indicateTESS=False, showSolarSystem=False, \
                      showStellarTrack=False, showNeptuneRadius=True, \
                      showJupiterRadius=True, titleStr='', wideFormat=False, \
                      dateStr='' ):
    """
    Creates a scatter plot of planets on a (Teq, Rp) graph

    Parameters:
    planetNames: list of planet names (str) (currently unused)
    Teq, RpVal, Ts, TESS: lists of planet values (float)
    SM: (TSM or ESM, list of values)
    survey: dictionary of survey properties
    """

    
    planetNames = plDict['planetName']
    Teq = plDict['TeqK']
    RpVal = plDict['RpValRE']
    Ts = plDict['TstarK']
    SM = plDict['SM']
    #if TOIGrid==False:
    if 1: # will need to test if this works for TOIs... do they have a 'TESS' array?
        TESS = np.array( plDict['TESS'], dtype=int )
        plTESS = planetNames[TESS>0]
        plTESS = list(plTESS)
        for i in range(len(plTESS)):
            plTESS[i] = plTESS[i].replace(' ', '')
    #else:
    #    plTESS = None

    
    n = len( Teq )
    nTESS = np.sum( TESS )
    cTESS = np.array( [ 213, 128, 255 ] )/256.
    zTESS = 2*n
    z0 = 200
    msTESS = 2*ms
    
    fig, ax, ax2 = generateAxisScatter( wideFormat=wideFormat, titleStr=titleStr )
    if showGrid==True:
        Tgrid, Rgrid = drawGrid( ax, survey=survey, zorder=1 )
    framework = survey['framework']
        
    c0 = 0.9*np.ones( 3 )
    for i in range( n ): # loop over each exoplanet
        if starColors==True:
            c = Utils.getStarColor( Ts[i] )
        else:
            c = c0
        
        if SMFlag == 'TSM':
            #SMi, SMstr = survey['thresholdTSM']( RpVal[i], Teq[i], framework=framework )
            SMi, SMstr = survey['thresholdTSM']( RpVal[i], framework=framework )
        elif SMFlag == 'ESM':
            #SMi, SMstr = survey['thresholdESM']( RpVal[i], Teq[i], framework=framework )
            SMi, SMstr = survey['thresholdESM']( RpVal[i], framework=framework )

        if applySMcuts==False: # plotting everything regardless of SM
            if ( indicateTESS==True )*( TESS[i]==1 ):
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                         mfc=c, mec=c, zorder=zTESS+nTESS+i )
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=msTESS, alpha=alpha, \
                         zorder=zTESS, mfc=cTESS, mec=cTESS )
            else:
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                         mfc=c, mec=c, zorder=z0+i )
            
        elif SM[i]>SMi: # if SM cuts applied, this one is high enough
            if ( indicateTESS==True )*( TESS[i]==1 ):
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                         mfc=c, mec=c, zorder=zTESS+nTESS+i )
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=msTESS, alpha=alpha, \
                         zorder=zTESS, mfc=cTESS, mec=cTESS )
            else:
                ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=ms, alpha=alpha, \
                         mfc=c, mec=c, zorder=z0+i )
        else: # otherwise plot as a smaller background point
            ax.plot( [Teq[i]], [RpVal[i]], 'o', ms=0.5*ms, alpha=alpha, \
                     mfc=c0, mec=c0, zorder=0 )
    if showStellarTrack==True:
        ax = addStellarTrack( ax )
    ax = addSolarSystem( ax, showSolarSystem=showSolarSystem, \
                         showNeptuneRadius=showNeptuneRadius, \
                         showJupiterRadius=showJupiterRadius )
    if wideFormat==False:
        subtitleY = 0.94
        dyNewLine = 0.01
    else:
        subtitleY = 0.925
        dySubTitle = 0.015
    if applySMcuts==True:
        xtxt = 0.04
        fig.text( xtxt, subtitleY, SMstr, c='green', fontsize=14, \
                  horizontalalignment='left', verticalalignment='bottom' )
        otherNotes = 'Grey points do not meet {0} thresholds'.format( SMFlag )
        fig.text( xtxt, subtitleY-dySubTitle, otherNotes, c='black', \
                  fontsize=14, horizontalalignment='left', verticalalignment='top' )
        
    return fig, ax

def addStellarTrack( ax ):
    Rs0, Ts0 = Utils.readStellarTrack()
    Ts = np.linspace( Ts0.min(), Ts0.max(), 1000 )
    Rs = np.interp( Ts, Ts0, Rs0 )
    ixs1 = ( Ts>2200 )
    ixs2 = ( Ts<=2200 )*( Ts>1300 )
    ixs3 = ( Ts<=1300 )
    lw = 5
    c1 = np.array( [178,24,43] )/256. #Cyan
    c2 = np.array( [191,129,45] )/256. #'HotPink'
    c3 = np.array( [140,81,10] )/256. #'GreenYellow'
    ax.plot( Ts[ixs1], Rs[ixs1], '-', c=c1, lw=lw, zorder=1e5 )
    ax.plot( Ts[ixs2], Rs[ixs2], '-', c=c2, lw=lw, zorder=1e5 )
    ax.plot( Ts[ixs3], Rs[ixs3], '-', c=c3, lw=lw, zorder=1e5 )
    return ax

def addSolarSystem( ax, showSolarSystem=True, showNeptuneRadius=True, \
                    showJupiterRadius=True ):
    ss = Utils.solarSystem()
    if showSolarSystem==True:
        ssc = 'HotPink'
        ssc = np.array( [0,204,0] )/256.
        ssPlanets = list( ss['TeqK'].keys() )
        ssms = 8
        for k in ssPlanets:
            ax.plot( [ss['TeqK'][k]], [ss['RpRE'][k]], 'o', ms=ssms, \
                     zorder=1e6, mfc=ssc, mec=ssc, alpha=0.5 )
            ax.plot( [ss['TeqK'][k]], [ss['RpRE'][k]], 'o', ms=ssms, \
                     zorder=1e6, mfc='none', mec=ssc, mew=2, alpha=1 )
    cgrey = 0.6*np.ones( 3 )
    if showNeptuneRadius==True:        
        ax.axhline( ss['RpRE']['Neptune'], ls='--', c=cgrey, lw=2, zorder=0 )
    if showJupiterRadius==True:
        ax.axhline( ss['RpRE']['Jupiter'], ls='--', c=cgrey, lw=2, zorder=0 )
    return ax

def addStellarSpectralTypeLegend( ax, ms=10, text_fs=12 ):
    c, SpTs = Utils.getAllStarColors()
    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)
    plt.setp( ax.xaxis.get_ticklabels(), visible=False )
    plt.setp( ax.yaxis.get_ticklabels(), visible=False )
    n = len( SpTs )
    for i in range( n ):
        k = SpTs[i]
        ax.plot( i+1, 0.5, 'o', ms=ms, mfc=c[k], mec=c[k] )
        ax.text( i+1, 0.4, k, rotation=45, fontsize=text_fs, \
                 horizontalalignment='right', verticalalignment='top' )
    ax.set_ylim( [ 0, 1 ] )
    ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False)
    ax.tick_params(axis = 'y', which = 'both', left = False, right = False)
    return None
        


def readConfirmedProperties( ipath='confirmedProperties.pkl', SMFlag='TSM' ):
    """
    Returns properties for all confirmed planets. For planets with
    available radius but no mass measurement, an empirical relation
    is used to estimate the mass. This has been done for the purpose
    of identifying targets for a UV host star survey, which is why
    it's OK if the planet masses haven't been published.
    """
    ifile = open( ipath, 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    planetName = z['planetName']
    RA = z['RA']
    Dec = z['Dec']
    RA_deg = z['RA_deg']
    Dec_deg = z['Dec_deg']
    RsRS = z['RsRS']
    aAU = z['aAU']
    TeqK = z['TeqK']
    Insol = z['Insol']
    TstarK = z['TstarK']
    T14hr = z['T14hr']
    b = z['b']
    RpRs = z['RpRs']
    Vmag = z['Vmag']
    Jmag = z['Jmag']
    Hmag = z['Hmag']
    Kmag = z['Kmag']
    RpValRE = z['RpValRE']
    RpLsigRE = z['RpLowErrRE']
    RpUsigRE = z['RpUppErrRE']
    MpValME = z['MpValME']
    MpLsigME = z['MpLowErrME']
    MpUsigME = z['MpUppErrME']
    TESS = z['discoveredByTESS']
    #Pday = z['pl_orbper'] #2-5-24

    if SMFlag == 'TSM':
        SM = z['TSM']
        AM = z['transitSignal']
        Mag = z['Jmag']
    elif SMFlag == 'ESM':
        SM = z['ESM']
        AM = z['eclipseDepth']
        Mag = z['Kmag']

    n0 = len( planetName )
    ixs = np.arange( n0 )[np.isnan( MpValME )*np.isfinite( RpValRE )]
    n = len( ixs )
    for i in range( n ):
        MpValME[ixs[i]] = Utils.planetMassFromRadius( RpValRE[ixs[i]] )
        MpUncFill = 1e9 #min( [ 0.1, MpValME[ixs[i]]/5.01 ] )
        MpLsigME[ixs[i]] = MpUncFill
        MpUsigME[ixs[i]] = MpUncFill
    
    if SMFlag == 'TSM':
        SM[ixs] = Utils.computeTSM( RpValRE[ixs], MpValME[ixs], \
                                    RsRS[ixs], TeqK[ixs], Jmag[ixs] )
        AM[ixs] = Utils.computeTransSignal( RpRs[ixs], RpValRE[ixs], TeqK[ixs], MpValME[ixs])
    elif SMFlag == 'ESM':
        #RpSI = RpValRE*Utils.REARTH_SI
        #RsSI = RsRS*Utils.RSUN_SI
        #RpRs = RpSI/RsSI
        #RpRs = ( RpValRE*Utils.REARTH_SI )/( RsRS*Utils.RSUN_SI )
        SM[ixs] = Utils.computeESM( TeqK[ixs], RpRs[ixs], TstarK[ixs], Kmag[ixs] )
        AM[ixs] = Utils.computeEclipseDepth( TeqK[ixs], RpRs[ixs], TstarK[ixs] )

    ixs = np.isfinite( TeqK )*np.isfinite( SM )*np.isfinite( RpValRE )
    print( '\nReading in {0:.0f} planets total.'.format( n0 ) )
    print( 'Returning {0:.0f} planets with radii, {1}, and Teq values.'\
           .format( ixs.sum(), SMFlag ) )
    outp = { 'planetName':planetName[ixs], 'TESS':TESS[ixs], \
             'RA':RA[ixs], 'Dec':Dec[ixs], 'RA_deg':RA_deg[ixs], 'Dec_deg':Dec_deg[ixs], \
             'SM':SM[ixs], 'AM':AM[ixs], 'Mag':Mag[ixs], 'T14hr':T14hr[ixs], \
             'Vmag':Vmag[ixs], 'Jmag':Jmag[ixs], 'Hmag':Hmag[ixs], 'Kmag':Kmag[ixs], \
             'b':b[ixs], 'RpRs':RpRs[ixs], 'TeqK':TeqK[ixs], 'Insol':Insol[ixs], \
             'TstarK':TstarK[ixs], 'RsRS':RsRS[ixs], 'aAU':aAU[ixs], #'Pday':Pday[ixs], \
             'RpValRE':RpValRE[ixs], 'RpLsigRE':RpLsigRE[ixs], 'RpUsigRE':RpUsigRE[ixs], \
             'MpValME':MpValME[ixs], 'MpLsigME':MpLsigME[ixs], 'MpUsigME':MpUsigME[ixs] }
    return outp, z0['dateStr']


def readTOIProperties( ipath='toiProperties.pkl', SMFlag='TSM' ):
    """
    """
    ifile = open( ipath, 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    z = Utils.fixAnomalousRpRs( z )
    planetName = z['planetName']
    RA = z['RA_deg']
    RAhr = RA*(24/360.)
    Dec = z['Dec_deg']
    RsRS = z['RsRS']
    RsUncRS = np.row_stack( [ np.abs( z['RsLowErrRS'] ), np.abs( z['RsUppErrRS'] ) ] )
    RsUncRS = np.max( RsUncRS, axis=0 )
    TeqK = z['TeqK']
    Jmag = z['Jmag']
    Kmag = z['Kmag']
    TstarK = z['TstarK']
    RpValRE = z['RpValRE']
    RpUncRE = np.row_stack( [ np.abs( z['RpLowErrRE'] ), np.abs( z['RpUppErrRE'] ) ] )
    RpUncRE = np.max( RpUncRE, axis=0 )
    MpValME = z['MpValME']
    RpRs = z['RpRs']
    Jmag = z['Jmag']
    aRs = z['aRs']
    TeqK_exofop = z['TeqK_exofop']

    if SMFlag == 'TSM':
        SM = z['TSM']
        AM = z['transitSignal']
        Mag = z['Jmag']
    elif SMFlag == 'ESM':
        SM = z['ESM']
        AM = z['eclipseDepth']
        Mag = z['Kmag']

    ixs = np.isfinite( TeqK )*np.isfinite( SM )*np.isfinite( RpValRE )
    outp = { 'planetName':planetName[ixs], 'SM':SM[ixs], 'AM':AM[ixs], 'Mag':Mag[ixs], \
             'RpRs':RpRs[ixs], 'RA_deg':RA[ixs], 'RA_hr':RAhr[ixs], \
             'Dec_deg':Dec[ixs], 'TeqK':TeqK[ixs], 'TstarK':TstarK[ixs], \
             'Jmag':Jmag[ixs], 'Kmag':Kmag[ixs], \
             'RsRS':RsRS[ixs], 'RsUncRS':RsUncRS[ixs], 'aRs':aRs[ixs], \
             'RpValRE':RpValRE[ixs], 'RpUncRE':RpUncRE[ixs], \
             'MpValME':MpValME[ixs], 'TeqK_exofop': TeqK_exofop[ixs]}
    return outp, z0['dateStr']


def readNoMassTESSProperties():
    """
    Returns properties of confirmed TESS planets lacking published masses.
    """
    ifile = open( 'confirmedProperties.pkl', 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    planetName = z['planetName']
    RsRS = z['RsRS']
    TeqK = z['TeqK']
    Jmag = z['Jmag']
    TstarK = z['TstarK']
    TSM = z['TSM']
    RpValRE = z['RpValRE']
    RpLsigRE = z['RpLowErrRE']
    RpUsigRE = z['RpUppErrRE']
    MpValME = z['MpValME']
    MpLsigME = z['MpLowErrME']
    MpUsigME = z['MpUppErrME']
    TESS = z['discoveredByTESS']
    print( '\nReading confirmed TESS planets lacking peer-reviewed published masses:' )
    ixs = np.isfinite( TeqK )*np.isfinite( RpValRE )*np.isnan( MpValME )*( TESS==1 )
    n = ixs.sum()
    for i in range( n ):
        print( '{0:.0f}. {1}'.format( i+1, planetName[ixs][i] ) )
    print( 'Returning {0:.0f} planets with measured radii and Teq values.'\
           .format( ixs.sum() ) )
    # Add in the estimated masses and then use these to compute estimated TSM values
    # (ESM values should already be included as it doesn't require the mass):
    MpValME = np.zeros( n )
    for i in range( n ):
        MpValME[i] = Utils.planetMassFromRadius( RpValRE[ixs][i] )
    TSM = Utils.computeTSM( RpValRE[ixs], MpValME, RsRS[ixs], TeqK[ixs], Jmag[ixs] )
    print( 'Masses taken from empirical relation and used for TSM calculation.\n' )
    outp = { 'planetName':planetName[ixs], 'TESS':TESS[ixs], \
             'TeqK':TeqK[ixs], 'TstarK':TstarK[ixs], 'RsRS':RsRS[ixs], \
             'RpValRE':RpValRE[ixs], 'RpLsigRE':RpLsigRE[ixs], 'RpUsigRE':RpUsigRE[ixs], \
             'MpValME':MpValME, 'TSM':TSM }
    return outp


def readConfirmedTESSProperties( publishedMasses=True, SMFlag = 'TSM' ):
    """
    Returns properties of confirmed TESS planets.
    """
    ifile = open( 'confirmedProperties.pkl', 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    z = Utils.fixAnomalousRpRs( z )
    planetName = z['planetName']
    RsRS = z['RsRS']
    TeqK = z['TeqK']
    Jmag = z['Jmag']
    Kmag = z['Kmag']
    TstarK = z['TstarK']
    RpValRE = z['RpValRE']
    RpLsigRE = z['RpLowErrRE']
    RpUsigRE = z['RpUppErrRE']
    MpValME = z['MpValME']
    MpLsigME = z['MpLowErrME']
    MpUsigME = z['MpUppErrME']
    TESS = z['discoveredByTESS']

    if SMFlag == 'TSM':
        SM = z['TSM']
    elif SMFlag == 'ESM':
        SM = z['ESM']

    if publishedMasses==True:
        print( '\nReading confirmed TESS planets with peer-reviewed published masses:' )
        ixs = np.isfinite( TeqK )*np.isfinite( RpValRE )*np.isfinite( MpValME )*( TESS==1 )
        n = ixs.sum()
        for i in range( n ):
            print( '{0:.0f}. {1}'.format( i+1, planetName[ixs][i] ) )

        print( 'Returning {0:.0f} planets with measured radii, {1}, and Teq values.'\
               .format( ixs.sum(), SMFlag ) )

        outp = { 'planetName':planetName[ixs], 'TESS':TESS[ixs], \
                 'TeqK':TeqK[ixs], 'TstarK':TstarK[ixs], 'RsRS':RsRS[ixs], \
                 'RpValRE':RpValRE[ixs], 'RpLsigRE':RpLsigRE[ixs], \
                 'RpUsigRE':RpUsigRE[ixs], 'MpValME':MpValME[ixs], 'SM':SM[ixs] }        
    else:
        print( '\nReading confirmed TESS planets lacking peer-reviewed published masses:' )
        ixs = np.isfinite( TeqK )*np.isfinite( RpValRE )*np.isnan( MpValME )*( TESS==1 )
        n = ixs.sum()
        for i in range( n ):
            print( '{0:.0f}. {1}'.format( i+1, planetName[ixs][i] ) )
        
        print( 'Returning {0:.0f} planets with measured radii and Teq values.'\
               .format( ixs.sum() ) )
        
        if SMFlag == 'TSM':
            # Add in the estimated masses and then use these to compute estimated TSM values
            # (ESM values should already be included as it doesn't require the mass):
            MpValME = np.zeros( n )
            for i in range( n ):
                MpValME[i] = Utils.planetMassFromRadius( RpValRE[ixs][i] )

            SM = Utils.computeTSM( RpValRE[ixs], MpValME, RsRS[ixs], \
                                    TeqK[ixs], Jmag[ixs] )
            print( 'Masses taken from empirical relation and used for TSM calculation.\n' )

        if SMFlag == 'ESM':
            RpRs = RpValRE/RsRS
            SM = Utils.computeESM( TeqK[ixs], RpRs[ixs], TstarK[ixs], Kmag)

        outp = { 'planetName':planetName[ixs], 'TESS':TESS[ixs], \
                 'TeqK':TeqK[ixs], 'TstarK':TstarK[ixs], 'RsRS':RsRS[ixs], \
                 'RpValRE':RpValRE[ixs], 'RpLsigRE':RpLsigRE[ixs], \
                 'RpUsigRE':RpUsigRE[ixs], 'MpValME':MpValME, 'SM':SM[ixs] }
    return outp


def readWithMassTESSProperties():
    """
    Returns properties of confirmed TESS planets lacking published masses.
    ESM not implemented because no references
    """

    ifile = open( 'confirmedProperties.pkl', 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    z = Utils.fixAnomalousRpRs( z )
    planetName = z['planetName']
    RsRS = z['RsRS']
    TeqK = z['TeqK']
    Jmag = z['Jmag']
    TstarK = z['TstarK']
    TSM = z['TSM']
    RpValRE = z['RpValRE']
    RpLsigRE = z['RpLowErrRE']
    RpUsigRE = z['RpUppErrRE']
    MpValME = z['MpValME']
    MpLsigME = z['MpLowErrME']
    MpUsigME = z['MpUppErrME']
    TESS = z['discoveredByTESS']
    print( '\nReading confirmed TESS planets lacking peer-reviewed published masses:' )
    ixs = np.isfinite( TeqK )*np.isfinite( RpValRE )*np.isfinite( MpValME )*( TESS==1 )
    n = ixs.sum()
    for i in range( n ):
        print( '{0:.0f}. {1}'.format( i+1, planetName[ixs][i] ) )

    print( 'Returning {0:.0f} planets with measured radii, TSM, and Teq values.'\
           .format( ixs.sum() ) )
    
    outp = { 'planetName':planetName[ixs], 'TESS':TESS[ixs], \
             'TeqK':TeqK[ixs], 'TstarK':TstarK[ixs], 'RsRS':RsRS[ixs], \
             'RpValRE':RpValRE[ixs], 'RpLsigRE':RpLsigRE[ixs], 'RpUsigRE':RpUsigRE[ixs], \
             'MpValME':MpValME[ixs], 'TSM':TSM[ixs] }
    return outp


def readPredictedProperties( SMFlag='TSM', source='Kunimoto2022' ):
    if source=='Kunimoto2022':
        z = readPredictedProperties_Kunimoto2022( SMFlag=SMFlag )
    elif source=='Barclay2018':
        z = readPredictedProperties_Barclay2018( SMFlag=SMFlag )
    return z

def readPredictedProperties_Kunimoto2022( SMFlag='TSM' ):
    return z

def readPredictedProperties_Barclay2018( SMFlag='TSM' ):

    """
    Processes the predicted planet information from Barclay (uses version 2)

    Parameters:
    SMFlag: TSM or ESM (float)
    """

    idir = os.path.dirname( __file__ )
    ipath = os.path.join( idir, 'predictedProperties_v2.pkl' )
    if os.path.isfile( ipath )==False:
        processTargetLists.predictedTESS()
    ifile = open( ipath, 'rb' )
    z = pickle.load( ifile )
    ifile.close()
    RsRS = z['RsRS']
    TeqK = z['TeqK']
    Insol = z['Insol']
    TstarK = z['TstarK']
    if SMFlag == 'TSM':
        SM = z['TSM']
    elif SMFlag == 'ESM':
        SM = z['ESM']
    aRs = z['aRs']
    b = z['b']
    ideg = np.rad2deg( np.arccos( b/aRs ) )
    T14hr = z['T14hr']
    Pday = z['Pday']
    MsMS = z['MsMS']
    RpValRE = z['RpValRE']
    MpValME = z['MpValME']
    cad2min = z['cad2min']
    Vmag = z['Vmag']
    Jmag = z['Jmag']
    Kmag = z['Kmag']

    ixs = np.isfinite( TeqK )*np.isfinite( SM )*np.isfinite( RpValRE )
    
    outp = { 'SM': SM[ixs], 'cad2min': cad2min, 'TeqK':TeqK[ixs], \
             'aRs':aRs [ixs], 'Pday':Pday[ixs], 'Insol':Insol[ixs], \
             'MsValMS':MsMS[ixs], 'RsRS':RsRS[ixs], 'TstarK':TstarK[ixs], \
             'RpValRE':RpValRE[ixs], 'MpValME':MpValME[ixs], \
             'ideg':ideg[ixs], 'b':b[ixs], 'T14hr':T14hr[ixs], \
             'Vmag':Vmag[ixs], 'Jmag':Jmag[ixs], 'Kmag':Kmag[ixs] }
    return outp


def getFifthPredicted( SMFlag='TSM', RpMax=0, RpMin=0, TeqMax=0, TeqMin=0 ):
    
    """
    Finds the fifth-highest ESM or TSM value for the predicted planets in a 
    given RpRE and Teq range.

    Parameters:
    SMFlag- TSM or ESM (str)
    RpMax, RpMin, TeqMax, TeqMin- Define the grid cell in question (float)
    """

    z = readPredictedProperties( SMFlag=SMFlag )

    numPlanets = len(z['RsRS'])
    ixs = [i for i in range(numPlanets) if RpMin < z['RpValRE'][i] < RpMax]
    ixs1 = [i for i in ixs if TeqMin < z['TeqK'][i] < TeqMax]
        
    highestSMs = [0, 0, 0, 0, 0]
    
    for n in ixs1:
        if z['SM'][n] > highestSMs[0]:
            highestSMs.pop(0)
            highestSMs.append(z['SM'][n])
            highestSMs.sort()

    return  highestSMs[0]

def getFifthPredicted_OLD( SMFlag='TSM', RpMax=0, RpMin=0, TeqMax=0, TeqMin=0 ):
    
    """
    Finds the fifth-highest ESM or TSM value for the predicted planets in a 
    given RpRE and Teq range.

    Parameters:
    SMFlag- TSM or ESM (str)
    RpMax, RpMin, TeqMax, TeqMin- Define the grid cell in question (float)
    """

    z = readPredictedProperties( SMFlag=SMFlag )

    numPlanets = len(z['RsRS'])
    ixs = [i for i in range(numPlanets) if RpMin < z['RpValRE'][i] < RpMax]
    ixs1 = [i for i in ixs if TeqMin < z['TeqK'][i] < TeqMax]
        
    highestSMs = [0, 0, 0, 0, 0]
    
    for n in ixs1:
        if z['SM'][n] > highestSMs[0]:
            highestSMs.pop(0)
            highestSMs.append(z['SM'][n])
            highestSMs.sort()

    return  highestSMs[0]

def SMRepeats( SMFlag = 'ESM', survey = {} ):
    
    data = pickle.load(open('toiProperties.pkl','rb'))['allVals']
    tic = data['TICID']
    RpRE = data['RpValRE']
    plName = data['planetName']
    ESM = data['ESM']
    TSM = data['TSM']
    
    topRanked = gridTOIs( survey=survey, SMFlag=SMFlag,\
                          ASCII=True )
    for i, j in enumerate( topRanked ):
        ix = j.find(' ')
        topRanked[i] = j[:ix]
    if SMFlag == 'ESM':
        SM = ESM
    else:
        SM = TSM
        
    namesToTIC = {}
    for i,j in enumerate( plName ):
        namesToTIC[j, SM[i], RpRE[i]] = tic[i]
        
    TICtoNames = {}
    for key, value in namesToTIC.items():
           if value in TICtoNames:
               TICtoNames[value].append(key)
           else:
              TICtoNames[value]=[key]
              
    bestSMs = {}
    for key, value in TICtoNames.items():  
        bestSMs[key] = []
        if len( value ) > 1:
            for l in value:
                if l[2] > 0:
                    name = None
                    pdb.set_trace() # how to pass Teq into threhold function here?
                    if SMFlag == 'ESM':
                        if l[1] > surveySetup.thresholdESM(l[2])[0]:
                            name = l[0]
                    else:
                        if l[1] > surveySetup.thresholdTSM(l[2])[0]:
                            name = l[0]
                    if name in topRanked:
                        name = f'{l[0]}*'
                    if name != None:
                        bestSMs[key].append(name)
        if len( bestSMs[key] ) < 2:
            del( bestSMs[key] )
            
    return bestSMs
            
def readExoFOP( forceDownload=False ):
    """
    Reads in the TFOP priority and comments for each TOI.
    """
    exoFOPpath = downloadTargetLists.ExoFOP( forceDownload=forceDownload )
    ifile = open( exoFOPpath )
    reader = csv.reader( ifile, quotechar='"' )
    d = []
    for i in reader:
        d += [ i ]
    cols = np.array( d[0] )
    rows = d[1:]
    n = len( rows )
    y = {}
    for i in range( n ):
        TOI = np.array( rows[i] )[cols=='TOI'][0]
        Priority = np.array( rows[i] )[cols=='Master'][0]
        Comments = np.array( rows[i] )[cols=='Comments'][0]
        y[TOI] = [ Priority, Comments ]
    return y

def CreateASCII_TOIs( ipath='toiProperties.pkl', survey={}, SMFlag = 'TSM', onlyPCs=False, \
                      topFivePredicted=False, multTIC=False, forceDownloadExoFOP=False ):
    
    Tgrid, Rgrid = survey['gridEdges']( survey['surveyName'] )
    ifile = open( ipath, 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    props = ['planetName', 'TICID', 'RA', 'Dec', \
             'Vmag', 'Imag', 'Jmag', 'Hmag', 'Kmag',\
             SMFlag, 'Kamp', 'Pday', \
             'TstarK', 'loggstarCGS', 'RsRS', 'MsMS', \
             'MpValME', 'RpValRE', 'TeqK' ]
  
    topRanked, dateStr = gridTOIs( ipath=ipath, survey=survey, SMFlag=SMFlag, \
                                   onlyPCs=onlyPCs, ASCII=True )
    nAll = len( z['planetName'] )
    ixsAll = np.arange( nAll )
    nTop = len( topRanked )
    
    if multTIC:
        topRanked = []
        for i in list( SMRepeats(SMFlag=SMFlag, survey=survey ).values()):
            for j in i:
                topRanked.append(j)
        nTop = len(topRanked)
        topFivePredicted = False

    topRankedIxs = np.zeros( nTop, dtype=int )
    for i in range( nTop ):
        #print( topRanked[i] )
        if 1:
            ixName = topRanked[i].find( ')' )
            ix = int( ixsAll[z['planetName']==topRanked[i][:ixName+1]] )
            topRankedIxs[i] = ix
            if topRanked[i][-1]=='*':
                z['planetName'][ix] = '{0}*'.format( z['planetName'][ix] )
    if topFivePredicted:
        topToPrintIxs = []
        for i in range( nTop ):
            ix = topRankedIxs[i]
            if z['planetName'][ix][-1]=='*':
                topToPrintIxs += [ ix ]
    else:
        topToPrintIxs = topRankedIxs
    
    #print(topToPrintIxs)
    n = len( topToPrintIxs )
    
    # Dictionary of properties for top-ranked to be written to ASCII output:
    ASCII = {} 
    for p in props:
        ASCII[p] = z[p][topToPrintIxs]
    RA_deg = z['RA_deg'][topToPrintIxs]
    Dec_deg = z['Dec_deg'][topToPrintIxs]

    # Sort by declination coordinate:
    #ixs = np.argsort( ASCII['Dec_deg'] )
    ixs = np.argsort( Dec_deg )
    for p in props:
        ASCII[p] = np.array( ASCII[p] )[ixs]

    # Add exofop data to file:
    exoFOP = readExoFOP( forceDownload=forceDownloadExoFOP )
    priority = []
    comments = []
    
    TOI = list(ASCII['planetName'])
    
    for i,j in enumerate(TOI):
        k = j.split('-')[1]
        TOI[i] = k.split('(')[0]
    for i in TOI:
        if i in exoFOP:
            priority.append(exoFOP[i][0])
            comments.append(exoFOP[i][1])
        else:
            priority.append(None)
            comments.append(None)
    ASCII['Priority'] = np.array(priority)
    ASCII['Comments'] = np.array(comments)
    for j in ['Priority', 'Comments']:
        props.append(j)
    
    # Correct missing Imags (probably most of them):
    if pysynphotImport==True:
        ixs = np.arange( n )[np.isnan( ASCII['Imag'] )]
        m = len( ixs )
        print( '\nEstimating {0:.0f} Imags...'.format( m ) )
        for i in range( m ):
            Jmag = ASCII['Jmag'][ixs[i]]
            TstarK = ASCII['TstarK'][ixs[i]]
            loggCGS = ASCII['loggstarCGS'][ixs[i]]
            if np.isfinite( Jmag )*( TstarK<31000 )*np.isfinite( loggCGS ):
                Imag = Utils.convertMag( Jmag, TstarK, loggCGS, \
                                         inputMag='J', outputMag='I' )
                ASCII['Imag'][ixs[i]] = Imag
    col0 = 'Target,'.rjust( 19 ) 
    col1 = 'TICID,'.rjust( 17 )
    col2 = 'RA,'.center( 17 )
    col3 = 'Dec,'.center( 15 )
    col4a = 'Vmag,'.rjust( 8 )
    col4b = 'Imag,'.rjust( 8 )
    col4c = 'Jmag,'.rjust( 8 )
    col4d = 'Hmag,'.rjust( 8 )
    col4e = 'Kmag,'.rjust( 8 )
    col5 = '{0},'.format( SMFlag ).rjust( 11 )
    col6 = 'K(m/s),'.rjust( 9 )
    col7 = 'P(d),'.rjust( 11 )
    col8 = 'Teff(K),'.rjust( 11 )
    col9 = 'logg(CGS),'.rjust( 11 )
    col10 = 'Rs(RS),'.rjust( 11 )
    col11 = 'Ms(MS),'.rjust( 11 )
    col12 = 'Mp(ME),'.rjust( 11 )
    col13 = 'Rp(RE),'.rjust( 11 )
    col14 = 'Teq(K),'.rjust( 11 )
    col15 = 'Priority,'.center( 13 )
    col16 = 'Comments'.ljust( 50 )
    hdr = '# TOIs accessed on date (YYYY-MM-DD): {0}\n# '.format( dateStr )
    hdr += '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}{15}{16}{17}{18}{19}{20}'\
           .format( col0, col1, col2, col3, \
                    col4a, col4b, col4c, col4d, col4e, \
                    col5, col6, col7, col8, col9, col10, \
                    col11, col12, col13, col14, col15, col16 )
    ncol = [ 18, 15, 16, 15, 7, 7, 7, 7, 7, 10, 8, \
             10, 10, 10, 10, 10, 10, 10, 10, 12, 50 ] # column width
    ndps = [  0,  0, 2,  2, 1, 1, 1, 1, 1, 1, 1,  \
              3,  0,  1, 1, 1,  1,  1, 0, 0, 0 ] # decimal places
    hdr += '\n#{0}'.format( 323*'-' )
    m = len( props )
    def rowStr( i, zin ):
        rstr = '\n  '
        for j in range( m ): # loop over each property
            k = props[j]
            if ( k!='planetName' )*( k!='TICID' )*( k!='RA' )*( k!='Dec' )*(k!='Comments')*(k!='Priority'):
                # numbers
                rstr += '{0:.{1}f},'.format( zin[k][i], ndps[j] ).rjust( ncol[j] )
            elif (k=='Priority'):
                rstr += '{0},'.format( zin[k][i] ).center( ncol[j] )
            elif (k=='Comments'):
                # numbers
                rstr += '{0}'.format( zin[k][i] ).ljust( ncol[j] )
            else: # strings
                rstr += '{0},'.format( zin[k][i] ).rjust( ncol[j] )
        return rstr
    ostr = '{0}'.format( hdr )
    for i in range( n ): # loop over each TOI
        rstr = rowStr( i, ASCII )
        ostr += rstr

    # Write to file:
    oname = 'RVvaluesBy{0}_{1}.txt'.format( SMFlag, survey['obsSample'] )
    if multTIC == True:
        oname = oname.replace('.txt', '_Multis.txt')
    else:
        if onlyPCs == True:
            oname = oname.replace( '.txt', '_onlyPCs.txt' )
        if topFivePredicted==True:
            oname = oname.replace( '.txt', '_topPredicted.txt' )
    odir = os.path.join( os.getcwd(), 'ASCII' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opath1 = os.path.join( odir, oname )

    ofile = open( opath1, 'w' )
    ofile.write( ostr )
    ofile.close()

    # Now write out the same file but ranked by SM:
    ixs = np.arange( n )[np.argsort( ASCII[SMFlag] )[::-1]]
    for k in props:
        ASCII[k] = ASCII[k][ixs]
    ostr2 = '{0}'.format( hdr )
    for i in range( n ): # loop over each TOI
        rstr = rowStr( i, ASCII )
        ostr2 += rstr
    opath2 = opath1.replace( '.txt', '_ranked{0}.txt'.format( SMFlag ) )
    ofile = open( opath2, 'w' )
    ofile.write( ostr2 )
    ofile.close()

    print( '\nSaved:\n{0}\n{1}'.format( opath1, opath2 ) )
    return opath1, opath2

def CreateASCII_Confirmed( ipath='confirmedProperties.pkl', survey={}, SMFlag = 'TSM' ):
    """
    Would be better to merge this in single routine for TOIs and Confirmed.
    """
    Tgrid, Rgrid = survey['gridEdges']( survey['surveyName'] )
    ifile = open( ipath, 'rb' )
    z0 = pickle.load( ifile )
    ifile.close()
    z = z0['allVals']
    # TODO = Need to add RA, Dec, loggstarCGS to confirmedProperties.pkl
    props = [ 'planetName', 'RA', 'Dec', \
              'Vmag', 'Jmag', 'Hmag', 'Kmag',\
              SMFlag, 'Kamp', 'Pday', 'TstarK', 'RsRS', 'MsMS', \
              'MpValME', 'MpLowErrME', 'MpUppErrME', 'RpValRE', 'TeqK' ]
  
    topRanked, dateStr = gridConfirmed( ipath=ipath, survey=survey, SMFlag=SMFlag, ASCII=True )
    nAll = len( z['planetName'] )
    ixsAll = np.arange( nAll )
    nTop = len( topRanked )
    
    for i in range( nAll ):
        z['planetName'][i] = z['planetName'][i].replace( ' ', '' )

    # Identify the indices of the top-ranked targets in each cell and add asterisks to
    # planet names in the array if they have been flagged as such by the addTopSMs() 
    # routine via the gridConfirmed() call above:
    topRankedIxs = np.zeros( nTop, dtype=int )
    for i in range( nTop ):
        if topRanked[i][-1]=='*':
            ix = int( ixsAll[z['planetName']==topRanked[i][:-1]] )
            z['planetName'][ix] = '{0}*'.format( z['planetName'][ix] )
        else:
            ix = int( ixsAll[z['planetName']==topRanked[i]] )
        topRankedIxs[i] = ix
    topToPrintIxs = topRankedIxs
    n = len( topToPrintIxs )
    
    # Dictionary of properties for top-ranked to be written to ASCII output:
    ASCII = {} 
    for p in props:
        ASCII[p] = z[p][topToPrintIxs]
    RA_deg = z['RA_deg'][topToPrintIxs]
    Dec_deg = z['Dec_deg'][topToPrintIxs]
    
    # TO ADD BACK IN ONCE DEC_DEG ADDED BACK IN...
    # Sort by declination coordinate:
    #ixs = np.argsort( ASCII['Dec_deg'] )
    ixs = np.argsort( Dec_deg )
    for p in props:
        ASCII[p] = np.array( ASCII[p] )[ixs]
    RA_deg = RA_deg[ixs]
    Dec_deg = Dec_deg[ixs]
    
    pl = list(ASCII['planetName'])
    npl = len( pl )    
    for i,j in enumerate(pl):
        try:
            k = j.split('-')[1]
        except:
            continue
    col0 = 'Target,'.rjust( 19 ) 
    col1 = 'RA,'.center( 17 )
    col2 = 'Dec,'.center( 15 )
    col3a = 'Vmag,'.rjust( 8 )
    col3b = 'Jmag,'.rjust( 8 )
    col3c = 'Hmag,'.rjust( 8 )
    col3d = 'Kmag,'.rjust( 8 )
    col4 = '{0},'.format( SMFlag ).rjust( 11 )
    col5 = 'K(m/s),'.rjust( 9 )
    col6 = 'P(d),'.rjust( 11 )
    col7 = 'Teff(K),'.rjust( 11 )
    #col7 = 'logg(CGS)'.rjust( 10 )
    col8 = 'Rs(RS),'.rjust( 11 )
    col9 = 'Ms(MS),'.rjust( 11 )
    col10a = 'MpVal(ME),'.rjust( 13 )
    col10b = 'MpSigL(ME),'.rjust( 13 )
    col10c = 'MpSigU(ME),'.rjust( 13 )
    col11 = 'Rp(RE),'.rjust( 11 )
    col12 = 'Teq(K)'.rjust( 11 )
    hdr = '# Exoplanet Archive accessed on date (YYYY-MM-DD): {0}\n# '.format( dateStr )
    hdr += '{0} {1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}{15}{16}{17}'\
           .format( col0, col1, col2, \
                     col3a, col3b, col3c, col3d, col4, \
                     col5, col6, col7, col8, col9, \
                     col10a, col10b, col10c, \
                     col11, col12 )
    ncol = [ 18, 16, 15, 7, 7, 7, 7, 10, 8, \
             10, 10, 10, 10, 12, 12, 12, 10, 10 ] # column width
    ndps = [  0,  0, 0, 1, 1, 1, 1, 1, 1, \
              3,  0,  1, 1,  3, 3, 3, 1, 0 ] # decimal places
    hdr += '\n#{0}'.format( 210*'-' )
    ostr = '{0}'.format( hdr )    
    m = len( props )
    def rowStr( i, zin ):
        rstr = '\n  '
        for j in range( m ): # loop over each property
            k = props[j]
            if ( k!='planetName' )*( k!='RA' )*( k!='Dec' ):
                # numbers
                rstr += '{0:.{1}f}'.format( zin[k][i], ndps[j] ).rjust( ncol[j] )
            else: # strings
                rstr += '{0}'.format( zin[k][i] ).rjust( ncol[j] )
            if j<m-1:
                rstr = '{0},'.format( rstr )
        return rstr

    for i in range( npl ): # loop over each planet
        rstr = rowStr( i, ASCII )
        ostr += rstr

    # Write to file:
    oname = 'RVvaluesBy{0}_confirmedPlanets_{1}.txt'.format( SMFlag, survey['obsSample'] )
    #if multTIC == True:
    #    oname = oname.replace('.txt', '_Multis.txt')
    #else:
    #    if onlyPCs == True:
    #        oname = oname.replace( '.txt', '_onlyPCs.txt' )
    #    if topFivePredicted==True:
    #        oname = oname.replace( '.txt', '_topPredicted.txt' )
    odir = os.path.join( os.getcwd(), 'ASCII' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opath1 = os.path.join( odir, oname )

    # Now write out the same file but ranked by SM:
    ixs = np.arange( n )[np.argsort( ASCII[SMFlag] )[::-1]]
    for k in props:
        ASCII[k] = ASCII[k][ixs]
    ostr2 = '{0}'.format( hdr )
    for i in range( n ): # loop over each TOI
        rstr = rowStr( i, ASCII )
        ostr2 += rstr
    opath2 = opath1.replace( '.txt', '_ranked{0}.txt'.format( SMFlag ) )
    ofile = open( opath2, 'w' )
    ofile.write( ostr2 )
    ofile.close()

    print( '\nSaved:\n{0}\n{1}'.format( opath1, opath2 ) )
    return opath1, opath2

    
    ofile = open( opath, 'w' )
    ofile.write( ostr )
    ofile.close()
    print( '\nSaved:\n{0}'.format( opath ) )

    return opath1, opath2

def TeqK_ExoFOPvsKempton (Kempton, ExoFOP):
    """
    Creates a plot that compares the values of TeqK computed as in Kempton et al. and those pulled
    from the Exoplanet Archive
    """
    fig = plt.figure()
    plt.plot(Kempton, ExoFOP, 'b.', Kempton, Kempton, 'r-')
    plt.xlabel('TeqK Kempton')
    plt.ylabel('TeqK ExoFOP')
    plt.title('Temperatures computed as in Kempton et al. (2018) \n vs those pulled from the Exoplanet Archive')

    oname = "TeqK_Kempton_vs_ExoFOP.pdf"
    odir = FIGDIR
    opathk = os.path.join( odir, oname )
    fig.savefig( opathk )
    print('\n Saved: ', oname)

def addHeatMap (ax, xLines, yLines, TeqK, RpRE, Tgrid, Rgrid, cmap):
    """
    Adds a Heat Map to a figure based on fraction of planets/TOIs found in box compared to 
    those predicted by Barclay et al.
    """
    boxes = []
    box_values = []
    boxn = 0
    zPredicted = readPredictedProperties()
    predTeqK = zPredicted['TeqK']
    predRpVal = zPredicted['RpValRE'] 

    # Generates the box coordinates and numbers, using ticks from 0.5 to 6.5
    for x in xLines[:-1]:
        for y in yLines[:-1]:
            box_value = Utils.HeatMapValues([Tgrid[int(x-0.5)], Tgrid[int(x+0.5)]], \
                [Rgrid[int(y-0.5)], Rgrid[int(y+0.5)]], \
                TeqK, RpRE, predTeqK, predRpVal) # Calculates fraction of TOIs:fraction of pred in box
            boxes.append([[x, x, x+1, x+1], [y, y+1, y+1, y], boxn]) # Box coordinates and number
            box_values.append(box_value)
            boxn +=1             

    box_values = np.array(box_values)

    box_values2 = []
    for value in box_values:
        if value != 0:
            box_values2.append(np.log2(value))
        else:
            box_values2.append(np.nan)

    #Normalizes the boxes. TOI < pred and TOI > pred are normalized separately into [0, 0.5] and [0.5, 1]
    box_norm = Utils.Normalize(box_values2, True)
    
    # Colors in the boxes
    for box in boxes:
        box_color = cmap(box_norm[box[2]])
        ax.fill(box[0], box[1], color=box_color, zorder = 0)
    
    return ax, np.log2(np.max(box_values))

def addColorBar(ax, val, cmap):
    tick_fs = 12
    ax.tick_params( labelsize=tick_fs ) # Changes the font size of tick marks
    cb_norm = matplotlib.colors.Normalize( vmin=-val, vmax=val )
    # Only right half of colorbar properly normalized. Unsure how to have both consistent with figure
    # normalization without changing so that all data normalized together
    cb = matplotlib.colorbar.ColorbarBase( ax, cmap=cmap, norm=cb_norm, \
                                        orientation='horizontal' ) # Creates the color bar in cb_axis
    cb.solids.set_rasterized( True )
    cb.solids.set_edgecolor( 'face' )
