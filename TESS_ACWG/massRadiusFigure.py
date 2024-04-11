import pdb, sys, os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from TESS_ACWG.Code import Utils
from . import Utils

FIGDIR = os.path.join( os.getcwd(), 'Figures' )

def Confirmed( ipath='confirmedProperties.pkl', weighting='None', alpha='default' ):
    ms = 10
    ifile = open( ipath, 'rb' )
    z = pickle.load( ifile )
    ifile.close()
    pl = z['allVals']['planetName']
    TeqK = z['allVals']['TeqK']
    MpME = z['allVals']['MpValME']
    RpRE = z['allVals']['RpValRE']
    b = z['allVals']['b']
    RpRs = z['allVals']['RpRs']
    if weighting=='TSM':
        SM = z['allVals']['TSM']
    elif weighting=='ESM':
        SM = z['allVals']['ESM']
    else:
        SM = None
    ixs = ( np.isfinite( TeqK ) )*( np.isfinite( MpME ) )*( np.isfinite( RpRE ) )
    pl = pl[ixs]
    TeqK = TeqK[ixs]
    MpME = MpME[ixs]
    RpRE = RpRE[ixs]
    MpLSigME = z['allVals']['MpLowErrME'][ixs]
    b = b[ixs]
    RpRs = RpRs[ixs]
    if weighting=='TSM':
        SM = SM[ixs]
    elif weighting=='ESM':
        SM = SM[ixs]
    if 1:#weighting!='ESM':
        # Unless considering ESM, remove those
        # without 3-sigma published masses:
        nsigMpME = 3.0
        ixs = ( MpME/MpLSigME>nsigMpME )
        pl = pl[ixs]
        TeqK = TeqK[ixs]
        MpME = MpME[ixs]
        RpRE = RpRE[ixs]
        b = b[ixs]
        RpRs = RpRs[ixs]
        if weighting=='TSM':
            SM = SM[ixs]
        elif weighting=='ESM':
            SM = SM[ixs]
        titleStr = 'Planets with published masses ($>{0:.0f}\\sigma$)'.format( nsigMpME )
    else:
        titleStr = ''
    Tupp = 3000
    fig, ax, cbAx, cb, cmap = createAxis( TeqK, Tupp=Tupp )
    plotData( ax, cmap, pl, MpME, RpRE, TeqK, SM, b, RpRs, ms=ms, Tupp=Tupp, weighting=weighting, alpha=alpha )
    addIsoDensityContours( ax )
    ax.text( 0.3, 23, titleStr, weight='normal', fontsize=18, \
             horizontalalignment='left', verticalalignment='bottom' )
    oname1 = 'confirmedPlanetsMassRadius_noEmpiricalRelation.pdf'
    oname2 = 'confirmedPlanetsMassRadius.pdf'
    if weighting!='None':
        oname1 = oname1.replace( '.pdf', '_{0}.pdf'.format( weighting ) )
        oname2 = oname2.replace( '.pdf', '_{0}.pdf'.format( weighting ) )
    opath1a = os.path.join( FIGDIR, oname1 )
    opath1b = opath1a.replace( '.pdf', '.png' )
    fig.savefig( opath1a )
    fig.savefig( opath1b )
    
    plotEmpiricalRelation( ax )
    opath2a = os.path.join( FIGDIR, oname2 )
    opath2b = opath2a.replace( '.pdf', '.png' )
    fig.savefig( opath2a )
    fig.savefig( opath2b )
    print( '\nSaved:\n{0}\n{1}\n{2}\n{3}\n'.format( opath1a, opath2a, opath1b, opath2b ) )
    opaths = [ opath1a, opath1b, opath2a, opath2b ]
    return opaths


def plotEmpiricalRelation( ax ):
    lw = 4
    RpRE = np.linspace( 10**-0.8, 10**1.5, 1000 )
    MpME = Utils.planetMassFromRadius( RpRE, whichRelation='Chen&Kipping2017' )
    ax.plot( MpME, RpRE, '-', c='Black', lw=lw, zorder=5 )
    ax.plot( [0.3,0.5], [19,19], '-', c='Black', lw=lw )
    label = 'empirical mass-radius relation'
    ax.text( 0.6, 19, label, fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    return None

def plotData( ax, cmap, pl, MpME, RpRE, TeqK, SM, b, RpRs, ms=10, Tupp=3000, weighting='None', alpha='default' ):
    cDat = TeqK*np.ones_like( TeqK )
    ixs = cDat>Tupp
    cDat[ixs] = Tupp
    Tmin = int( 100*np.floor( cDat.min()/100. ) )
    Tmax = int( 100*np.ceil( cDat.max()/100. ) )
    Trange = Tmax-Tmin
    if weighting!='None':
        alphaMax = 1
        ixs = np.isfinite( SM )
        pl = pl[ixs]
        MpME = MpME[ixs]
        RpRE = RpRE[ixs]
        cDat = cDat[ixs]
        SM = SM[ixs]
        b = b[ixs]
        RpRs = RpRs[ixs]
        SMmin = 97 # to include HAT-P-7b as the lower cutoff
        ixs = ( SM>SMmin )*( b+RpRs<1 )
        pl = pl[ixs]
        MpME = MpME[ixs]
        RpRE = RpRE[ixs]
        cDat = cDat[ixs]
        SM = SM[ixs]
        # To avoid outlier TSM/ESM values skewing the colourscale,
        # set TSM/ESM=250 as the maximum for the purpose of defining
        # the colourscale:
        excellentSM = 250
        #SM[SM>excellentSM] = excellentSM
        alpha0 = alphaMax*np.ones_like(SM)#*SM/SM.max()
    else:
        alpha0 = 0.7*np.ones_like( MpME )
    if alpha=='default':
        alpha = alpha0
    else:
        alpha = alpha*np.ones_like( cDat )
    n = len( MpME )
    for i in range( n ):
        c = cmap( ( cDat[i]-Tmin )/Trange )
        ax.plot( [MpME[i]], [RpRE[i]], 'o', mfc=c, mec=c, ms=ms, alpha=alpha[i], zorder=10 )
        #if weighting=='ESM': # TEMPORARY
        #    ax.text( MpME[i], RpRE[i], pl[i] )
    return None

def createAxis( TeqK, Tupp=3000 ):
    label_fs = 18
    tick_fs = 16
    figw = 14
    figh = 8.8
    xlow = 0.065
    ylow = 0.1
    axw = 0.80
    axh = 0.85
    TeqKmin = int( 100*np.floor( TeqK.min()/100. ) )
    TeqKmax = min( [ Tupp, int( 100*np.ceil( TeqK.max()/100. ) ) ] )
    TeqKrange = TeqKmax-TeqKmin
    #cmap = matplotlib.cm.Paired
    cmap = matplotlib.cm.get_cmap( 'jet', 8 )
    lgrey = 0.9*np.ones( 3 )
    fig = plt.figure( figsize=[ figw, figh ] )
    ax = fig.add_axes( [ xlow, ylow, axw, axh ] )
    ax.set_xscale( 'log' )
    ax.set_xlim( [ 0.2, 3500 ] )
    ax.set_ylim( [ 0.0, 25 ] )
    ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter( customLogFormat ) )
    ax.text( 0.5, -0.075, 'Mass (Earth masses)', fontsize=label_fs, \
             rotation=0, horizontalalignment='center', verticalalignment='top', \
             transform=ax.transAxes )
    ax.text( -0.055, 0.5, 'Radius (Earth radii)', fontsize=label_fs, \
             rotation=90, horizontalalignment='right', verticalalignment='center', \
             transform=ax.transAxes )
    ax.yaxis.set_ticks( np.arange( 0, 25, 1 ), minor=True )
    ax.yaxis.set_ticks( np.arange( 0, 30, 5 ), minor=False )
    ax.tick_params( labelsize=tick_fs, length=14, which='major' )
    ax.tick_params( length=7, which='minor' )
    
    cbAx = fig.add_axes( [ xlow+axw+0.025, ylow+0.02, 0.03, axh-0.04 ] )
    cbAx.text( 3.5, 0.5, 'Planet Temperature (Kelvin)', fontsize=label_fs, \
               horizontalalignment='right', verticalalignment='center', \
               rotation=270, transform=cbAx.transAxes )
    cbAx.tick_params( labelsize=tick_fs )
    cbNorm = matplotlib.colors.Normalize( vmin=TeqKmin, vmax=TeqKmax )
    cb = matplotlib.colorbar.ColorbarBase( cbAx, cmap=cmap, norm=cbNorm, \
                                           orientation='vertical' )
    cb.solids.set_rasterized( True )
    cb.solids.set_edgecolor( 'face' )
    return fig, ax, cbAx, cb, cmap


def customLogFormat( y, pos ):
    # Find the number of decimal places required
    decimalplaces = int(np.ceil(np.maximum(-np.log10(y),0))) # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

def addIsoDensityContours( ax ):
    # Add on the composition density contours:
    h2, h2o, mgsio3, fe = Utils.densityContours()
    chydrogen = 'plum'
    cwater = 'powderblue'
    crock = 'burlywood'
    ciron = 'pink'
    lw = 4
    ax.plot( h2[:,0], h2[:,1], '-', lw=lw, c=chydrogen, zorder=0, label='hydrogen' )
    ax.plot( h2o[:,0], h2o[:,1], '-', lw=lw, c=cwater, zorder=0, label='water' )
    ax.plot( mgsio3[:,0], mgsio3[:,1], '-', lw=lw, c=crock, zorder=0, label='rock' )
    ax.plot( fe[:,0], fe[:,1], '-', lw=lw, c=ciron, zorder=0, label='iron' )

    ax.plot( [0.5,0.85], [15,15], '-', c=chydrogen, lw=lw )
    ax.plot( [0.5,0.85], [14,14], '-', c=cwater, lw=lw )
    ax.plot( [0.5,0.85], [13,13], '-', c=crock, lw=lw )
    ax.plot( [0.5,0.88], [12,12], '-', c=ciron, lw=lw )
    
    ax.text( 0.3, 16.1, 'Isodensity contours:', fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    ax.text( 1, 15, 'hydrogen', fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    ax.text( 1, 14, 'water', fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    ax.text( 1, 13, 'rock', fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    ax.text( 1, 12, 'iron', fontsize=18, \
             horizontalalignment='left', verticalalignment='center' )
    return None
