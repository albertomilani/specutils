import sys
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import re
import glob

class Spettro1D:
    
    def __init__(self):
        # list for wavelenght and flux values
        self.wl        = []
        self.flux      = []
        
        # ancillary informations
        self.aperture  = None
        self.beam      = None
        self.flux_unit = None
        self.wl_unit   = None
        
    def readFits(self, filename):
        with fits.open(filename) as file:
            cd1_1  = file[0].header['CD1_1']
            crval1 = file[0].header['CRVAL1']
            nw     = file[0].header['NAXIS1']

            # get axisx units
            self.flux_unit = file[0].header['BUNIT']
            matches = re.findall(r"wtype=\w+ label=\w+ units=(\w+)", file[0].header['WAT1_001'])
            self.wl_unit = matches[0]
            
            self.flux = file[0].data[0:nw]
            self.wl   = [crval1 + cd1_1*i for i in range(nw)]
            
    def fillFromData(self, wl, flux):
        self.wl   = wl
        self.flux = flux
        
    def fillFlux(self, flux):
        self.flux = flux
    
    #####################
    #    GET METHODS    #
    #####################
    
    def getCentralWl(self):
        return ((max(self.wl) + min(self.wl))/2.0)
    
    def getData(self):
        return (self.wl, self.flux)
    
    def getFluxArray(self):
        return self.flux
    
    def getFluxUnit(self):
        return self.flux_unit
    
    def getWlArray(self):
        return self.wl
    
    def getWlUnit(self):
        return self.wl_unit
    
    #####################
    #    SET METHODS    #
    #####################
    
    def setAperture(self, aperture):
        self.aperture = aperture
        
    def setBeam(self, beam):
        self.beam = beam
    
    def setFluxUnit(self, unit):
        self.flux_unit = unit
    
    def setWlUnit(self, unit):
        self.wl_unit = unit
        
    ########################
    #    USEFUL METHODS    #
    ########################
    
    def linearInterpolation(self, x):
        ia = min(range(len(self.wl)), key=lambda i: abs(self.wl[i]-x))
        if ia == (len(self.wl) - 1):
            raise Exception('x value out of wavelenght range')
        ib = ia + 1
        xa = self.wl[ia]
        xb = self.wl[ib]
        ya = self.flux[ia]
        yb = self.flux[ib]
        y = ya + (x - xa) / (xb - xa) * (yb - ya)
        return y

    def dopplerCorrection(self, vel):
        # vel in km/s
        light_speed = 299792.458 # km/s
        self.wl = [x*(1 + vel/light_speed) for x in self.wl]
        
    def squareDiff(self, compare_sp):
        a_flux = [compare_sp.linearInterpolation(x) for x in self.wl]
        b_flux = self.getFluxArray()
        square_diff = [(a-b)**2 for a, b in zip(a_flux, b_flux)]
        return sum(square_diff)
    
    def continuumCorrection(self, order=3, hi_rej=1, lo_rej=1, iterations=10, output=None, outputfile=None):
        x = self.wl
        y = self.flux
        x_rej = []
        y_rej = []
        for i in range(iterations):
            fit = np.polynomial.legendre.Legendre.fit(x, y, order)
            residuals = np.asarray([y[i] - fit(x[i]) for i in range(len(x))])
            sigma = residuals.std()
            new_x = [x[j] for j in range(len(x)) if residuals[j] < sigma*hi_rej and residuals[j] > (-sigma*lo_rej)]
            new_y = [y[j] for j in range(len(x)) if residuals[j] < sigma*hi_rej and residuals[j] > (-sigma*lo_rej)]
            x_rej = x_rej + [x[j] for j in range(len(x)) if residuals[j] > sigma*hi_rej or residuals[j] < (-sigma*lo_rej)]
            y_rej = y_rej + [y[j] for j in range(len(x)) if residuals[j] > sigma*hi_rej or residuals[j] < (-sigma*lo_rej)]
            x = new_x
            y = new_y

        self.cc_y = [self.flux[j]/fit(self.wl[j]) for j in range(len(self.wl))]
    
        if output is not None:
            plt.clf()
            plt.close()
            fig = plt.figure(figsize=(10, 6), dpi=100)
            plt.plot(self.wl, self.flux, linewidth=0.5)
            plt.scatter(x, y, marker='o', c='none', edgecolors='b')
            plt.plot(self.wl, [fit(x) for x in self.wl], linewidth=0.5)
            plt.scatter(x_rej, y_rej, marker='x')
            plt.show()
            #plt.savefig(output)

        if outputfile is not None:
            outstr = ''
            for i in range(len(self.wl)):
                outstr = outstr + str(self.wl[i]) + ' ' + str(self.cc_y[i]) + '\n'
            with open(outputfile, "w+") as f:
                f.write(outstr)
            
        return self.cc_y
        
class SpettroEchelle:
    
    def __init__(self):
        self.onedspecs = {}
        self.flux_unit = 'ADU' # ADU as default
        self.wl_unit   = None
    
    def readFits(self, filename):
        with fits.open(filename) as file:
        
            # take wavelenght information from header
            wl_infos = ''
            for k in file[0].header:
                #print(k, '=', file[0].header[k])
                if re.search(r"WAT2_\d+", k):
                    wl_infos += file[0].header[k].ljust(68, ' ')
            
            # get wl unit
            matches = re.findall(r"wtype=\w+ label=\w+ units=(\w+)", file[0].header['WAT1_001'])
            self.wl_unit = matches[0]
            
            # parse wl info
            matches = re.findall(r"spec\d+\s+=\s+\"([\d\s\.]+)\"", wl_infos)
            i = 0
            for match in matches:
                data = re.split(r"\s+", match)
                # http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs
                ap   = int(data[0])
                beam = int(data[1])
                w1   = float(data[3])
                dw   = float(data[4])
                nw   = int(data[5])

                self.onedspecs[ap] = Spettro1D()
                self.onedspecs[ap].fillFromData([w1 + dw*i for i in range(nw)], file[0].data[i][0:nw])
                self.onedspecs[ap].setAperture(ap)
                self.onedspecs[ap].setBeam(beam)

                i += 1

    def getApertureData(self, aperture):
        return self.onedspecs[aperture].getData()
    
    def getApertureAs1DSpec(self, aperture):
        spec = self.onedspecs[aperture]
        spec.setWlUnit(self.wl_unit)
        spec.setFluxUnit(self.flux_unit)
        return spec
    
    def getBestApertureByLambdaAs1DSpec(self, wl_0):
        mins = {}
        for ap in self.onedspecs:
            mins[ap] = abs(self.onedspecs[ap].getCentralWl() - wl_0)
        return self.getApertureAs1DSpec(min(mins, key=mins.get))
