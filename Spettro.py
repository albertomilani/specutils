#!/usr/bin/env python3

import sys
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

class Spettro:
    def __init__(self):
        self.x = []
        self.y = []

    def loadFromFile(self, filename):
        with open(filename, "r") as f:
            for line in f:
                data = line.split()
                self.x.append(float(data[0]))
                self.y.append(float(data[1]))

    def continuumCorrection(self, order, hi_rej, lo_rej, iterations, output=None, outputfile=None):
        x = self.x
        y = self.y
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

        self.cc_y = [self.y[j]/fit(self.x[j]) for j in range(len(self.x))]
    
        if output is not None:
            plt.clf()
            plt.close()
            plt.plot(self.x, self.y, linewidth=0.5)
            plt.scatter(x, y, marker='o', c='none', edgecolors='b')
            plt.plot(self.x, [fit(x) for x in self.x], linewidth=0.5)
            plt.scatter(x_rej, y_rej, marker='x')
            plt.savefig(output)

        if outputfile is not None:
            outstr = ''
            for i in range(len(self.x)):
                outstr = outstr + str(self.x[i]) + ' ' + str(self.cc_y[i]) + '\n'
            with open(outputfile, "w+") as f:
                f.write(outstr)
            
        return self.cc_y


