#!/usr/bin/python
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#from savitzky_golay_lowpassfilter import savgol_irregular_brute
from savitzky_golay_lowpassfilter import savgol_irregular_interp

## Sounding file name
fname = "example_data.txt"

## Height of ridge for inverse Froude number
hridge = 225.

## Parameters for Savitzky-Golay low-pass filter
zwindow = 200.

## Limit to plot
zmax = 4E3
# zmax = None


## Funtions to read sounding
def read_sounding (fname):
    """
    The purpose of this funtion is to read radio-sounding data. The data
    is a table in the ASCII format structured by several columns, delimited
    by spaces. The program does not assumes a specific number of columns,
    instead it works by trying to read an header with the variables names.

    This function will not work unless the following variable names are found:
        Time, Alt_AGL, Press, Temp, DP, VTemp, RelHum, Mix_Rat, WSpeed
    """
    def isfloat (s):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    def allfloat (s):
        for ss in s:
            if not isfloat (ss):
                return False
        return True
    if not os.path.isfile (fname):
        raise RuntimeError ("file '%s' does not exists?" % fname)
    fid = open(fname, 'Ur')
    lines = fid.readlines()
    fid.close()
    ## get station name
    for i in xrange(len(lines)):
        l = lines[i].replace('\n','').split(':')
        l = [s.strip() for s in l]
        if 'Station Name' in l:
            station = ''.join(l).replace('Station Name','')
            break
    ## get launching date
    for i in xrange(len(lines)):
        l = lines[i].replace('\n','')
        if 'launched' in l.lower():
            launched = ' '.join(l.split()).replace(' :',':')
            break
    ## get header with variables
    header = None
    for i in xrange(len(lines)):
        l = lines[i].replace('\n','').lower().split()
        if 'time' and 'alt_agl' and 'press' and 'temp' and 'relhum' in l:
            ## found header
            # header = l
            header = lines[i].replace('\n','').split()
            break
    if None == header:
        print "failed reading file '%s'" % fname
        raise RuntimeError ("failed to find header with variables... aborting")
    else:
        print "header with variables found."
        print "number of variables: %s" % len(header)
        print "printing variable names:"
        print np.array(header)
    ## build output dict
    o = {}
    o['nvars'] = len(header)
    o['header'] = header
    o['var'] = [h.lower() for h in header]
    o['station'] = station
    o['launched'] = launched
    for v in o['var']:
        o[v] = []
    ## lower case strings inside list 'header' for later processing
    header = o['var']
    ## establish indices to see if value is a floating point
    nj = []
    nj.append(header.index('time'))
    nj.append(header.index('alt_agl'))
    nj.append(header.index('press'))
    nj.append(header.index('temp'))
    nj.append(header.index('dp'))
    nj.append(header.index('vtemp'))
    nj.append(header.index('relhum'))
    nj.append(header.index('mix_rat'))
    nj.append(header.index('wspeed'))
    ## start reading file
    for i in xrange(len(lines)):
        l = lines[i].replace('\n','').lower()
        ## remove 'am' and 'pm' if not there are problems
        l = l.replace('am','').replace('pm','')
        ## split
        l = l.split()
        ## see if values are floats
        if len(l) == o['nvars'] and allfloat([l[j] for j in nj]):
            ## list l contains the values
            for (v, n, ) in zip(o['var'], xrange(o['nvars'])):
                o[v].append( l[n] )
    ## convert to float
    vv = ['Time', 'GPM_AGL', 'Alt_AGL', 'Alt_MSL', 'North', 'East', 'Ascent',\
        'Press', 'Temp', 'RelHum', 'WSpeed', 'VTemp', 'DP', 'Dens',\
        'VP', 'Mix_Rat', 'SVP']
    for v in vv:
        if v.lower() in o['var']:
            print v
            o[v.lower()] = np.array(o[v.lower()], float)
    ## convert to integer
    vv = ['WDirn']
    for v in vv:
        if v.lower() in o['var']:
            o[v.lower()] = np.array(o[v.lower()], int)
    return o

def correct_units (o):
    """
    This function works with the output of funtion 'read_sounding'.

    It converts the units of several variables (columns), namely
        Press, VP, SVP    - from hPa to Pa
        Temp, DP, VTemp   - from Celsius to Kelvein degrees
        Mix_Rat           - from g/kg to kg/kg
    """
    vv = ['press', 'vp', 'svp']
    for v in vv:
        if v in o['var']:
            V = o['header'][o['var'].index(v)]
            print "converting %s from hPa to Pa" % V
            o[v] *= 1E2
    vv = ['temp', 'vtemp', 'dp']
    for v in vv:
        if v in o['var']:
            V = o['header'][o['var'].index(v)]
            print "converting %s from degrees Celsius to Kelvin" % V
            o[v] += 273.15
    vv = ['mix_rat']
    for v in vv:
        if v in o['var']:
            V = o['header'][o['var'].index(v)]
            print "converting %s from g_vapor/kg_dry to kg_vapor / kg_dry" % V
            o[v] *= 1E-3
    return o


## Read radio sounding data
data = read_sounding (fname)
data = correct_units (data)


## Thermodynamic parameters
R = 8.31451
Rd = R / 28.9644E-3
Rv = R / 18.0153E-3
Cpd = 7/2. * Rd
Rd_Cp = 2/7.
g = 9.80665


print "Checking some of the data..."

def calc_err (x, y):
    m = np.abs(.5 * (x + y))
    k = m > 0
    relerr = np.abs(x[k] - y[k]) / m[k] * 100
    return "max rel. error = %g%%" % np.max(relerr)

RH = data['vp'] / data['svp'] * 100
print "relative humidity \t>>>\t %s" % calc_err (RH, data['relhum'])

wv = Rd/Rv * data['vp'] / (data['press'] - data['vp'])
print "vapor mix. ratio \t>>>\t %s" % calc_err (wv, data['mix_rat'])

Tv = data['temp'] * (1. + wv * Rv/Rd) / (1. + wv)
print "virtual temperature \t>>>\t %s" % calc_err (Tv, data['vtemp'])


## Limiting maximum height
k = (data['alt_agl'] <= (zmax + zwindow))
k = slice(0, np.flatnonzero(k).max()+1)
for var in data['var']:
    data[var] = data[var][k]



## Recomputing some of the values
def fun_Psat (T, P=None):
    """
    Psat = fun_Psat (T, P)    Computes vapor saturation pressure.

    The functions are from Buck (JAM, 1981, 20:1527-1532). Input assumes
    the temperature is in Kelvins and Pressure in Pascal.
    The same procedure from ECMWF IFS model is employed, namely in
    interpolating betwen saturation pressures over liquid water and ice
    for temperatures between -23 < T < 0 ( degrees centigrades). See:
    "IFS DOCUMENTATION – Cy43r1, PART IV: PHYSICAL PROCESSES", Chapter 12,
    http://www.ecmwf.int/search/elibrary/part?title=part&secondary_title=43R1
    """
    T0 = 273.16  # triple point
    ## saturation over liquid water, Buck's ew1 function
    # Psatl = 611.21 * np.exp(17.502 * (T - T0) / (T - T0 + 240.97))
    ## saturation over liquid water, Buck's ew4 function
    Psatl = 611.21 * np.exp((18.729-(T-T0)/227.3)/(T-T0+257.87)*(T-T0))
    if None != P:
        ## enhancement factor, Buck's fw3 function 
        # fl = 1 + 7E-4 + P * 1E-2 * 3.46E-6
        ## enhancement factor, Buck's fw4 function
        fl = 1 + 7.2E-4 + P * 1E-2 * (3.2E-6 + 5.9E-10 * (T-T0)**2)
        Psatl *= fl
    ## saturation over ice, ECMWF function
    # Psati = 611.21 * np.exp(22.587 * (T - T0) / (T + 0.7))
    ## saturation over ice, Buck's ei2 function
    # Psati = 611.15 * np.exp(22.452 * (T - T0) / (T - T0 + 272.55))
    ## saturation over ice, Buck's ei3 function
    Psati = 611.15 * np.exp((23.036-(T-T0)/333.7)/(T-T0+279.82)*(T-T0))
    if None != P:
        ## enhancement factor, Buck's fi3 function
        # fi = 1 + 3E-4 + P * 1E-2 * 4.18E-6
        ## enhancement factor, Buck's fi4 function 
        fi = 1 + 2.2E-4 + P * 1E-2 * (3.83E-6 + 6.4E-10 * (T-T0)**2)
        Psati *= fi
    ## between -23 < T - T0 < 0 interpolate between Psatl and Psati
    a = ((T - T0 + 23.) / 23.)
    a = np.maximum(0, np.minimum(1, a))**2
    Psat = a * Psatl + (1 - a) * Psati
    return Psat

z = data['alt_agl']
P = data['press']
T = data['temp']
Tdew = data['dp']
U = data['wspeed']

print "\nRecomputing some values assuming the main measurements are T and Tdew"

## Pvapor == Psat (Tdew) with corrections due to Pressure
Pv = fun_Psat (Tdew, P)
print "vapor pressure \t>>>\t %s" % calc_err (Pv, data['vp'])

## Pvsat == Psat (T) with corrections due to Pressure
Pvs = fun_Psat (T, P)
print "sat. vapor pressure \t>>>\t %s" % calc_err (Pvs, data['svp'])

## relative humidity (just to compare against the value in soundings)
RH = Pv / Pvs * 100
print "relative humidity \t>>>\t %s" % calc_err (RH, data['relhum'])

## vapor mixing ratio
wv = Pv / (P - Pv) * Rd / Rv
print "vapor mix. ratio \t>>>\t %s" % calc_err (wv, data['mix_rat'])

## specific humidity
# qv = wv / (1. + wv)

## virtual temperature
Tv = T * (1. + wv * Rv/Rd) / (1. + wv)
print "virtual temperature \t>>>\t %s" % calc_err (Tv, data['vtemp'])

## use radio-sounding Tv instead
Tv = data['vtemp']

## virtual potential temperature
Exner = (P / 1E5)**Rd_Cp
TPd = T / Exner
TPv = Tv / Exner

## compute Brunt-Vaisala frequency (squared)
#lnTPv_f, dlnTPvdz = savgol_irregular_brute (np.log(TPv), z, zwindow, deriv=1)
lnTPv_f, dlnTPvdz = \
    savgol_irregular_interp (np.log(TPv), z, zwindow, deriv=1)
TPv_f = np.exp(lnTPv_f)
N2 = g * dlnTPvdz


## compute 2nd derivative for vertical profile of wind speed
#U_f, dUdz, d2Udz2 = savgol_irregular_brute (U, z, zwindow, deriv=2)
U_f, dUdz, d2Udz2 = savgol_irregular_interp (U, z, zwindow, deriv=2)

## compute scorer parameter (squared)
lscorer2 = N2 / U_f**2 - d2Udz2 / U_f

## compute inverse Froude number (squared)
nhu2 = N2 * hridge**2 / U_f**2 

## compute Froude number (squared)
Fr2 = U_f**2 / N2 / hridge**2

## compute Richardson gradient and bulk numbers
Rig = N2 / dUdz**2
Rib = g / TPv_f * (TPv_f - TPv_f[0]) * (z - z[0]) / U_f**2


## Plotting
mpl.rcParams['figure.figsize'] = (3*3, 4*2)
mpl.rcParams['figure.dpi'] = 96
mpl.rcParams['savefig.dpi'] = 300
## PS/PDF Output Type 3 (Type3) or Type 42 (TrueType)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['axes.titlesize'] ='medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['xtick.labelsize'] = 'x-small'
mpl.rcParams['ytick.labelsize'] = 'x-small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.pad'] = 6
mpl.rcParams['xtick.minor.pad'] = 4
mpl.rcParams['ytick.major.pad'] = 6
mpl.rcParams['ytick.minor.pad'] = 4
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

fig, ax = plt.subplots(2, 3, sharey=True)
for axes in np.array(ax).flatten():
    #axes.set_yscale('log')
    axes.grid(axis='x', which='major', c='k', ls=':')
    axes.grid(axis='y', which='major', c='k', ls=':')


cdark = ['forestgreen', 'darkgoldenrod', 'firebrick', 'navy']
csoft = ['yellowgreen', 'goldenrod', 'tomato', 'slateblue']

if 'zmax' in dir() and zmax != None:
    zmax = np.minimum(zmax, np.max(z))
else:
    zmax = np.max(z)
    zmax = zmax + 10**int(np.log10(zmax)) - zmax % 10**int(np.log10(zmax))

z = z * 1E-3
zmax = zmax * 1E-3

ax[0,0].plot(Tv[1:], z[1:], label=r"$T_v$", c='forestgreen', ls='-')
ax[0,0].plot(TPv[1:], z[1:], label=r"$\theta_v$", c='darkgoldenrod', ls='-')
ax[0,0].plot(TPv_f[1:], z[1:], label=r"$<\theta_v>$",\
    c='goldenrod', ls='-', lw=2)
ax[0,0].set_ylim(ymax=zmax)
ax[0,0].set_ylabel(r"z (km)")
ax[0,0].set_xlabel(r"Temperature (K)")
lg0 = ax[0,0].legend(loc='upper left', borderaxespad=.2, frameon=True)

ax[1,0].plot(U[1:], z[1:], label=r"$U$", c='cyan', ls='-', lw=3)
ax[1,0].plot(U_f[1:], z[1:], label=r"$<U>$", c='slateblue', ls='-')
ax[1,0].set_ylim(ax[0,0].get_ylim())
ax[1,0].set_xlabel(r"Wind speed (m/s)")

#N_signed = N2 / np.sqrt(np.abs(N2))
N_signed = np.sign(N2) * np.sqrt(np.abs(N2))
ax[0,1].axvline(0, c='k', ls='-', lw=.8)
#ax[0,1].plot(N2, z[1:], label=r"$N^2$", c='goldenrod', ls='-', lw=3)
#ax[0,1].set_xlabel(r"$N^2\;(1/s^2)$")
ax[0,1].plot(N_signed[1:], z[1:], label=r"$N^2\!/\sqrt{|N^2|}$",\
    c='forestgreen', ls='-')
ax[0,1].set_xlabel(r"$N^2\!/\sqrt{|N^2|}\;(1/s)$")
ax[0,1].set_ylim(ax[0,0].get_ylim())

nhu_signed = np.sign(nhu2) * np.sqrt(np.abs(nhu2))
Fr_signed = np.sign(Fr2) * np.sqrt(np.abs(Fr2)) 
Frconstant = np.max(U_f[z <= 1]) / np.mean(N_signed) / hridge
print "\nmean(N_signed) = %s 1/s" % (np.mean(N_signed))
print "max(U[z < 1 km]) = %s m/s" % (np.max(U_f[z <= 1]))
print "Fr contant = %s" % Frconstant
ax[0,2].axvline(0, c='k', ls='-', lw=.8)
#ax[0,2].plot(nhu_signed, z[1:], c='firebrick', ls='-', lw=3)
#ax[0,2].set_xlabel(r"$(N\,h/U)^2 / \sqrt{|(N\,h/U)^2|}$")
ax[0,2].plot(Fr_signed[1:], z[1:], label=r"$Fr^2/\sqrt{|Fr^2|}$",\
    c='tomato', ls='-')
ax[0,2].set_xlabel(r"$\rm{Fr^2} / \sqrt{|\rm{Fr}^2|}$")
ax[0,2].axvline(Frconstant, c='firebrick', ls='-', lw=3)
ax[0,2].set_ylim(ax[0,0].get_ylim())
ax[0,2].set_xlim(xmin=-.5, xmax=5)


ax[1,1].axvline(0, c='k', ls='-', lw=.8)
ax[1,1].plot(lscorer2[1:], z[1:], label=r"$l_z^2$", c='slateblue', ls='-')
ax[1,1].set_xlabel(r"$l_z^2\;(1/m^2)$")
ax[1,1].set_ylim(ax[0,0].get_ylim())

ax[1,2].axvline(0, c='k', ls='-', lw=.8)
#ax[1,2].plot(Rig[1:], z[1:], label=r"$\rm{Ri}_g$", c='yellowgreen', ls='-')
ax[1,2].plot(Rib[1:], z[1:], label=r"$\rm{Ri}_b$", c='goldenrod', ls='-')
ax[1,2].set_xlabel(r"$\rm{Ri}_b$")
ax[1,2].set_ylim(ax[0,0].get_ylim())

fig.suptitle(data['station'] + ", " + data['launched'])
# fig.text(ax[0,0].get_position().xmin, .2 + .8*ax[0,0].get_position().ymax,\
#     data['station'] + ", " + data['launched'],\
#     ha='left', va='baseline')

fig.tight_layout(w_pad=.15, h_pad=.2, rect=(0,0,1,.98))

figname = data['station'] + " " + data['launched']
figname = figname.replace(": "," ").replace("/","-").replace(":","-")
figname = figname.replace("Launched"," ").replace("(UTC)"," ")
figname = "_".join(figname.strip().split()) + ".pdf"
print "\nsaving figure to file %s" % figname
plt.savefig(figname, format="pdf")

plt.show()

