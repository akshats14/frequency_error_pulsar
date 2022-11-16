import numpy as np
from numpy import pi, sin, cos, arctan2, sinc
import math
import time
from datetime import timedelta
import glob
import os
import errfunctions as ef
from error_in_period import Period_error
import pretty_function as ppf

#from scipy.optimize import curve_fit
#from scipy.signal import square
#from scipy.signal import chirp, find_peaks, peak_widths
#from scipy.integrate import quad

from stingray import Lightcurve
from stingray.events import EventList
from stingray.pulse.search import z_n_search

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta

from matplotlib import pyplot as plt


per=Period_error(0,10,3)
a, b, fr, T = 10, 5, 1, 600
bintime = (1/fr)/64
num = per.sim_num
nharm = per.nharm
shapes = ['sine', 'rect', 'gauss']
#shapes = ['rect']
path0 = 'sim_para05_'

for shape in shapes:
    starttime_0 = time.process_time()
    print('\n{0:s} shaped sims are running'.format(shape))
    dirt = r'sim_{0:s}'.format(shape)
    if not os.path.exists(dirt):
        os.makedirs(dirt)
    time_elap_1 = 0
    for i in range(num): 

        starttime_1 = time.process_time()
        ppf.pretty_progress_iter(i, num, time_elap_1, 0)
        
        phase=np.random.random()
        para = [a,b,fr,phase,0.5]
        per.simulate_toa(T, bintime, para, shape=shape)
        per.guess_freq_array(z_res=500, factor=2)
        k0=0
        harm = []
        iters = range(0,nharm,1)
        knum = len(iters)
        zn_zeros = np.zeros((knum, len(per.guess_freq)))
        
        time_elap_2 = 0
        for j in iters:
            starttime_2 = time.process_time()
            ppf.pretty_progress_iter(
                [i,j], [num, knum], [time_elap_1, time_elap_2], 1)
            
            harm += [j+1]
            per.measure_zn2(nharm=j+1, nbin=32, fdots=0)
            zn_zeros[k0] = per.znstat
            k0+=1
            endtime_2 = time.process_time()
            time_elap_2 = endtime_2 - starttime_2
            
        zn_table = Table()
        zn_table['freq'] = per.freq
        for k in range(len(zn_zeros)):
            coln = r'Z{0:02d}'.format(harm[k])
            zn_table[coln] = zn_zeros[k]
        path = 'zn_table_{2:s}{1:s}_{0:04d}.fits'.format(i, shape, path0)
        path = os.path.join(dirt,path)
        zn_table.write(path, format='fits', overwrite=True) 
        hdu = fits.open(path)
        hdr = hdu[0].header
        hdr['a'] = a
        hdr['b'] = b
        hdr['freq_inj'] = fr
        hdr['freq_rnd'] = per.g_fr
        hdr['T'] = T 
        hdr['T_obs'] = per.toa[-1] - per.toa[0] # should have been per.T_obs
        hdr['bintime'] = bintime
        hdr['f_resol'] = per.freq[1] - per.freq[0]
        hdr['shape'] = shape
        hdr['exctime'] = time.process_time() - starttime_1
        hdu[0].header = hdr
        hdu.writeto(path, overwrite=True)
        endtime_1 = time.process_time()
        time_elap_1 = endtime_1 - starttime_1
    endtime_0 = time.process_time()
    time_elap_0 = endtime_0 - starttime_0
    print(' | Time elapsed {0:s}'.format(str(timedelta(seconds=time_elap_0))))
print('Simulation complete')


sinc_list=['sinc','sinc2','sinc^2','sinc**2','sincsquared']
gauss_list=['gauss', 'gaussian']
quad_list=['quad','quadratic']


for shape in shapes:
    print('\n{0:s} shaped extractions are running'.format(shape))
    sim_pars = []
    plotter = 0
    dirt = r'sim_{0:s}'.format(shape)
    time_elap_1 = 0
    for sim in range(num):
        starttime_1 = time.process_time()
        ppf.pretty_progress_iter(sim, num, time_elap_1, 0)

        path = 'zn_table_{2:s}{1:s}_{0:04d}.fits'.format(sim, shape, path0)
        path = os.path.join(dirt,path)

        hdu = fits.open(path)
        data = Table(hdu[1].data)
        freq = data['freq']
        hdr = hdu[0].header
        T_obs = hdr['T'] #notice we are not using hdr[T_obs] so that
        #the analysis is similar for every sim

        nharms_fits = [int(i[1:]) for i in data.colnames[1:]]
        num_harm = len(nharms_fits)
        if plotter >0:
            fig, ax = plt.subplots(num_harm, 3,
                               #gridspec_kw={'width_ratios': [1, 3]},
                               figsize=(8,1*num_harm))
        harm_wise_fit_para = []
        time_elap_2 = 0
        for n in range(num_harm):
            starttime_2 = time.process_time()
            ppf.pretty_progress_iter(
                [sim,n], [num, num_harm], [time_elap_1, time_elap_2], 1)
            col = nharms_fits[n]
            coln = r'Z{0:02d}'.format(col)
            if n == 0:
                harm_data = data[coln]
            else:
                col_p = nharms_fits[n-1]
                coln_p = r'Z{0:02d}'.format(col_p)
                harm_data = data[coln]-data[coln_p]

            zndata = data[coln]
            par_hws, cov_hws, freq_para_hws = ef.zn_fit(freq, harm_data, T_obs, func='sinc', window=2*col)
            par_ts, cov_ts, freq_para_ts = ef.zn_fit(freq, zndata, T_obs, func='sinc', window=2*pi)
            par_tg, cov_tg, freq_para_tg = ef.zn_fit(freq, zndata, T_obs, func='gauss', window=2*pi)
            par_tq, cov_tq, freq_para_tq = ef.zn_fit(freq, zndata, T_obs, func='quad', window=2*pi)
            harm_wise_fit_para += [[*par_hws, *freq_para_hws, 
                                   *par_ts, *freq_para_ts, 
                                   *par_tg, *freq_para_tg, 
                                   *par_tq, *freq_para_tq]]

            if plotter >0:
                if num_harm == 1:
                    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
                else:
                    ax1, ax2, ax3 = ax[n,0], ax[n, 1], ax[n, 2]

                ppf.pretty_fit_plot(
                    ax1, freq, harm_data, par_hws, func='sinc', window=2*T_obs, skip_data=0)
                ppf.pretty_fit_plot(
                    ax2, freq, zndata, par_ts, func='sinc', window=2*pi*T_obs, skip_data=0)
                ppf.pretty_fit_plot(
                    ax2, freq, zndata, par_tg, func='gauss', window=2*pi*T_obs, skip_data=1)
                ppf.pretty_fit_plot(
                    ax3, freq, zndata, par_tq, func='quad', window=2*pi*T_obs, skip_data=0)
        
            endtime_2 = time.process_time()
            time_elap_2 = endtime_2 - starttime_2
        if plotter >0:
            fig.tight_layout()    
            plt.show()

        sim_pars += [ harm_wise_fit_para ]     

        
        endtime_1 = time.process_time()
        time_elap_1 = endtime_1 - starttime_1
    sim_pars = np.array(sim_pars)
    sim_path = 'PE_{1:s}_{0:s}.npy'.format(shape, path0)
    sim_path = os.path.join(dirt, sim_path)
    np.save(sim_path, sim_pars)
print('\nExtraction complete')

'''
sim_pars index
#,_,_ = index of simulation

#,##,_ = index of harmonics for simulation # 

#, ##, ###  [FWHM[0], rel_height, freq_fwhm_mid, freq_max]

#00 harmonic wise sinc2 Amplitude fit
#01 harmonic wise sinc2 frequency fit
#02 harmonic wise sinc2 Width fit
#03 harmonic wise FWHM
#04 harmonic wise rel height of FWHM. if it is 0.5 it is fine. anything else :/
#05 harmonic wise [freq] mid of FW at HM (FWHM)
#06 harmonic wise frequency at maximum Zn^2 - Zn-1^2
#07 Total Zn2 wise sinc2 Amplitude fit
#08 Total Zn2 wise sinc2 frequency fit
#09 Total Zn2 wise sinc2 Width fit
#10 Total Zn2 wise FWHM
#11 Total Zn2 wise rel height of FWHM. if it is 0.5 it is fine. anything else :/
#12 Total Zn2 wise [freq] mid of FW at HM (FWHM)
#13 Total Zn2 wise frequency at maximum Zn^2
#14 Total Zn2 wise gauss Amplitude fit
#15 Total Zn2 wise gauss frequency fit
#16 Total Zn2 wise sinc2 Width fit
#17 Total Zn2 wise gauss
#18 Total Zn2 wise rel height of FWHM. if it is 0.5 it is fine. anything else :/
#19 Total Zn2 wise [freq] mid of FW at HM (FWHM)
#20 Total Zn2 wise frequency at maximum Zn^2
#21 Total Zn2 wise quad Amplitude fit
#22 Total Zn2 wise quad frequency fit
#23 Total Zn2 wise quad Width fit
#24 Total Zn2 wise FWHM
#25 Total Zn2 wise rel height of FWHM. if it is 0.5 it is fine. anything else :/
#26 Total Zn2 wise [freq] mid of FW at HM (FWHM)
#27 Total Zn2 wise frequency at maximum Zn^2
'''



