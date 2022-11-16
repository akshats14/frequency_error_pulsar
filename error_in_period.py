import numpy as np
from numpy import pi, sin, cos, arctan2, sinc
import math
import time
import glob
import os, sys
import errfunctions as ef

from scipy.optimize import curve_fit
from scipy.signal import square
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.integrate import quad

from stingray import Lightcurve
from stingray.events import EventList
from stingray.pulse.search import z_n_search

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta

from matplotlib import pyplot as plt


class Period_error():

    def __init__(self, sim, sim_num, nharm):
        self.sim = sim
        self.sim_num = sim_num
        self.nharm = nharm

        self.choose_shape = {}
        self.choose_shape.update(dict.fromkeys(
            ['sin','sine','sinu','sinusoid','sinusoidal'],
            ef.shape_sinu))
        self.choose_shape.update(dict.fromkeys(
            ['gauss', 'gaussian'], ef.shape_gauss))
        self.choose_shape.update(dict.fromkeys(
            ['rect','rectangle'], ef.shape_rect))

        self.zn_fit = {}
        self.zn_fit.update(dict.fromkeys(
            ['sinc','sinc2','sinc^2','sinc**2','sincsquared'],
            ef.sinc2))
        self.zn_fit.update(dict.fromkeys(
            ['gauss', 'gaussian'], ef.gauss))
        self.zn_fit.update(dict.fromkeys(
            ['quad','quadratic'], ef.quad))

        self.zn_profile = {}
        self.zn_profile.update(dict.fromkeys(
            ['sinc','sinc2','sinc^2','sinc**2','sincsquared'],
            ef.sinc2))
        self.zn_profile.update(dict.fromkeys(
            ['gauss', 'gaussian'], ef.gauss))
        self.zn_profile.update(dict.fromkeys(
            ['quad','quadratic'], ef.quad))




    def freq2omega(self, freqs):
        freqs = np.array(freqs)
        return 2 * pi * freqs

    def omega2freq(self, omega):
        omega = np.array(omega)
        return omega / (2 * pi)

    def shape_from_profile(self):
        '''
        define a shape of the pulsar signal
        '''
        return

    def shape_from_toa(self):
        '''
        define a shape of the pulsar signal
        '''
        return

    def choose_shape_profile(self,shape='sine'):
        self.shape = shape
        self.shape_func=self.choose_shape[shape.lower()]
        self.shape_fit = self.shape_func

    def simulate_toa(self,T_obs,bintime,para,shape='sine'):
        '''
        Given a shape of the pulse profile,
        the observational time and the bintime
        this code creates artificial pulsar like
        data where time of arrive of individual
        photons are returned in seconds from Zero
        '''
        self.para = para
        a, b, freq, phase, width = para
        self.inj_freq = freq
        T_new= np.round((T_obs) / bintime) * (bintime)
        self.T_obs = T_new
        self.times=np.linspace(0, T_new, int(T_new/bintime) + 1)
        self.choose_shape_profile(shape)
        self.counts = self.shape_func(
            self.times, *para) * bintime
        lc = Lightcurve(self.times, self.counts, gti=[[0, T_obs]],
                        dt=bintime, skip_checks=True)
        events = EventList()
        events.simulate_times(lc)
        self.toa=events.time
        self.T_obs_act = self.toa[-1] - self.toa[0]

    def read_toa(self, file, inst_num = 1, inst_name = 'None' ):
        '''
        Read pulsar data file to get time of arrive of individual
        photons.
        '''
        self.hdu = fits.open(file)
        inst_ind = 1
        self.pulsar_data_list = []
        self.pulsar_toa_list = []
        for instr in range(1, inst_num+1, 1):
            try:
                self.pulsar_tab = Table(self.hdu[instr].data)
                coln=[]
                for col in self.pulsar_tab.colnames:
                    if col.lower() == 'time':
                        self.col_time = col
                        coln += [col]
                    if col.lower() == 'energy':
                        self.col_energy = col
                        coln += [col]
                self.pulsar_col = coln
                self.pulsar_data_list += [self.pulsar_tab[coln]]
                self.pulsar_toa_list += [np.array(
                    self.pulsar_tab[self.col_time])]
            except:
                pass
        self.rotpara = [57480.5635028439,
                         29.6553306828504,
                         -3.69261197216621e-10,
                         -3.9353064617231e-20]

        times = self.pulsar_toa_list[0]
        times = times - times[0]
        self.T_obs = times[-1]


    def include_gap(self):
        '''
        remove certain data in the time series
        '''

    def t_ref(self):
        '''
        I don't know
        '''

    def get_toa(self):
        '''
        get toa either simulated or actual
        '''
        if self.sim == 0:
            T_obs = 20
            bintime = 1/32
            a, b = 200, 100
            freq = 1./2
            phase = np.random.random()
            para = [a, b, freq, phase, [0]]
            self.simulate_toa(T, bintime, para)
        elif self.sim > 7:
            dirt1 = os.path.join('..', 'Pulsar_Physics', 'Assignments')
            dirt2 = os.path.join('Assgn-4', 'ASTROSAT')
            file  = 'ObsID406_02741_laxpc_bary.fits'
            file_path = os.path.join(dirt1,dirt2,file)
            self.read_toa(file_path,1, 'CZTI')
            self.toa = self.pulsar_toa_list[0]
            self.rotpara = [57480.5635028439,
                         #29.6553306828504,
                           29.65540342577391,
                         -3.69261197216621e-10,
                         -3.9353064617231e-20]

    def plot_time_series(self,freq_guess,
                         save=0, show=0, tbin_plot=None):
        '''
        Plot the time of arrival time series
        '''
        #freq_guess = omega_guess/(2*pi)
        period_guess = 1/freq_guess
        if tbin_plot == None:
            tbin_plot = period_guess/10
        if hasattr(self ,'toa'):
            times = self.toa
        else:
            self.get_toa()
            times = self.toa
        cycle = 5
        fig, ax = plt.subplots(1, 2,
                           gridspec_kw={'width_ratios': [1, 3]},
                           figsize=(7,2))
        if self.sim == 0 :
            times = times-times[0]
            time_plot = cycle * period_guess
            data_plot = times[times <= time_plot]
            tmax = data_plot[-1] + tbin_plot
            ax[1].hist(data_plot, bins=np.arange(0,tmax,tbin_plot))
            ax[1].set_xlim(0,tmax)
            t_array = np.arange(0,tmax,tbin_plot/10)
            ax[1].plot(t_array,
                     tbin_plot * self.shape_func(t_array, *self.para),
                     lw=4, alpha=0.7)
            t_array_pp = np.arange(0, period_guess + tbin_plot/10,
                                   tbin_plot/10)
            ax[1].set_xlabel('Time [s]')
            ax[0].plot(t_array_pp/period_guess,
                     self.shape_func(t_array_pp, *self.para),
                     lw=2, alpha=1)
            ax[0].set_xlabel(r'Phase ($\phi$)')
        elif self.sim > 7 :
            p=1./freq_guess
            toa_all = np.array([])
            for tt in self.pulsar_toa_list:
                toa_all = np.append(toa_all, tt)
            toa_all = toa_all - np.min(toa_all)
            ph = toa_all*(freq_guess) % cycle
            h,b=np.histogram(ph, bins=128, density=True)
            ax[1].plot(0.5 * (b[1:] + b[:-1]) * p, h)
            ax[1].set_xlabel('Time [s]')
            ph = toa_all * (freq_guess) % 1
            h,b=np.histogram(ph, bins=64, density=True)
            ax[0].plot(0.5 * (b[1:] + b[:-1]), h, lw=2)
            ax[0].set_xlabel(r'Phase ($\phi$)')
        fig.tight_layout()
        if save > 0 :
            fig.savefig('test.pdf')
        if show > 0:
            plt.show()
        else:
            plt.close()

    def guess_freq_array(self, z_res=5000, factor=2 ):
        '''
        which frequency array to measure Z_n^2
        '''
        win = 1/(factor*self.T_obs)
        if self.sim == 0:
            fr = self.inj_freq
            binsize = win*(1/z_res)
            rand = np.random.uniform(0.2, 0.8) * 0.5 * binsize
            shift = (-1)**np.random.randint(1,3)
            fr += shift*rand
            self.g_fr = fr
            guess_fr_array = np.linspace(
                #fr-1/(2*pi*self.T_obs), fr+1/(2*pi*self.T_obs), z_res+1)
                fr - win, fr + win, z_res+1)
        elif self.sim > 7:
            fr = self.rotpara[1]
            self.g_fr = fr
            guess_fr_array = np.linspace(
                    fr - win, fr + win, z_res+1) 
        self.guess_freq = guess_fr_array
        self.z_res = z_res

    def measure_zn2(self, nharm=1, nbin=32, fdots=0 ):
        '''
        measure the value of Z_n^2 of one given value of

            Parameters
        ----------
        times : array-like
            the event arrival times

        frequencies : array-like
            the trial values for the frequencies

        Other Parameters
        ----------------
        nbin : int
            the number of bins of the folded profiles

        segment_size : float
            the length of the segments to be averaged in the
            periodogram

        fdots : array-like
            trial values of the first frequency derivative (optional)

        expocorr : bool
            correct for the exposure (Use it if the period is
            comparable to the
            length of the good time intervals.)

        gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good time intervals

        weights : array-like
            weight for each time. This might be, for example,
            the number of counts
            if the times array contains the time bins of a
            light curve
        '''
        #if not hasattr(self ,'guess_freq'):
        #self.guess_freq_array(z_res = z_res)
        self.freq, self.znstat = z_n_search(self.toa, self.guess_freq,
                    nharm=nharm, nbin=nbin, segment_size=self.T_obs,
                    gti=None, fdots=fdots)


    def measure_zn2_iter(self):
        '''
        measure Z_n^2 for array of angular frequency
        '''



    def zn2_with_fdarray(self):
        '''
        why?
        '''


    def get_guess_par(self):
        '''
        p0 = ?
        '''
    
    def zero_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges
    
    def fwhm_m_ind(ydata):
        rel_half = 0.5*(np.max(ydata))
        if np.min(ydata) > rel_half:
            rel_half = 0.5*(np.max(ydata) + np.min(ydata))
        fwhm_mask = (ydata < rel_half ) * 1
        runs = zero_runs(fwhm_mask)
        leng = np.diff(runs, axis=1)
        fwhm_x = runs[np.argmax(leng)]
        fwhm_mid = int(np.mean(fwhm_x))
        return fwhm_mid, rel_half, fwhm_x 

    

    def znstat_fit(self, zn_fit_shape='sinc2', window=2):
        '''
        curve_fit
        '''
        sinc=['sinc','sinc2','sinc^2','sinc**2','sincsquared']
        guass=['gauss', 'gaussian']
        quad=['quad','quadratic']
        try:
            self.zn_func=self.zn_fit[zn_fit_shape.lower()]
        except:
            self.zn_func = ef.sinc2
            print('poor function mentioned. sinc2 used')
        imax = np.argmax(self.znstat)
        fwhm_mid, rel_half, fwhm_x  = fwhm_m_ind(self.znstat)
        self.f_c = self.freq[fwhm_mid]
        self.f_cm, self.A = self.freq[imax], self.znstat[imax]
        if zn_fit_shape in sinc:
            p0 = [self.znstat[imax],
                  self.f_c,
                  1/(pi*self.T_obs)]
            bias = 0

        elif zn_fit_shape in guass:
            p0 = [self.znstat[imax],
                  self.f_c,
                  1.2/(pi*self.T_obs)]
            bias = 0

        elif zn_fit_shape in quad:
            W = 1/(np.pi * self.T_obs)
            err_co = (self.A/3)/(W**2)
            p0 = [self.znstat[imax],
                  0, err_co]
            bias = self.f_c
           
        win = 1/(window*T_obs)
        x_mask = np.abs(freq - self.f_c) <= win

        par, cov = curve_fit(self.zn_func, self.freq[x_mask] - bias,
                             self.znstat[x_mask], p0=p0
                       )
        err = np.sqrt(np.abs(np.diag(cov)))
        self.par_fit = par
        self.cov_fit = cov
        self.err_fit = err
        self.H_F_W = par + np.array([0, bias, 0])

    def znstat_fit_all(self):

        self.pars = []
        for func in ['sinc2', 'gauss', 'quad']:
            self.znstat_fit(zn_fit_shape=func)
            self.pars += [self.H_F_W]
        self.pars = np.array(self.pars)

    def latex_float(self, f, significant=3):
        float_str = "{0:.{1:d}g}".format(f, significant)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0} \times 10^{{{1}}}".format(
                base, int(exponent))
        else:
            return float_str

    def neat_xticks(self, ax, xdata, num=3):
        xticks = np.linspace(min(xdata), max(xdata), num)
        expon = int('{0:e}'.format(
            np.diff(xticks)[0]).split('e')[-1]
                 )
        if expon < 0:
            deci = expon*-1+1 
        else:
            deci = 0
        xtick_label = [r'{0:1.{1:d}f}'.format(i, dec)
                       for i in xticks]
        xticks = np.round(xticks, decimals=deci)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_label)

    def ploting(self, fit=0, large=1):
        '''
        Plotting
        '''
        fig, ax = plt.subplots(2,2,
                    #gridspec_kw={'width_ratios': [1, 2]},
                   figsize=(7,4))

        if large < 1:
            ax[0,0].plot(self.freq, self.znstat,'-')
            self.neat_xticks(ax[0,0], self.freq, 3)
        else:
            fr = self.inj_freq
            guess_freq1 = np.linspace(
                fr-2/(1*self.T_obs), fr+2/(1*self.T_obs), 1000)
            freq1, znstat1 = z_n_search(self.toa, guess_freq1,
                    nharm=self.nharm, nbin=32, segment_size=self.T_obs,
                    gti=None, fdots=0)
            ax[0,0].plot(freq1, znstat1)
            self.neat_xticks(ax[0,0], freq1, 3)


        fr=np.linspace(min(self.freq), max(self.freq), 1000)
        func = ['sinc2', 'gauss', 'quad']

        for i in range(1,len(func)+1,1):
            self.zn_func=self.zn_fit[func[i-1].lower()]
            pars = self.pars[i-1]
            guess_freq = pars[1]

            ax0 = ax[int(i/2),i%2]
            ax0.plot(self.freq, self.znstat,'-o')
            ax0.plot(fr, self.zn_func(fr, *pars),'--', label=func[i-1])
            self.neat_xticks(ax0, self.freq, 3)
            title = r'$\Delta f = {0:s}$ Hz & '.format(
                self.latex_float(guess_freq-self.inj_freq))
            if func[i-1] == 'sinc2':
                error = np.sqrt(3/pars[0])*pars[2]
            elif func[i-1] == 'gauss':
                error = np.sqrt(3/pars[0])*pars[2]/1.2
            elif func[i-1] == 'quad':
                error = 1/np.sqrt(pars[2])
            title += r'$\sigma f = {}$'.format(
                self.latex_float(error))

            ax0.set_title(title)
            ax0.legend(loc=0)
        fig.tight_layout()
        plt.show()


    def zn2array_to_freq(self):
        '''
        extract frequency when
        '''

    def pulse_profile(self):
        x=1

    def profile_to_parameters(self):
        x=1

