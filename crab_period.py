import numpy as np
from numpy import pi, sin, cos, arctan2, sinc
import time
from datetime import timedelta
import glob
import os, sys
import gc
import errfunctions as ef
import pretty_function as ppf

from stingray.pulse.search import z_n_search

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta
#from astropy.stats import sigma_clip


from matplotlib import pyplot as plt

#hdu = fits.open('crab_data/ObsID406_02741_laxpc_bary.fits')

rotpara = [57480.5635028439,
         29.6553306828504,
         -3.69261197216621e-10,
         -3.9353064617231e-20]
f_0 = rotpara[1]
f_dot = rotpara[2]
f_ddot =rotpara[3]


def read_data(dirt, file_glob):
    glob_path = os.path.join(dirt, file_glob)
    files = glob.glob(glob_path)
    files.sort()
    toa_all = np.array([])
    for file in files:
        hdu = fits.open(file)
        toa = hdu[1].data['TIME']
        toa_all = np.append(toa_all, toa)
    toa_all = np.sort(toa_all)
    astrosat_starttime = toa_all[0]
    toa_all_zerostart = toa_all - astrosat_starttime       
        
    return toa_all_zerostart, astrosat_starttime

load_fits = 0
toa_npy = int(sys.argv[1])
toa_files = '1876_*.fits'
#toa_files = 'ObsID406_02741_laxpc_bary.fits'
u0 = time.process_time()
if load_fits > 0:
    #toa_all_zerostart, astrosat_starttime = read_data(os.getcwd(),'ObsID406_02741_laxpc_bary.fits')
    toa_all_zerostart, astrosat_starttime = read_data(os.getcwd(), toa_files)
    toa_npy = 1
else:
    toa_all_zerostart = np.load('toa_for_crab{0:1d}.npy'.format(toa_npy))
    astrosat_starttimes = [254952000.0006468, 
                          255025889.25867856, 
                          255138315.9275132, 
                          255238663.91626203]
    astrosat_starttime = astrosat_starttimes[toa_npy-1]
    
v0 = time.process_time()
#astrosat_starttime = 254952000.0006468
print('ToA load time {0:s}'.format(str(timedelta(seconds=v0-u0))))

secondwise_binc_file = 'crab_secondwise_bincount_001.npy'
overwrite = 1
gap_visual = 0
plotter = 0
extraction = 0
T = int(sys.argv[2])
nharm = 6
sim_num = 400
shape = 'crab'
dirt = r'per_{0:s}'.format(shape)
if not os.path.exists(dirt):
    os.makedirs(dirt)

    
print('Observation clip {0:5.1f}s'.format(T))    

if load_fits > 0 or toa_npy > 0:
    T_obs = (toa_all_zerostart[-1] - toa_all_zerostart[0]) #Ideally second term is not needed
binsize=1

u1 = time.process_time()
if overwrite == 0 and os.path.exists(secondwise_binc_file):
    sec_data = np.load(secondwise_binc_file)    
    bin_times= np.arange(0, len(sec_data), 1)    
else:
    sec_bin = np.histogram(toa_all_zerostart, bins=np.arange(0, T_obs+binsize, binsize))
    sec_data = sec_bin[0]
    net_time = len(sec_data[sec_data>0])
    bin_times = sec_bin[1][:-1]
    np.save(secondwise_binc_file, sec_data)
    del(sec_bin)
    gc.collect()
v1 = time.process_time()
print('Binning time {0:s}'.format(str(timedelta(seconds=v1-u1))))


#avg_count = ef.fil_mean(sec_data, sigma=2, filters=['abs','pos'])
#std_count = ef.fil_std(sec_data, sigma=3, filters=['abs','pos'])
#print(avg_count, std_count)
#sec_data[ sec_data < avg_count - 2 * std_count ] = 0


    
net_time = len(sec_data[sec_data>0]) * binsize
if load_fits > 0  or toa_npy > 0:
    print('Eff. T_obs:  {0:4.3f} ks ; Total Observ: {1:4.3f} ks'.format(
    net_time/1e3, T_obs/1e3))

obs_list = []
j, counter = 0, 0
bin_index = np.arange(0, len(sec_data), 1, dtype=int)
eff_data_index = bin_index[sec_data > 0]

u2 = time.process_time()
time_elap_1 = 0
while j < sim_num and counter < 5000:
    starttime_1 = time.process_time()
    ppf.pretty_progress_iter(j, sim_num, time_elap_1, 0)
    counter += 1
    st_index = np.random.choice(bin_index)
    en_index = st_index + np.round(T/binsize)
    obs_index = np.arange(st_index, en_index, 1, dtype=int)
    obs_eff_ind = np.intersect1d(obs_index, eff_data_index)
    gapless_fra = len(obs_eff_ind)/len(obs_index)
    if gapless_fra >= 0.95:
        j+=1
        rand = np.random.uniform(-0.5,0.5)
        start_obstime = (st_index + rand) * binsize
        obs_list+= [start_obstime]
    endtime_1 = time.process_time()
    time_elap_1 = endtime_1 - starttime_1
print('')
print('attempts: {0:5d} ; Observs {1:5d}'.format(counter, len(obs_list)))
v2 = time.process_time()
print('Obs. clip time {0:s}'.format(str(timedelta(seconds=v2-u2))))    
del(sec_data)
gc.collect()

    
time0 = '2010-01-01T00:00:00'
t0 = Time(time0,format='isot',scale='utc')
dt = TimeDelta(astrosat_starttime, format='sec')
delt_t = (t0 + dt).mjd - rotpara[0]
delt_ts = TimeDelta(delt_t, format='jd').sec

sim_pars = []
u3 = time.process_time()
time_elap_2 = 0
for i in range(len(obs_list)):
    starttime_2 = time.process_time()
    ppf.pretty_progress_iter(i, len(obs_list), time_elap_2, 0)
    start_time = obs_list[i]
    end_time = start_time + T
    if i > 0:
        toa_all_zerostart = np.load('toa_for_crab{0:1d}.npy'.format(toa_npy))
    mask = np.logical_and(toa_all_zerostart >= start_time, 
                          toa_all_zerostart <= end_time )
    toa_clip = toa_all_zerostart[mask]
    del(toa_all_zerostart)
    del(mask)
    gc.collect()
    toa_del = (toa_clip[-1] - toa_clip[0])/2
    delta_toa = toa_del + delt_ts
    f_t = f_0 + 1*f_dot*(delta_toa) + 0*f_ddot*(delta_toa)**2
    t_mid = start_time + toa_del
    if gap_visual > 0:
        plt.figure(figsize=(15,4))
        plt.plot(bin_times, sec_data)
        plot_mark = [start_time, t_mid,  end_time]
        plt.plot(plot_mark, 
                 np.zeros(len(plot_mark))+avg_count, 'x')
        plt.tight_layout()
        plt.show()
    win = 2/(1*T)   
    z_res = 200
    freq = np.linspace(f_t - win, f_t + win, z_res+1)
    
    iters = range(0,nharm,1)
    knum = len(iters)
    harm = []
    k0 = 0
    zn_zeros = np.zeros((knum, len(freq)))
    for j in iters:
#         starttime_2 = time.process_time()
#         ppf.pretty_progress_iter(
#             [i,j], [num, knum], [time_elap_1, time_elap_2], 1)

        harm += [j+1]
        freq, znstat = z_n_search(toa_clip, freq, 
                    nharm=j+1, nbin=32, segment_size=T,
                    gti=None, fdots=0)
        zn_zeros[k0] = znstat
        k0+=1
        del(znstat)
        gc.collect()
    zn_table = Table()
    zn_table['freq'] = freq
    for k in range(len(zn_zeros)):
        coln = r'Z{0:02d}'.format(harm[k])
        zn_table[coln] = zn_zeros[k]
    del(zn_zeros)
    gc.collect()
    path = 'zn_table_{2:06.0f}_{1:s}_{0:04d}.fits'.format(i + (toa_npy-1)*sim_num, 
                                                          shape, T)
    path = os.path.join(dirt,path)
    zn_table.write(path, format='fits', overwrite=True) 
    hdu = fits.open(path)
    hdr = hdu[0].header
    hdr['s_time'] = start_time
    hdr['e_time'] = end_time
    hdr['a_start'] = astrosat_starttime
    hdr['files'] = toa_files
    hdr['fr_mid'] = f_t
    hdr['T'] = T 
    hdr['T_obs'] = (toa_clip[-1] - toa_clip[0])
    hdr['T_mid'] = t_mid
    hdr['f_resol'] = freq[1] - freq[0]
    hdr['shape'] = shape
    #hdr['exctime'] = time.process_time() - starttime_1
    hdu[0].header = hdr
    hdu.writeto(path, overwrite=True)
    del(zn_table)
    del(hdu)
    del(hdr)
    gc.collect()

    if extraction > 0:
        nharms_fits = [int(i[1:]) for i in zn_table.colnames[1:]]
        num_harm = len(nharms_fits)
        if plotter >0:
            fig, ax = plt.subplots(num_harm, 3,
                       #gridspec_kw={'width_ratios': [1, 3]},
                       figsize=(12,1*num_harm))

        harm_wise_fit_para = []
        for n in range(num_harm):
            col = nharms_fits[n]
            coln = r'Z{0:02d}'.format(col)
            if n == 0:
                harm_data = zn_table[coln]
            else:
                col_p = nharms_fits[n-1]
                coln_p = r'Z{0:02d}'.format(col_p)
                harm_data = zn_table[coln]-zn_table[coln_p]

            zndata = zn_table[coln]
            par_hws, cov_hws, freq_para_hws = ef.zn_fit(freq, harm_data, T, func='sinc', window=2*col)
            par_ts, cov_ts, freq_para_ts = ef.zn_fit(freq, zndata, T, func='sinc', window=2*pi)
            par_tg, cov_tg, freq_para_tg = ef.zn_fit(freq, zndata, T, func='gauss', window=2*pi)
            par_tq, cov_tq, freq_para_tq = ef.zn_fit(freq, zndata, T, func='quad', window=2*pi)
            harm_wise_fit_para += [[*par_hws, *freq_para_hws, 
                                   *par_ts, *freq_para_ts, 
                                   *par_tg, *freq_para_tg, 
                                   *par_tq, *freq_para_tq, t_mid, col]]

            if plotter >0:
                if num_harm == 1:
                    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
                else:
                    ax1, ax2, ax3 = ax[n,0], ax[n, 1], ax[n, 2]

    #             ax1.plot(freq, harm_data)
    #             winp = win = 2/(1*T*col)  
    #             ax1.set_xlim(f_t - winp, f_t + winp)

    #             ax2.plot(freq, zn_table[coln])
    #             winp = win = 1/(1*T*1)  
    #             ax2.set_xlim(f_t - winp, f_t + winp)

    #             ax3.plot(freq, zn_table[coln])
    #             winp = win = 1/(1*2*T)  
    #             ax3.set_xlim(f_t - winp, f_t + winp)

                ppf.pretty_fit_plot(
                    ax1, freq, harm_data, par_hws, func='sinc', window=2*T, skip_data=0)
                ppf.pretty_fit_plot(
                    ax2, freq, zndata, par_ts, func='sinc', window=2*T, skip_data=0)
                ppf.pretty_fit_plot(
                    ax2, freq, zndata, par_tg, func='gauss', window=2*T, skip_data=1)
                ppf.pretty_fit_plot(
                    ax3, freq, zndata, par_tq, func='quad', window=2*pi*T, skip_data=0)
        if plotter >0:
            plt.tight_layout()   
            plt.show()
        sim_pars += [ harm_wise_fit_para ]  
    endtime_2 = time.process_time()
    time_elap_2 = endtime_2 - starttime_2
v3 = time.process_time()
print('')
print('Freq. meas. time {0:s}'.format(str(timedelta(seconds=v3-u3))))   

if extraction > 0:
    sim_pars = np.array(sim_pars)
    sim_path = 'PE_{1:05d}_{0:s}_{2:1d}.npy'.format(shape, int(T), toa_npy)
    sim_path = os.path.join(dirt, sim_path)
    np.save(sim_path, sim_pars)
    print('\nExtraction complete')





