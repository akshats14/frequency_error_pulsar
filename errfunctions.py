import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import curve_fit
from scipy.signal import chirp, find_peaks, peak_widths
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt


sinc_list=['sinc','sinc2','sinc^2','sinc**2','sincsquared']
gauss_list=['gauss', 'gaussian']
quad_list=['quad','quadratic']


def sinc2(x, A, mean, width):
    return A * np.sinc((x - mean)/(width * np.pi))**2

def gauss(x, amp, mean, sigma):
    x1 = (x - mean)/sigma
    return amp * np.exp(-0.5 * x1**2)

def quad(x, amp, mean_d, err_co):
    return amp - err_co * ((x-mean_d)**2)


zn_shape = {}
zn_shape.update(dict.fromkeys(sinc_list, sinc2))
zn_shape.update(dict.fromkeys(gauss_list, gauss))
zn_shape.update(dict.fromkeys(quad_list, quad))

def shape2func(shape='sinc'):
        return zn_shape[shape.lower()]

def shape_sinu(time, a, b, freq, phase, width=0.5):
    '''
    for given time and source parameters,
    gives expected count for a sinusoid shaped pulsar
    '''
    time=np.array(time)
    phi = 2 * pi * phase
    omega = 2 * pi * freq
    return a + b * np.sin(omega * time + phi)

def shape_rect(time, a, b, freq, phase, width=0.5):
    '''
    define a shape of the pulsar signal
    '''
    P = 1/freq
    tt = np.asarray(time)
    tt = (tt + phase * P) % P
    tt = tt - (1 - width) * P   
    return np.heaviside(tt,1)*(2*b)+(a-b)

def shape_gauss(time, a, b, freq, phase, width=0.5):
    '''
    define a shape of the pulsar signal
    '''
    P = 1/freq
    tt = np.asarray(time)
    tt = (tt + phase * P) % P
    tt = tt-0.5*P
    y = np.exp(-0.5* (tt/(P*width/4))**2 )
    return b*y + a

def shape_triag(x):
    '''
    define a shape of the pulsar signal
    '''
    return

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def fwhm_m_ind(ydata):
    
    rel_half_height = 0.5*(np.max(ydata))
    if np.min(ydata) > rel_half_height:
        rel_half_height = 0.5*(np.max(ydata) + np.min(ydata))
    fwhm_mask = (ydata < rel_half_height ) * 1
    zeros_in_mask = zero_runs(fwhm_mask)
    zero_length = np.diff(zeros_in_mask, axis=1)
    fwhm_xrange_index = zeros_in_mask[np.argmax(zero_length)]
    fwhm_mid_index = int(np.mean(fwhm_xrange_index))
    return fwhm_mid_index, rel_half_height, fwhm_xrange_index 

def zn_fit(freq, zndata, T_obs, func='sinc', window=2):
    debug = 0
    f_bin = freq[1]-freq[0]
    ind_max = np.argmax(zndata)
    freq_max, A_max = freq[ind_max], zndata[ind_max]
    rel_height = 0.5
    if np.min(zndata) > rel_height*np.max(zndata):
        rel_height = 0.5*(np.max(zndata) + np.min(zndata))/np.max(zndata)
    results_half = peak_widths(zndata, [ind_max], rel_height=rel_height) #scipy function
    #read its documentaion if needed
    FWHM = results_half[0]*f_bin
    freq_fwhm_mid = (np.mean(results_half[2:]))*f_bin + freq[0]
    
    W = 1/(pi * T_obs)
    err_coeff = (A_max/3)/(W**2)
    
    win = 1/(window*T_obs)
    f_mask = np.abs(freq - freq_fwhm_mid) <= win
    
    bias = 0
    p0 = [ A_max, freq_fwhm_mid, W]
    if func in sinc_list:
        func_fit = sinc2
    elif func in gauss_list:
        func_fit = gauss
    elif func in quad_list:
        func_fit = quad
        bias = freq_fwhm_mid
        p0[2] = err_coeff
    else:
        func_fit = sinc2     
    p0[1] = p0[1] - bias
    
    if debug == 1:
        plt.plot(freq, zndata)
        plt.title(str(p0))
        plt.show()
        
    try:
        par, cov = curve_fit(func_fit, freq[f_mask]-bias, 
                                 zndata[f_mask], p0=p0
                           )
        par[1] += bias
    except:
        par, cov = np.zeros(3), np.zeros((3,3))
        #print('{0:s} fit not achieved'.format(func))
        
        
    return par, cov, [FWHM[0], rel_height, freq_fwhm_mid, freq_max]

def clean_data(data, filters=['abs','pos']):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if 'abs' in filters:
        data = np.abs(data)
    if 'pos' in filters:
        data = data[data>0]
    return data   

def filtered(data, sigma=3, filters=['abs','pos']):
    data1 = clean_data(data, filters=filters)
    filtered_data = sigma_clip(data1, sigma=sigma, maxiters=5)
    filtered_data = np.array(filtered_data[~filtered_data.mask])
    return filtered_data

def fil_mean(data, sigma=3, filters=['abs','pos']):
    filtered_data = filtered(data, sigma=sigma, filters=filters)
    return np.mean(filtered_data)

def fil_std(data, sigma=3, filters=['abs','pos']):
    filtered_data = filtered(data, sigma=sigma, filters=filters)
    return np.std(filtered_data)

def fil_mad(data, injection=1, sigma=3, filters=['abs','pos']):
    filtered_data = filtered(data, sigma=sigma, filters=filters)
    diff = np.abs(filtered_data - injection)
    mad = np.ma.median(diff)
    return mad

