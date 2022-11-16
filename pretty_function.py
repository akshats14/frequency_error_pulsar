import numpy as np
import errfunctions as ef
from matplotlib import pyplot as plt

def pretty_progress(i, num, time_elap=0):
    k='..'*(int(10*(i+1)/num)+1)
    if time_elap == 0:
        prog_1 = '{1:s}{0:02.1f}%'.format((i+1)*100/num,k)
    else:
        if time_elap < 60:
            time = r'{0:3.1f}s'.format(time_elap/1)
        elif time_elap >= 60 and time_elap < 3600:
            time = r'{0:3.1f}m'.format(time_elap/60)
        else:
            time = r'{0:3.1f}h'.format(time_elap/3600)
        prog_1 = '{1:s}{0:02.1f}% : {2:s}'.format((i+1)*100/num, k, time)
        
    return '{0:s}'.format(prog_1)


def pretty_progress_iter(i_arr, num_arr, time_elap_arr, clean=1):
    
    if clean > 0:
        print('\r{0:s}'.format(' '.rjust(100)), end='')
    i_arr = np.array([i_arr]).flatten()
    num_arr = np.array([num_arr]).flatten()
    time_elap_arr = np.array([time_elap_arr]).flatten()
    progress = []
    for i in range(len(i_arr)):
        prog = pretty_progress(i_arr[i], num_arr[i], time_elap_arr[i])
        progress += [prog]
    progress_pp = ' => '.join(progress)
    print('\r{0:s}'.format(progress_pp),end='')
    
    
def latex_float(self, f, significant=3):
        float_str = "{0:.{1:d}g}".format(f, significant)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0} \times 10^{{{1}}}".format(
                base, int(exponent))
        else:
            return float_str
    
def neat_xticks(ax, xdata, num=3):
    xticks = np.linspace(min(xdata), max(xdata), num)
    expon = int('{0:e}'.format(
        np.diff(xticks)[0]).split('e')[-1]
             )
    if expon < 0:
        deci = expon*-1+1 
    else:
        deci = 0
    xtick_label = [r'{0:1.{1:d}f}'.format(i, deci)
                   for i in xticks]
    xticks = np.round(xticks, decimals=deci)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_label)
    
def pretty_fit_plot(ax, freq, zndata, par, func='sinc', window=2, skip_data=0):
    win = 1/(window)
    f_mask = np.abs(freq - par[1]) <= win
    if len(freq[f_mask]) == 0:
        f_mask = np.abs(freq - np.mean(freq)) <= win
    freq1, zndata1 = freq[f_mask], zndata[f_mask]
    fr_pp=np.linspace(min(freq1), max(freq1), 1000)
    if skip_data == 0:
        ax.plot(freq1, zndata1, alpha=0.5)
    func_fit = ef.shape2func(func)
    try:
        ax.plot(fr_pp, func_fit(fr_pp, *par), '--', alpha=0.5)
    except:
        pass
    neat_xticks(ax, fr_pp, num=3)
    
