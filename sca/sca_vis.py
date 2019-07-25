# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import expon
import glob

import neurodsp as ndsp
import warnings
warnings.filterwarnings('ignore')

from sca import utils

# bokeh imports
import bokeh
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Span, BoxAnnotation, Title
from bokeh.models.widgets import Select, Slider
from bokeh.models.glyphs import Rect
from bokeh.palettes import viridis

import ipywidgets as widgets
from ipywidgets import Layout, HBox, interactive

output_notebook()

def percentile_spectrogram(spg, f_axis, rank_freqs=(8., 12.), pct=(0, 25, 50, 75), sum_log_power=True):
    """ Compute the percentile spectrogram separated via spectral component spg
        Input
        spg : array, 3D (chan x frequency x time)
            The complex (or real) components of the spectrogram
        f_axis : array
            The frequency axis
        rank_freqs : tuple (frequency range)
            The first frequency must be lower or equal to the second value.
            The frequency range identified will be used to rank according to the percentile ranges.
        pct : tuple
            Percentile of the spectrogram cut offs
        sum_log_power : boolean
            if True, the sum of the log powers (based 10) is calculated.
            if False, the sum of the powers is calculated
        Return
        power_dgt : array
            indices in which each power data corresponds to each percentile category
        power_binned : array
            power data corresponds to each percentile category
    """
    f_ind = np.where(np.logical_and(
        f_axis >= rank_freqs[0], f_axis <= rank_freqs[1]))

    if sum_log_power:
        power_vals = np.sum(np.log10(spg[f_ind, :][0]), axis=0)
    else:
        power_vals = np.sum(spg[f_ind, :][0], axis=0)

    bins = np.percentile(power_vals, q=pct)
    power_dgt = np.digitize(power_vals, bins, right=False)
    power_binned = np.asarray(
        [np.mean(spg[:, power_dgt == i], axis=1) for i in np.unique(power_dgt)])
    
    return power_dgt, power_binned

def power_examples(data, fs, t_spg, pwr_dgt, rank_freqs):
    """ Set up data for power examples from the raw data
        Input
        data : array, sD (chan x frequency)
            raw data
        fs : int
            sampling frequency
        power_dgt : array
            indices in which each power data corresponds to each percentile category
        rank_freqs : tuple (frequency range)
            The first frequency must be lower or equal to the second value.
        Return
        pwr_ex : dict
            raw and filtered random window of data within a certain power bin
    """
  
    plot_t=1.
    ymin, ymax = 0, 0
    N_cycles=5
    power_adj=5
    
    # filter data and multiplier power constant for ease of visualization
    if power_adj:
        data_filt = ndsp.filt.filter_signal(
            data, fs, 'bandpass', f_range=rank_freqs, n_cycles=N_cycles) * power_adj

    plot_len = int(plot_t*fs/2)
    t_plot = np.arange(-plot_len, plot_len) / fs
    
    pwr_ex = {}
    # loop through bins
    for ind, j in enumerate(np.unique(pwr_dgt)):
        # grab a random window of data that fell within the current power bin
        plot_ind = int(t_spg[np.where(pwr_dgt == j)[0]][np.random.choice(
            len(np.where(pwr_dgt == j)[0]))] * fs)
        y = data[plot_ind - plot_len:plot_ind + plot_len]
        pwr_ex['raw_y' + str(ind+1)] = y - y.mean()
        pwr_ex['filt_y'+ str(ind+1)] = data_filt[plot_ind - plot_len:plot_ind + plot_len]

    pwr_ex['t_plot'] = t_plot
    
    return pwr_ex     

def plot_pct_spectrogram(sc, chan=0, rank_freqs=(8,12), pct_step=25, plot_side_length=350, show_pct_ex=False):
    """ Plot percentile spectrogram
        Input
        sc : SCA object
            spectral component analysis object
        chan : int
            channel
        rank_freqs : tuple (frequency range)
            The first frequency must be lower or equal to the second value.
        pct_step : int
            Percentile step size
        plot_side_length : int
            form a square plot
        show_pct_ex : boolean
            If true, show the percentile example from raw data.
            Only 4 panels will show.
    """
    # calculating the percentile spectrogram
    pct_range = range(0,100,pct_step)
    power_dgt, power_binned = percentile_spectrogram(np.abs(sc.spg[chan,:,:]**2), sc.f_axis, rank_freqs, pct_range);
    numlines = power_binned.T[1:].shape[1]
    freq_vals = [sc.f_axis[1:]]*numlines
    power = list(zip(*power_binned.T[1:].tolist()))
    
    # color it accordingly
    palette = viridis(numlines)
    
    # filling in data
    DEFAULT_TICKERS = sc.chan_labels
    
    source = ColumnDataSource(data=dict(freq_vals=freq_vals, 
                                        power=power,
                                        color=palette))

    # set up plot
    pct_spct_plot = figure(title='Percentile Spectrogram', x_axis_type='log', y_axis_type='log', 
                           plot_width=plot_side_length, plot_height=plot_side_length)
    pct_spct_plot.legend.location = 'top_left'
    
    # plotting the box
    bg_box = BoxAnnotation(left=rank_freqs[0], right=rank_freqs[1], fill_color='grey', fill_alpha=0.2, line_alpha=0)
    pct_spct_plot.add_layout(bg_box)
    
    # plotting the multi line
    pct_spct_plot.multi_line(xs='freq_vals', ys='power', color='color', source=source)
    
    # create interact
    def update_pct_spct(channel=DEFAULT_TICKERS[0], rank_freqs=(8,12), pct_step=25):   
        """ Update percentile spectrogram via interact ipywidget
            Input
            channel : int
            rank_freqs : tuple (frequency range)
                The first frequency must be lower or equal to the second value.
            pct_step : int
                Percentile step size
        """
        # updating information based on sliders
        channel = int(channel.split('_')[1])
        plot_chan = int(channel)
        pct_range = range(0,100,pct_step)
        power_dgt, power_binned = percentile_spectrogram(np.abs(sc.spg[plot_chan,:,:])**2, sc.f_axis, rank_freqs, pct_range);
        numlines = power_binned.T[1:].shape[1]
        freq_vals = [sc.f_axis[1:]]*numlines
        power = list(zip(*power_binned.T[1:].tolist()))

        palette = viridis(numlines)

        # create a column data source for the plots to share
        updated_source = dict(freq_vals=freq_vals, power=power, color=palette)

        source.data = updated_source

        # update box position
        bg_box.left = rank_freqs[0]
        bg_box.right = rank_freqs[1]

        if show_pct_ex:
            if pct_step == 25 or pct_step == 30:
                power_dgt, power_binned = percentile_spectrogram(np.abs(sc.spg[plot_chan,:,:]**2), 
                                                                 sc.f_axis, rank_freqs, pct_range);
                pwr_ex = power_examples(sc.data[plot_chan,:], sc.fs, sc.t_axis, power_dgt, rank_freqs)
                pwr_ex_source.data = pwr_ex
                push_notebook()

        push_notebook()
        
    if show_pct_ex:
        # power examples from the raw data
        power_dgt, power_binned = percentile_spectrogram(np.abs(sc.spg[chan,:,:]**2), sc.f_axis, rank_freqs, pct_range);
        pwr_ex = power_examples(sc.data[chan,:], sc.fs, sc.t_axis, power_dgt, rank_freqs)
        pwr_ex_source = ColumnDataSource(data=pwr_ex)

        # set up plots
        row_layout = row()

        # looping through each Q
        for ind, j in enumerate(np.unique(power_dgt)):
            plot_side_length = 300
            plot = figure(plot_width=int(plot_side_length*2/3), plot_height=int(plot_side_length/3), name='Q'+str(ind+1))
            plot.legend.location = 'top_left'
            plot.yaxis.ticker = [0]
            plot.grid.grid_line_alpha=0.3
            plot.line('t_plot', 'raw_y'+str(ind+1), source=pwr_ex_source, color=palette[ind])
            plot.line('t_plot', 'filt_y'+str(ind+1), source=pwr_ex_source, line_alpha=0.5, color=palette[ind])
            plot.add_layout(Title(text='Q'+str(ind+1), align='center'), 'above')
            row_layout.children.append(plot)

        show(column(pct_spct_plot, row_layout), notebook_handle=True)
    else:
        show(pct_spct_plot, notebook_handle=True)

        
    # defining the widge we are using
    widget = interactive(update_pct_spct, channel=DEFAULT_TICKERS, 
                                     rank_freqs=widgets.IntRangeSlider(
                                                value=[8, 12],
                                                min=1,
                                                max=199,
                                                step=1,
                                                description='rank_freqs',
                                                disabled=False,
                                                continuous_update=False,
                                                orientation='horizontal',
                                                readout=True,
                                                readout_format='d',
                                            ), pct_step=(5,100,5))
    items = [kid for kid in widget.children]
    display(HBox(children=items))

def plot_vis(sc, chan=0, select_freq=10, select_bin=20, plot_side_length=310, plot_complex=False):
    """ Plot PSD, SCV, KS-Pval, and complex spectrogram if desired
        Input
        sc : SCA object
            spectral component analysis object
        chan : int
            channel selected for plotting
        select_freq : int
            frequency selected for plotting
        select_bin : int
            bin selected for plotting
        plot_side_length : int
            form three (or four) square plots in row layouts
        plot_complex : boolean
            if True, plots the complex spectrogram
    """
    freq_vals = sc.f_axis[1:]
    psd_vals = sc.psd[chan].T[1:]
    scv_vals = sc.scv[chan].T[1:]
    real_vals = sc.spg.real[chan][select_freq]
    imag_vals = sc.spg.imag[chan][select_freq]

    source = ColumnDataSource(data=dict(freq_vals=freq_vals, 
                                        psd_vals=psd_vals, 
                                        scv_vals=scv_vals))

    complex_source = ColumnDataSource(data=dict(real_vals=real_vals, imag_vals=imag_vals))
    DEFAULT_TICKERS = sc.chan_labels
    
    # create interact
    def update_spct(channel=DEFAULT_TICKERS[0], f=10, numbins=20): 
        channel = int(channel.split('_')[1])
        plot_chan = int(channel)
        plot_freq = np.where(sc.f_axis==f)[0][0]
        y, x = np.histogram(abs(sc.spg[plot_chan,plot_freq,:])**2, bins=numbins, density=True)

        # create a column data source for the plots to share
        data_source = {
                       'freq_vals': sc.f_axis[1:],
                       'psd_vals': sc.psd[plot_chan].T[1:],
                       'scv_vals': sc.scv[plot_chan].T[1:]
                       }   
        source.data = data_source
        vline_psd.location = f
        vline_scv.location = f
        # update histogram with data from frequency f
        hist_plot.data_source.data['left'] = x[:-1]
        hist_plot.data_source.data['right'] = x[1:]
        hist_plot.data_source.data['top'] = y    
        # update fitted
        rv = expon(scale=sp.stats.expon.fit(abs(sc.spg[plot_chan,plot_freq,:])**2,floc=0)[1])
        fit_plot.data_source.data['x'] = x
        fit_plot.data_source.data['y'] = rv.pdf(x)

        # update complex data
        complex_source.data['real_vals'] = sc.spg.real[plot_chan][f]
        complex_source.data['imag_vals'] = sc.spg.imag[plot_chan][f]

        hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(f, sc.ks_pvals[int(plot_chan), f])
        push_notebook()
    
    # set up histogram
    y, x = np.histogram(abs(sc.spg**2)[0,10,:], bins=20, density=True)
    hist_fig = figure(plot_height=plot_side_length, plot_width=plot_side_length, x_axis_label='Power', y_axis_label='Probability')
    hist_plot = hist_fig.quad(top=y, bottom=0, left=x[:-1], right=x[1:], fill_color='purple', line_color="lightblue")
    hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(10, sc.ks_pvals[int(1), 10])
    hist_fig.axis.major_label_text_font_size= '0pt'
    fit_plot = hist_fig.line(x[:-1],y, line_width=8,alpha=0.7,line_color="#D53B54",legend='Fit PDF')

    # set up psd plot
    psd_plot = figure(title='PSD', x_axis_type='log', y_axis_type='log', plot_width=plot_side_length, plot_height=plot_side_length)
    psd_plot.legend.location = 'top_left'
    psd_plot.xaxis.axis_label = 'Frequency (Hz)'
    psd_plot.yaxis.axis_label = 'Power/Frequency (dB/Hz)'
    psd_plot.grid.grid_line_alpha=0.3
    psd_plot.line('freq_vals', 'psd_vals', source=source)    

    # set up scv plot
    scv_plot = figure(title='SCV', x_axis_type='log', y_axis_type='log', plot_width=plot_side_length, plot_height=plot_side_length)
    scv_plot.legend.location='top_left'
    scv_plot.xaxis.axis_label = 'Frequency (Hz)'
    scv_plot.yaxis.axis_label = '(Unitless)'
    scv_plot.grid.grid_line_alpha=0.3
    fit_line = bokeh.models.glyphs.Line(x='freq_vals', y=1, line_width=5, line_alpha=0.5, line_color='darkgrey')
    scv_plot.add_glyph(source, fit_line)
    scv_plot.line('freq_vals', 'scv_vals', source=source, color='navy')

    # add in frequency slider vertical lines
    vline_psd = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
    vline_scv = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
    psd_plot.add_layout(vline_psd)
    scv_plot.add_layout(vline_scv)

    if plot_complex:
        # set up complex plot
        complex_plot = figure(title='Complex SPG', plot_width=plot_side_length, plot_height=plot_side_length)
        complex_plot.legend.location = 'top_left'
        complex_plot.xaxis.axis_label = 'Real values'
        complex_plot.yaxis.axis_label = 'Imaginary values'
        basic_xline = bokeh.models.glyphs.Line(x='real_vals', y=0, line_width=5, line_alpha=0.5, line_color='darkgrey')
        complex_plot.add_glyph(complex_source, basic_xline)
        basic_yline = bokeh.models.glyphs.Line(x=0, y='imag_vals', line_width=5, line_alpha=0.5, line_color='darkgrey')
        complex_plot.add_glyph(complex_source, basic_yline)
        complex_plot.grid.grid_line_alpha=0.3
        complex_plot.circle('real_vals', 'imag_vals', size=5, color="purple", alpha=0.5, source=complex_source)
        # set up layout and interact tools
        layout = column(row(psd_plot, scv_plot), row(hist_fig, complex_plot))
        show(layout, notebook_handle=True)
    else:
        # set up layout and interact tools
        layout = row(psd_plot, scv_plot, hist_fig)
        show(layout, notebook_handle=True)

    warnings.filterwarnings('ignore')
    widget = interactive(update_spct, channel=DEFAULT_TICKERS, f=(1,199), numbins=(10,55,5))
    items = [kid for kid in widget.children]
    display(HBox(children=items))
 
