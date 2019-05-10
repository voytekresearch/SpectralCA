# imports
import numpy as np
import scipy as sp
from scipy.stats import expon
import glob

import neurodsp as ndsp
import warnings
warnings.filterwarnings('ignore')

# bokeh imports
import bokeh
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Span
from bokeh.models.widgets import Select, Slider
from ipywidgets import Layout, HBox, interactive

output_notebook()

def plot_vis(sc, chan=0, select_freq=10, select_bin=20):
    
    freq_vals = sc.f_axis[1:]
    psd_vals = sc.psd[chan].T[1:]
    scv_vals = sc.scv[chan].T[1:]
    real_vals = sc.spg.real[chan][select_freq]
    imag_vals = sc.spg.imag[chan][select_freq]

    source = ColumnDataSource(data=dict(freq_vals=freq_vals, 
                                        psd_vals=psd_vals, 
                                        scv_vals=scv_vals))

    complex_source = ColumnDataSource(data=dict(real_vals=real_vals, imag_vals=imag_vals))
    # grabbing channel count from psd
    chan_count, freq = sc.psd.shape
    DEFAULT_TICKERS = list(map(str, range(chan_count)))
    
    # create interact
    def update_spct(channel=1, f=10, numbins=20):    
        plot_chan = int(channel)
        plot_freq = np.where(sc.f_axis==f)[0][0]
        y, x = np.histogram(abs(sc.spg[plot_chan,plot_freq,:])**2, bins=numbins, density=True)

        # create a column data source for the plots to share
        data_source = {
                       'freq_vals': sc.f_axis[1:],
                       'psd_vals': sc.psd[channel].T[1:],
                       'scv_vals': sc.scv[channel].T[1:]
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
        complex_source.data['real_vals'] = sc.spg.real[channel][f]
        complex_source.data['imag_vals'] = sc.spg.imag[channel][f]

        hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(f, sc.ks_pvals[int(channel), f])
        push_notebook()

    plot_side_length = 310
    
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


    # add in frequency slider vertical lines
    vline_psd = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
    vline_scv = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
    psd_plot.add_layout(vline_psd)
    scv_plot.add_layout(vline_scv)

    # set up layout and interact toola
    layout = column(row(psd_plot, scv_plot), row(hist_fig, complex_plot))
    show(layout, notebook_handle=True)

    warnings.filterwarnings('ignore')
    widget = interactive(update_spct, channel=range(1,len(DEFAULT_TICKERS)-1), f=(1,199), numbins=(10,55,5))
    items = [kid for kid in widget.children]
    display(HBox(children=items))
