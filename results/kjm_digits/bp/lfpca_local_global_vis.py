import sys
sys.path.append('../../../../spectralCV')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

import neurodsp as ndsp
from scv_funcs import lfpca
from scv_funcs import utils

CKEYS = plt.rcParams['axes.prop_cycle'].by_key()['color']
font = {'family' : 'arial',
        'weight' : 'regular',
        'size'   : 15}

import matplotlib
matplotlib.rc('font', **font)

# bokeh imports
from ipywidgets import interact
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from scipy.stats import expon
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, TapTool
from bokeh.models.widgets import PreText, Select, Slider
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Slider, Span, CustomJS
from bokeh.models import Range1d
from bokeh.events import Tap 
import bokeh

from bokeh.models import ColumnDataSource, OpenURL, TapTool
from bokeh.plotting import figure, output_file, show

output_file("test_sample.html")

# obtain data information
trial_data = np.load('trial_info.npz')
trial_info = trial_data['trial_info']
elec_regions = trial_data['elec_regions']
conditions = ['pre', 'move', 'whole']
lfpca_all = []
for cond in conditions:
    lfpca_all.append(lfpca.lfpca_load_spec(cond+'.npz'))
    
lf_pre = lfpca_all[0]
lf_move = lfpca_all[1]
lf_whole = lfpca_all[2]
lfpca_all = {'lf_pre': lfpca_all[0], 'lf_move': lfpca_all[1], 'lf_whole': lfpca_all[2]}

# using lf_pre as an example
lf = lfpca_all['lf_pre']

# grabbing channel count from psd
chan_count, freq = lf.psd.shape

# mapping all the channels
DEFAULT_TICKERS = list(map(str, range(chan_count)))
LF_TICKERS = [key for key in lfpca_all.keys()]

# initializing values for frequency, psd, scv, histogram plot
chan = 0
select_freq = 10
select_bin = 20
freq_vals = lf.f_axis[1:]
psd_vals = lf.psd[chan].T[1:]
scv_vals = lf.scv[chan].T[1:]

# creating a selector and slider
lf_ticker = Select(value='lf_pre', title='lf_condition', options=LF_TICKERS)
ticker = Select(value=str(chan), title='channel', options=DEFAULT_TICKERS)
freq_slider = Slider(start=1, end=199, value=select_freq, step=1, title="Frequency", callback_policy="mouseup")
bin_slider = Slider(start=10, end=55, value=select_bin, step=5, title="Number of bins", callback_policy="mouseup")

# create data and selection tools
source = ColumnDataSource(data=dict(freq_vals=freq_vals, psd_vals=psd_vals, scv_vals=scv_vals))

TOOLS = "tap, help"

# setting up plots
psd_plot = figure(tools=TOOLS, title='PSD', x_axis_type='log', y_axis_type='log')
psd_plot.legend.location = 'top_left'
psd_plot.xaxis.axis_label = 'Frequency (Hz)'
psd_plot.yaxis.axis_label = 'Power/Frequency (dB/Hz)'
psd_plot.grid.grid_line_alpha=0.3

scv_plot = figure(tools=TOOLS, title='SCV', x_axis_type='log', y_axis_type='log')
scv_plot.legend.location='top_left'
scv_plot.xaxis.axis_label = 'Frequency (Hz)'
scv_plot.yaxis.axis_label = '(Unitless)'
scv_plot.grid.grid_line_alpha=0.3

# create histogram frame
hist_source = ColumnDataSource({'top': [], 'left': [], 'right': []})
fit_hist_source = ColumnDataSource({'x': [], 'y': []})
hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}

# create fit line for the histogram
rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}

hist_fig = figure(x_axis_label='Power', 
                  y_axis_label='Probability', background_fill_color="#E8DDCB")
hist_fig.axis.visible = False
hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])


# customize plot to psd
def create_psd_plot(psd_plot, source):
    psd_plot.line('freq_vals', 'psd_vals', source=source, color='navy')
    psd_plot.circle('freq_vals', 'psd_vals', source=source, size=8, color='darkgrey', alpha=0.2, 
                    # set visual properties for selected glyphs
                    selection_color="firebrick",
                    # set visual properties for non-selected glyphs
                    nonselection_fill_alpha=0.2,
                    nonselection_fill_color="darkgrey")
    
# customize plot to psd
def create_scv_plot(scv_plot, source):
    scv_plot.line('freq_vals', 'scv_vals', source=source, color='navy')
    scv_plot.circle('freq_vals', 'scv_vals', source=source, size=8, color='darkgrey', alpha=0.2, 
                    # set visual properties for selected glyphs
                    selection_color="firebrick",
                    # set visual properties for non-selected glyphs
                    nonselection_fill_alpha=0.2,
                    nonselection_fill_color="darkgrey")

# customize histogram
def create_hist(hist_fig, hist_source):
    hist_fig.quad(top='top', bottom=0, left='left', right='right', fill_color="#036564", line_color="#033649", source=hist_source)

# initializing plots
create_psd_plot(psd_plot, source)
create_scv_plot(scv_plot, source)
vline_psd = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
vline_scv = Span(location=select_freq, dimension='height', line_color='red', line_dash='dashed', line_width=3)
psd_plot.add_layout(vline_psd)
scv_plot.add_layout(vline_scv)
create_hist(hist_fig, hist_source)

fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
hist_fig.add_glyph(fit_hist_source, fit_line)

all_plots = gridplot([[psd_plot, scv_plot, hist_fig]], plot_width=400, plot_height=400)

# choosing a different lfpca object
def lf_selection_change(attr, old, new):
    lf = lfpca_all[lf_ticker.value]
    psd_vals = lf.psd[chan].T[1:]
    scv_vals = lf.scv[chan].T[1:]
    data = dict(freq_vals=freq_vals, psd_vals=psd_vals, scv_vals=scv_vals)
    # create a column data source for the plots to share
    source.data = data
    create_psd_plot(psd_plot=psd_plot, source=source)
    create_scv_plot(scv_plot=scv_plot, source=source)
    select_freq = freq_slider.value
    select_bin = bin_slider.value
    hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
    rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
    hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
    fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}
    hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])
    create_hist(hist_fig=hist_fig, hist_source=hist_source)
    fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
    hist_fig.add_glyph(fit_hist_source, fit_line)
    
# what to do when channel selection changes
def selection_change(attr, old, new):
    chan = int(ticker.value)
    psd_vals = lf.psd[chan].T[1:]
    scv_vals = lf.scv[chan].T[1:]
    data = dict(freq_vals=freq_vals, psd_vals=psd_vals, scv_vals=scv_vals)
    # create a column data source for the plots to share
    source.data = data
    # update histogram and fit line
    select_freq = freq_slider.value
    select_bin = bin_slider.value
    hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
    rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
    hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
    fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}
    create_psd_plot(psd_plot=psd_plot, source=source)
    create_scv_plot(scv_plot=scv_plot, source=source)
    hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])
    create_hist(hist_fig=hist_fig, hist_source=hist_source)
    fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
    hist_fig.add_glyph(fit_hist_source, fit_line)
    return chan

# set up connector spans
freq_slider.callback = CustomJS(args=dict(span1 = vline_psd,
                                          span2 = vline_scv,
                                          slider = freq_slider),
                                          code = """span1.location = slider.value; 
                                                    span2.location = slider.value""")

taptool_psd = psd_plot.select(type=TapTool)
taptool_scv = scv_plot.select(type=TapTool)

def update_hist(attrname, old, new):
    # get the current slider values
    select_bin = bin_slider.value
    select_freq = freq_slider.value
    
    # update histogram and fit line
    hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
    rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
    hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
    fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}
    hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])
    create_hist(hist_fig=hist_fig, hist_source=hist_source)
    fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
    hist_fig.add_glyph(fit_hist_source, fit_line)

# taptool = TapTool()

# def callback(event):
#     select_freq = event.x
#     # update histogram and fit line
#     hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
#     rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
#     hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
#     fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}
#     hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])
#     create_hist(hist_fig=hist_fig, hist_source=hist_source)
#     fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
#     hist_fig.add_glyph(fit_hist_source, fit_line)

# taptool.on_event(Tap, callback)

for widget in [bin_slider, freq_slider]:
    widget.on_change('value', update_hist)

# when selected value changes, take the following methods of actions
lf_ticker.on_change('value', lf_selection_change)
ticker.on_change('value', selection_change)

# what to do when freq slider value changes
def freq_change(attr, old, new):
    select_freq = source.selected.indices[0]
    # update histogram and fit line
    hist, edges = np.histogram(lf.spg[chan, select_freq, :], bins=select_bin, density=True)
    rv = expon(scale=sp.stats.expon.fit(lf.spg[chan,select_freq,:],floc=0)[1])
    hist_source.data = {'top': hist, 'left': edges[:-1], 'right': edges[1:]}
    fit_hist_source.data = {'x': edges, 'y': rv.pdf(edges)}
    hist_fig.title.text = 'Freq = %.1fHz, p-value = %.4f'%(select_freq, lf.ks_pvals[chan, select_freq])
    create_hist(hist_fig=hist_fig, hist_source=hist_source)
    fit_line = bokeh.models.glyphs.Line(x='x', y='y', line_width=8, line_alpha=0.7, line_color="#D95B43")
    hist_fig.add_glyph(fit_hist_source, fit_line)

renderer = psd_plot.patches('freq_vals', 'psd_vals', source=source)
renderer.data_source.on_change("selected", freq_change)

# taptool_psd.on_change('value', selection_change)
# taptool_scv = scv_plot.select(type=TapTool)
# taptool_scv.callback = OpenURL(url=url)
# freq_slider.on_change('value', freq_slider_change)
# bin_slider.on_change('value', bin_slider_change)

# organize layout
widgets = row(lf_ticker, ticker, freq_slider, bin_slider)
main_control = row(widgets)
layout = column(main_control, all_plots)

show(layout)
curdoc().add_root(layout)