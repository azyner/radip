#!/usr/bin/env python
import NetworkManager
import os
import pandas as pd
import dill as pickle
import argparse
from bokeh.plotting import figure, output_file, show, gridplot, save
from bokeh.layouts import widgetbox
from bokeh.layouts import layout

import numpy as np

# Args is only the path to directory with all pkl files. It will also be the plot dir.
parser = argparse.ArgumentParser(description='Collate all results to make one graph.')
parser.add_argument('path', help='Path to folder with pkl files in it.')
args = parser.parse_args()

print args.path

data_dict = {}

for dirpath, dirnames, filenames in os.walk(args.path):
    for filename in filenames:
        if filename.endswith('.pkl'):
            with open(os.path.join(args.path,filename),'rb') as file:
                from_pkl = pickle.load(file)
                params = from_pkl['parameters']
                results_per_dis = from_pkl['results_per_dis']
                dict_key = 'Steps ' + str(params['observation_steps']) + " Learning Rate: " + str(params['learning_rate'])
                data_dict[dict_key] = results_per_dis

plt_path = os.path.join(args.path, 'plot.html')
# If I am running this many times, make new filenames
if os.path.exists(plt_path):
    path_idx = 1
    while os.path.exists(plt_path):
        plt_path = os.path.join(args.path, "plot-%02d" % path_idx + '.html')
        path_idx += 1

output_file(plt_path)
first_key,first_results = list(data_dict.iteritems())[0]
plot_titles = np.sort(first_results['origin'].unique())
plots = []

for origin in plot_titles:
    print "plotting: " + origin
    plt_title = 'Accuracy as measured relative to 20m mark. Averaged over all tracks'
    p1 = figure(title='Origin: ' + origin, x_axis_label='Dis from Ref Line (m)', y_axis_label='Acc.',
                plot_width=500, plot_height=500, x_range=(-12, 35), y_range=(0, 1.05), )  # ~half a 1080p screen

    for key_str, graph_results in data_dict.iteritems():

        # plot 1
        dataset = graph_results[graph_results['origin'] == origin]
        x_data = []
        y_data = []
        tp_data = []
        fp_data = []
        fn_data = []
        for range_val in np.unique(dataset['d_thresh']):
            # If I group by track number here, I can get a collection of accuracy scores
            # and therefore a std dev
            data_at_range = dataset[dataset['d_thresh'] == range_val]
            acc = np.average(np.equal(data_at_range['output_idxs'],
                                      data_at_range['destination_vec']))
            x_data.append(range_val)
            y_data.append(acc)

        p1.line(x_data, y_data, legend=key_str, line_width=2)

    p1.legend.location = "bottom_right"
    plots.append(p1)

l = layout([plots])
save(l)