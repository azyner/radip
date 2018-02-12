# Read pkl file
# make directory
# for each track
# plot full track
# naming scheme is csv_name, track_idx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameters
import ibeoCSVImporter
import os
import dill as pickle

short_wrangle = False
if short_wrangle:
    range_max = 1
else:
    range_max = 999

if 'leith-croydon' in parameters.parameters['data_list']:
    source_list = ['split_20170601-stationary-3-leith-croydon_01.csv']
    if not short_wrangle:
        source_list.extend([
                   'split_20170601-stationary-3-leith-croydon_02.csv',
                   'split_20170601-stationary-3-leith-croydon_03.csv',
                   'split_20170601-stationary-3-leith-croydon_04.csv',
                   'split_20170601-stationary-3-leith-croydon_05.csv'
        ])
    for i in range(min(range_max, 41)):
        source_list.append("split_20170802-stationary-4-leith-croydon_%02d.csv" % (i+1))
    for i in range(min(range_max, 35)):
        source_list.append("split_20170804-stationary-5-leith-croydon_%02d.csv" % (i+1))
if 'queen-hanks' in parameters.parameters['data_list']:
    for i in range(min(range_max, 31)):
        source_list.append("split_20180116-082129-urban-stationary-queen-hanks_%02d.csv" % (i+1))
if 'roslyn-crieff' in parameters.parameters['data_list']:
    for i in range(min(range_max, 24)):
        source_list.append("split_20180119-112135-urban-stationary-roslyn-crieff_%02d.csv" % (i + 1))
if 'oliver-wyndora' in parameters.parameters['data_list']:
    for i in range(min(range_max, 46)):
        source_list.append("split_20180123-072840-urban-stationary-oliver-wyndora_%02d.csv" % (i + 1))
if 'orchard-mitchell' in parameters.parameters['data_list']:
    for i in range(min(range_max, 21)):
        source_list.append("split_20180124-081438-urban-stationary-orchard-mitchell_%02d.csv" % (i + 1))
sourcename = source_list[0]


plot_directory = 'all_tracks_data_plot'
ibeoCSV = ibeoCSVImporter.ibeoCSVImporter(parameters,source_list)

# Start plotting
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

track_count = 1

for track in ibeoCSV.get_track_list():
    print "Plotting track: " + str(track_count)
    track_count += 1
    legend_str = []
    x_range = (-20, 20)
    y_range = (-30, 30)
    fig = plt.figure(figsize=(10, 10))
    plt.ylim(*y_range)
    plt.xlim(*x_range)  # , xlim=x_range, ylim=y_range)
    plt.plot(track.relative_x, track.relative_y, 'b-')
    # plot start point
    plt.plot(track.relative_x.iloc[0], track.relative_y.iloc[0], 'rx', ms=10)

    legend_str.append([track.origin.iloc[0] + " " + track.relative_destination.iloc[0] + ' ' + track.csv_name.iloc[0]])
    plt.legend(legend_str)
    fig_path = os.path.join(plot_directory, str(track.uniqueId.iloc[0]) + '.png') #, ###### + '.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()