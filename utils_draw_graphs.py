import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import signal
import StringIO
import scipy.stats
import time
import sys
import dill as pickle


def draw_png_heatmap_graph(obs, preds_dict, gt, mixes, padding_logits, trackwise_padding, plt_size, draw_prediction_track, plot_directory, log_file_name,
                           multi_sample, global_step, graph_number, fig_dir, csv_name, parameters):
    ##FIXME
    padding_bool = trackwise_padding
    #
    #padding_bool = np.argmax(padding_logits, axis=1) == 1
    # 'results/20180412-104825/plots_img_final'
    legend_str = []
    fig = plt.figure(figsize=plt_size)
    plt.plot(gt[:, 0], gt[:, 1], 'b-', zorder=3, label="Ground Truth")
    legend_str.append(['Ground Truth'])
    plt.plot(obs[:, 0], obs[:, 1], 'g-', zorder=4, label="observations")
    legend_str.append(['Observations'])

    plot_colors = ['r', 'c', 'm', 'y', 'k']
    plot_colors_idx = 0

    for name, preds in preds_dict.iteritems():
        # The input is designed for multiple future tracks. If only 1 is produced, the axis is missing. So reproduce it.
        # This is the most common case (one track)
        if len(preds.shape) < 3:
            preds = np.array([preds])

        if name == 'RNN' and not draw_prediction_track:
            continue
        else:
            for j in range(preds.shape[0]):
                # `Real data'
                plt.plot(preds[j][~padding_bool, 0], preds[j][~padding_bool, 1],
                         plot_colors[plot_colors_idx] + 'o', ms=2, zorder=5, label=name + ' Pred')
                plt.plot(preds[j][~padding_bool, 0], preds[j][~padding_bool, 1],
                         plot_colors[plot_colors_idx] + '-', ms=1, zorder=5)
                # Padding `fake' data
                plt.plot(preds[j][padding_bool, 0], preds[j][padding_bool, 1],
                         plot_colors[plot_colors_idx] + 'x', ms=2, zorder=5)
                plot_colors_idx += 1
            #legend_str.append([name + ' Pred'])
    plt.legend()

    if 'relative' in parameters['ibeo_data_columns'][0]:
        x_range = (-20, 20)
        y_range = (-10, 30)
    elif 'queen-hanks' in csv_name:
        x_range = (3, 47)
        y_range = (-17, 11)
    elif 'leith-croydon' in csv_name:
        x_range = (-35, 10)
        y_range = (-30, 15)
    elif 'roslyn-crieff' in csv_name:
        x_range = (-31, -10)
        y_range = (-15, 8)
    elif 'oliver-wyndora' in csv_name:
        x_range = (-28, -8)
        y_range = (-12, 6)
    elif 'orchard-mitchell' in csv_name:
        x_range = (-32, -5)
        y_range = (-23, 5)

    dx, dy = 0.1, 0.1
    x = np.arange(min(x_range), max(x_range), dx)
    y = np.flip(np.arange(min(y_range), max(y_range), dy), axis=0)  # Image Y axes are down positive, map axes are up positive.
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    extent = np.min(x), np.max(x), np.min(y), np.max(y)

    # Return probability sum here.
    heatmaps = None
    plot_time = time.time()

    for sampled_mix in mixes:
        # Sleep in process to improve niceness.
        time.sleep(0.05)
        #print "len sampled_mix: " + str(len(sampled_mix))
        sample_time = time.time()
        timeslot_num = 0
        for timeslot in sampled_mix:
            timeslot_num += 1
            #print "timeslot_num " + str(timeslot_num)
            gaussian_heatmaps = []
            gaus_num = 0
            for gaussian in timeslot:
                ##FIXME does not check padding_logit
                gaus_num += 1
                #print gaus_num
                pi, mu1, mu2, s1, s2, rho = gaussian
                cov = np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])
                norm = scipy.stats.multivariate_normal(mean=(mu1, mu2), cov=cov)
                zz = norm.pdf(xxyy)
                zz *= pi
                zz = zz.reshape((len(xx), len(yy[0])))
                gaussian_heatmaps.append(zz)
            gaussian_heatmaps /= np.max(gaussian_heatmaps)  # Normalize such that each timestep has equal weight
            #heatmaps.extend(gaussian_heatmaps) #  This explodes
            if heatmaps is None:
                heatmaps = gaussian_heatmaps
            else:
                heatmaps += gaussian_heatmaps
            #print "Time for this sample: " + str(time.time() - sample_time)
    #print "Time for gaussian plot of one track: " + str(time.time() - plot_time)
    # Its about 7 seconds per plot

    final_heatmap = sum(heatmaps)
    if 'relative' in parameters['ibeo_data_columns'][0]:
        _ = 0  # Blank line to preserve lower logic flow
    elif 'queen-hanks' in csv_name:
        x_range = (3, 47)
        y_range = (-17, 11)
    elif 'leith-croydon' in csv_name:
        x_range = (-35, 10)
        y_range = (-30, 15)
    elif 'leith-croydon' in csv_name:
        image_filename = 'leith-croydon.png'
        background_img = plt.imread(os.path.join('images', image_filename))
        plt.imshow(background_img, zorder=0,
                   extent=[-15.275 - (147.45 / 2), -15.275 + (147.45 / 2), -3.1 - (77 / 2), -3.1 + (77 / 2)])
    plt.imshow(final_heatmap, cmap=plt.cm.viridis, alpha=.7, interpolation='bilinear', extent=extent, zorder=1)
    plt.legend()
    fig_path = os.path.join(fig_dir,
                            ("no_pred_track-" if draw_prediction_track is False else "")
                            + str(multi_sample) + "-" + log_file_name + '-' +
                            str(global_step) + '-' + str(graph_number) + '.png')
    plt.savefig(fig_path, bbox_inches='tight')
    # Now inject into tensorboard
    fig.canvas.draw()
    fig_s = fig.canvas.tostring_rgb()
    fig_data = np.fromstring(fig_s, np.uint8)
    fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # This last string return allows images to be saved in tensorboard. I don't use it anymore, and I want the threads
    # to run in the background, so I dropped the return value.
    s = StringIO.StringIO()
    plt.imsave(s, fig_data, format='png')
    fig_data = s.getvalue()
    plt.close()
    return fig_data

def multiprocess_helper(args):
    return draw_png_heatmap_graph(*args)

if __name__ == "__main__":
    #read args from stdin
    def sigint_handler(signum, frame):
        nothing = None  # Do nothing.

    signal.signal(signal.SIGINT, sigint_handler)

    data = pickle.loads(sys.stdin.read())
    fig_return_data = draw_png_heatmap_graph(data['obs'], data['preds'], data['gt'], data['mixes'], data['padding_logits'],
                                             data['trackwise_padding'],
                                             data['plt_size'],
                                      data['draw_prediction_track'], data['plot_directory'], data['log_file_name'],
                                      data['multi_sample'], data['global_step'], data['graph_number'], data['fig_dir'],
                                             data['csv_name'], data['parameters'])


