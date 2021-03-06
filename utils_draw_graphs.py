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


def draw_png_heatmap_graph(obs, preds_dict, gt, mixes, network_padding_logits, trackwise_padding, plt_size, draw_prediction_track, plot_directory, log_file_name,
                           multi_sample, global_step, graph_number, fig_dir, csv_name, rel_destination, parameters, padding_mask='None', distance=0):

    """

    :param obs:
    :param preds_dict:
    :param gt:
    :param mixes:
    :param network_padding_logits:
    :param trackwise_padding:
    :param plt_size:
    :param draw_prediction_track:
    :param plot_directory:
    :param log_file_name:
    :param multi_sample:
    :param global_step:
    :param graph_number:
    :param fig_dir:
    :param csv_name:
    :param rel_destination:
    :param parameters:
    :param padding_mask: "None", "GT" or "Network"
    :return:
    """
    ##FIXME
    gt_padding_bool = trackwise_padding
    #
    #padding_bool = np.argmax(padding_logits, axis=1) == 1
    # 'results/20180412-104825/plots_img_final'
    legend_str = []
    fig = plt.figure(figsize=plt_size)
    plt.plot(gt[:, 0], gt[:, 1], 'b-', zorder=3, label="Ground Truth")
    plt.plot(gt[:, 0], gt[:, 1], 'bo', zorder=3, ms=2)
    legend_str.append(['Ground Truth'])
    plt.plot(obs[:, 0], obs[:, 1], 'g-', zorder=4, label="Observations")
    plt.plot(obs[:, 0], obs[:, 1], 'go', zorder=4, ms=2)
    legend_str.append(['Observations'])

    plot_colors = ['r', 'c', 'm', 'y', 'k']
    plot_colors_idx = 0
    first_RNN = True

    for name, preds in preds_dict.iteritems():
        # The input is designed for multiple future tracks. If only 1 is produced, the axis is missing. So reproduce it.
        # This is the most common case (one track)
        if len(preds.shape) < 3:
            preds = np.array([preds])

        if name == 'RNN' and not draw_prediction_track:
            continue
        else:
            for j in range(preds.shape[0]):
                prediction = preds[j]
                # `Real data'
                if 'multipath' in name:
                    plot_color = 'w'
                    if first_RNN:
                        first_RNN = False
                        label_name = "RNN Proposed"
                    else:
                        label_name = None
                else:
                    plot_color = plot_colors[plot_colors_idx]
                    plot_colors_idx += 1
                    label_name = name
                if len(prediction) is not len(gt_padding_bool):
                    padding_amount = len(gt_padding_bool) - len(prediction)
                    if padding_amount < 0:
                        prediction = prediction[:len(gt_padding_bool), :]
                    else:
                        prediction = np.pad(prediction, [[0, padding_amount], [0, 0]], 'edge')
                plt.plot(prediction[~gt_padding_bool, 0], prediction[~gt_padding_bool, 1],
                         plot_color + 'o', ms=2, zorder=5)
                plt.plot(prediction[~gt_padding_bool, 0], prediction[~gt_padding_bool, 1],
                         plot_color + '-', ms=1, zorder=5, label=label_name)
                # Padding `fake' data
                #plt.plot(prediction[gt_padding_bool, 0], prediction[gt_padding_bool, 1],
                #         plot_color + 'x', ms=2, zorder=5)

            #legend_str.append([name + ' Pred'])
    plt.legend()

    if 'relative' in parameters['ibeo_data_columns'][0]:
        x_range = (-20, 20)
        y_range = (-10, 30)
        x_range = (-18, 18)
        y_range = (-8, 28)
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

    dx, dy = 0.5, 0.5
    x = np.arange(min(x_range), max(x_range), dx)
    y = np.flip(np.arange(min(y_range), max(y_range), dy), axis=0)  # Image Y axes are down positive, map axes are up positive.
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    extent = np.min(x), np.max(x), np.min(y), np.max(y)

    # Return probability sum here.
    heatmaps = None
    plot_time = time.time()

    for sampled_mix, sampled_padding_logits in zip(mixes, network_padding_logits):
        # Sleep in process to improve niceness.
        time.sleep(0.05)
        #print "len sampled_mix: " + str(len(sampled_mix))
        sample_time = time.time()

        network_padding_bools = np.argmax(sampled_padding_logits, axis=1) == 1

        timeslot_num = 0
        for timeslot, n_padded, gt_padded in zip(sampled_mix, network_padding_bools, gt_padding_bool):
            if 'Network' in padding_mask and n_padded:
                continue
            if 'GT' in padding_mask and gt_padded:
                continue
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
            #TODO Does not work!
            save_each_timestep = False
            if save_each_timestep:
                import copy
                timestep_plt = copy.deepcopy(plt)
                timestep_plt.imshow(gaussian_heatmaps, cmap=plt.cm.viridis, alpha=.7, interpolation='bilinear', extent=extent,
                               zorder=1)
                timestep_plt.legend()
                distance_str = ('n' if distance < 0 else 'p') + "%02i" % abs(distance+50)
                fig_name = padding_mask + '-' + str(graph_number) + '-' + distance_str + '-' + ("no_pred_track-" if draw_prediction_track is False else "") + str(
                    multi_sample) + "-" + log_file_name + '-' + str(global_step) + '-' + rel_destination + 't_' + str(timeslot_num) + '.png'
                fig_path = os.path.join(fig_dir, fig_name)
                timestep_plt.savefig(fig_path, bbox_inches='tight')

            if heatmaps is None:
                heatmaps = gaussian_heatmaps
            else:
                heatmaps += gaussian_heatmaps
            timeslot_num += 1
            #print "Time for this sample: " + str(time.time() - sample_time)
    #print "Time for gaussian plot of one track: " + str(time.time() - plot_time)
    # Its about 7 seconds per plot

    final_heatmap = sum(heatmaps) if heatmaps is not None else None
    if 'relative' in parameters['ibeo_data_columns'][0]:
        _ = 0  # Blank line to preserve lower logic flow
        image_filename = 'intersection_diagram_background.png'
        background_img = plt.imread(os.path.join('images', image_filename))
        plt.imshow(background_img, zorder=0,    #     x_range = (-20, 20)      y_range = (-10, 30)
                   extent=extent)#[-20, 20, -10, 30])

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
    if final_heatmap is not None:
        plt.imshow(final_heatmap, cmap=plt.cm.viridis, alpha=.7, interpolation='bilinear', extent=extent, zorder=1)
    plt.legend()
    plt.xlabel("x (metres)")
    plt.ylabel("y (metres)")
    distance_str = ('n' if distance < 0 else 'p') + "%02i" % abs(distance+50)
    fig_name = padding_mask + '-' + str(graph_number) + '-' + distance_str + '-' + ("no_pred_track-" if draw_prediction_track is False else "") + str(
                multi_sample) + "-" + log_file_name + '-' + str(global_step) + '-' + rel_destination + '.png'
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path, bbox_inches='tight')
    print "Finished plotting " + fig_name
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
    return None  # fig_data  # I don't use it, and it fills memory

def multiprocess_helper(args):
    try:
        return draw_png_heatmap_graph(*args)
    except TypeError:
        print "TypeError Caught!"
        print args

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
                                             data['csv_name'], data['relative_destination'], data['parameters'], data['padding_mask'])


