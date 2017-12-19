import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

import StringIO
import scipy.stats
import time
import sys
import dill as pickle



def draw_png_heatmap_graph(obs, preds, gt, mixes,plt_size, draw_prediction_track, plot_directory, log_file_name,
                           multi_sample, global_step, graph_number):
    legend_str = []
    fig = plt.figure(figsize=plt_size)
    plt.plot(gt[:, 0], gt[:, 1], 'b-', zorder=3)
    legend_str.append(['Ground Truth'])
    plt.plot(obs[:, 0], obs[:, 1], 'g-', zorder=4)
    legend_str.append(['Observations'])
    if draw_prediction_track:
        for j in range(preds.shape[0]):
            plt.plot(preds[j][:, 0], preds[j][:, 1], 'r-', zorder=5)
        legend_str.append(['Predictions'])

    dx, dy = 0.1, 0.1
    x = np.arange(-35, 10, dx)
    y = np.flip(np.arange(-30, 15, dy), axis=0)  # Image Y axes are down positive, map axes are up positive.
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
                gaus_num += 1
                #print gaus_num
                pi, mu1, mu2, s1, s2, rho = gaussian
                cov = np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])
                norm = scipy.stats.multivariate_normal(mean=(mu1, mu2), cov=cov)
                zz = norm.pdf(xxyy)
                zz *= pi
                zz = zz.reshape((len(xx), len(yy)))
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
    image_filename = 'leith-croydon.png'
    background_img = plt.imread(os.path.join('images', image_filename))
    plt.imshow(background_img, zorder=0,
               extent=[-15.275 - (147.45 / 2), -15.275 + (147.45 / 2), -3.1 - (77 / 2), -3.1 + (77 / 2)])
    plt.imshow(final_heatmap, cmap=plt.cm.viridis, alpha=.7, interpolation='bilinear', extent=extent, zorder=1)
    fig_path = os.path.join(plot_directory + "_img",
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

if __name__ == "__main__":
    #read args from stdin

    data = pickle.loads(sys.stdin.read())
    fig_return_data = draw_png_heatmap_graph(data['obs'], data['preds'], data['gt'], data['mixes'], data['plt_size'],
                                      data['draw_prediction_track'], data['plot_directory'], data['log_file_name'],
                                      data['multi_sample'], data['global_step'], data['graph_number'])


