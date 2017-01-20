# Class that handles the network itself. Defines the training / testing state,
#  manages tensorboard handles.

# Step function should go here, that means it needs to be passed a BatchHandler,
# so that it can grab data easily
# This should also have the different test types, such as the accuracy graph
# Should it handle the entirety of crossfolding?
# I don't think so, that should go into another class maybe
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
import os
import numpy as np
import pandas as pd

class NetworkManager:
    def __init__(self, parameters, log_file_name=None):
        self.parameters = parameters
        self.network = None
        self.batchHandler = None
        self.sess = None
        self.device = None
        self.log_file_name = log_file_name
        self.model = None
        self.plot_directory = 'plots'
        self.network_name_string = "temp123456"

        return

    def build_model(self):
        self.device = tf.device('gpu:0')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
        self.model = Seq2SeqModel(self.parameters)
        if not os.path.exists(self.parameters['train_dir']):
            os.makedirs(self.parameters['train_dir'])
        if not os.path.exists(os.path.join(self.parameters['train_dir'], self.log_file_name)):
            os.makedirs(os.path.join(self.parameters['train_dir'], self.log_file_name))
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.parameters['train_dir'], self.log_file_name))
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

        return

    def run_training_step(self, X, Y, weights, train_model, summary_writer=None):
         return self.model.step(self.sess, X, Y, weights, train_model, summary_writer=summary_writer)

    def draw_graphs(self,graph_results):
        if True:  # Plot HTML bokeh
            from bokeh.plotting import figure, output_file, show, gridplot
            from bokeh.models.widgets import Button
            from bokeh.layouts import widgetbox
            from bokeh.layouts import layout
            plot_titles = graph_results['destination'].unique()
            plots = []
            if not os.path.exists(self.plot_directory):
                os.makedirs(self.plot_directory)
            plt_path = os.path.join(self.plot_directory, self.network_name_string + '.html')
            output_file(plt_path)
            for origin in plot_titles:

                if os.path.exists("QDA/" + origin + ".npy"):
                    QDA_data = np.load("QDA/" + origin + ".npy")
                QDA_mean = QDA_data[0] / 100
                QDA_meanpstd = QDA_data[1] / 100
                QDA_meanmstd = QDA_data[2] / 100
                QDA_range = range(-40, 71)

                plt_title = 'Accuracy as measured relative to 20m mark. Averaged over all tracks'
                # plot 1
                dataset = graph_results[graph_results['origin'] == origin]
                x_data = []
                y_data = []
                for range_val in np.unique(dataset['d_thresh']):
                    # If I group by track number here, I can get a collection of accuracy scores
                    # and therefore a std dev
                    data_at_range = dataset[dataset['d_thresh'] == range_val]
                    acc = np.average(np.equal(data_at_range['output_idxs'],
                                              data_at_range['destination_vec']))
                    x_data.append(range_val)
                    y_data.append(acc)

                p1 = figure(title='Origin: ' + origin, x_axis_label='Dis from Ref Line (m)', y_axis_label='Acc.',
                            plot_width=400, plot_height=400)  # ~half a 1080p screen
                p1.line(x_data, y_data, legend="Acc. RNN", line_width=2, color='green')
                p1.line(QDA_range, QDA_mean, legend="Acc. QDA", line_width=2, color='red', line_alpha=1)
                # p1.line(QDA_range, QDA_meanmstd, line_width=2, color='red', line_alpha=0.5)
                # p1.line(QDA_range, QDA_meanpstd, line_width=2, color='red', line_alpha=0.5)
                # p1.line(bbox_range, loss, legend="Loss.", line_width=2, color='blue')
                # p1.line(bbox_range, output_gen_plt[:, 1], legend="Generated Output.", line_width=2, color='red')
                p1.legend.location = "bottom_right"
                plots.append(p1)

            button_1 = Button(label='Log: ' + self.network_name_string)

            # put the results in a row

            # p1 = figure(title='Log: ' + get_log_filename(),plot_width=400, plot_height=40)
            # p1.line(1, 1, line_width=2, color='green')
            # plots.append(p1)
            p = gridplot([plots])
            l = layout([plots, [widgetbox(button_1, width=300)]])
            show(l)
            # show(widgetbox(button_1, width=300))
        # if False:  # Use matplotlib to plot PNG
        #     if not os.path.exists(FLAGS.plot_dir):
        #         os.makedirs(FLAGS.plot_dir)
        #     legend_str = []
        #     import matplotlib.pyplot as plt
        #     plt.figure(figsize=(20, 10))
        #     plt.plot(input_plot[:, 0], input_plot[:, 1])
        #     legend_str.append(['Input'])
        #     plt.plot(true_output_plot[:, 0], true_output_plot[:, 1])
        #     legend_str.append(['True Output'])
        #     plt.plot(output_gen_plt[:, 0], output_gen_plt[:, 1])
        #     legend_str.append(['Generated Output'])
        #     plt.legend(legend_str, loc='upper left')
        #     fig_num = 0
        #     while True:
        #         fig_path = os.path.join(FLAGS.plot_dir, get_title_from_params() +
        #                                 '-' + str(fig_num).zfill(3) + '.png')
        #         if not os.path.exists(fig_path):
        #             break
        #         fig_num += 1
        #     plt.savefig(fig_path, bbox_inches='tight')
        #     # plt.show()

        return

    # This function needs the validation batch (or test batch)
    def collect_graph_data(self, batch_handler):
        bbox_range_plot = np.arange(-35,60,1).tolist()

        graph_results = []
        for d in bbox_range_plot:
            junk = 1
            #
            # Set d_thresh
            # Do it in a loop in case batch_size < num_val_tracks


            batch_complete = False
            while not batch_complete:
                        batch_handler.set_distance_threshold(d)

                        mini_batch_frame,batch_complete = batch_handler.get_sequential_minibatch()
                        #FIXME Assumption. format minibatch data preserves ordering. Is this correct?
                        val_x, _, val_weights, val_y = batch_handler.format_minibatch_data(mini_batch_frame['encoder_sample'],
                                                                                           mini_batch_frame['dest_1_hot'],
                                                                                           mini_batch_frame['padding'])
                        valid_data = np.logical_not(mini_batch_frame['padding'].values)
                        acc, loss, outputs = self.model.step(self.sess, val_x, val_y, val_weights, False, summary_writer=None)
                        output_idxs = np.argmax(outputs[0][valid_data], axis=1)
                        #y_idxs = np.argmax(np.array(val_y)[0][valid_data], axis=1)

                        mini_batch_frame = mini_batch_frame[mini_batch_frame['padding'] == False]
                        mini_batch_frame = mini_batch_frame.assign(output_idxs=output_idxs)
                        mini_batch_frame = mini_batch_frame.assign(d_thresh=np.repeat(d,len(mini_batch_frame)))

                        graph_results.append(mini_batch_frame)

        #Concat once only, much faster
        graph_results_frame = pd.concat(graph_results)

        # Reset handler
        batch_handler.set_distance_threshold(None)

        return graph_results_frame

    # Function that passes the entire validation dataset through the network once and only once.
    # Return cumulative accuracy, loss
    def run_validation(self, batch_handler, summary_writer=None):
        batch_complete = False
        batch_losses = []
        total_correct = 0
        total_valid = 0
        while not batch_complete:

            #val_x, val_y, val_weights, pad_vector, batch_complete = batch_handler.get_sequential_minibatch()
            mini_batch_frame,batch_complete = batch_handler.get_sequential_minibatch()
            val_x, _, val_weights, val_y = batch_handler.format_minibatch_data(mini_batch_frame['encoder_sample'],
                                                                               mini_batch_frame['dest_1_hot'],
                                                                               mini_batch_frame['padding'])
            valid_data = np.logical_not(mini_batch_frame['padding'].values)
            acc, loss, outputs = self.model.step(self.sess, val_x, val_y, val_weights, False, summary_writer=summary_writer)

            output_idxs = np.argmax(outputs[0][valid_data], axis=1)
            y_idxs = np.argmax(np.array(val_y)[0][valid_data], axis=1)
            num_correct = np.sum(np.equal(output_idxs,y_idxs)*1)
            num_valid = np.sum(valid_data*1)
            total_correct += num_correct
            total_valid += num_valid
            batch_losses.append(loss)

        batch_acc = np.float32(total_correct) / np.float32(total_valid)

        return batch_acc, np.average(batch_losses), None

