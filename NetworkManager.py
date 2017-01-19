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

    # def collect_data_for_graph(self, batch_handler):
    #     output_dataframe = pd.DataFrame()
    #     bbox_range_plot = np.arange(-35,60,1).tolist()
    #     for d in range(bbox_range_plot):
    #         batch_handler.set_distance_threshold(d)
    #         output_dataframe = pd.DataFrame({'track_index': indices,
    #                                          'prediction':map(tuple,output_prediction[0]),
    #                                          'gt_label':map(tuple,label_sequences[0]),
    #                                          'range':np.repeat(bbox,len(indices)),
    #                                          'iteration':np.repeat(x,len(indices)),
    #                                          'cross-fold':np.repeat(cross_fold,len(indices)),
    #                                          'destination':[x[x.find('-')+1:] for x in np.array(raw_classes)[indices]],
    #                                          'origin':[x[:x.find('-')] for x in np.array(raw_classes)[indices]]})
    #                     iteration_output_list.append(output_dataframe)
    #     return

    def generate_graph(self):


        return

    # Function that passes the entire validation dataset through the network once and only once.
    # Return cumulative accuracy, loss
    def run_validation(self, batch_handler, summary_writer=None):
        batch_complete = False
        batch_losses = []
        total_correct = 0
        total_valid = 0
        while not batch_complete:

            #val_x, val_y, val_weights, pad_vector, batch_complete = batch_handler.get_sequential_minibatch()
            batch_frame,batch_complete = batch_handler.get_sequential_minibatch()
            val_x, _, val_weights, val_y = batch_handler.format_minibatch_data(batch_frame['encoder_sample'],
                                                                                        batch_frame['dest_1_hot'],
                                                                                        batch_frame['padding'])
            valid_data = np.logical_not(batch_frame['padding'])
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

