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

    def step(self,X,Y,weights,train_model,summary_writer=None):
         return self.model.step(self.sess, X, Y, weights, train_model, summary_writer=summary_writer)

    def generate_graph(self):
        return
