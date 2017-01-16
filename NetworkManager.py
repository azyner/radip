# Class that handles the network itself. Defines the training / testing state,
#  manages tensorboard handles.

# Step function should go here, that means it needs to be passed a BatchHandler,
# so that it can grab data easily
# This should also have the different test types, such as the accuracy graph
# Should it handle the entirety of crossfolding?
# I don't think so, that should go into another class maybe
import tensorflow as tf
from seq2seq_model import Seq2SeqModel

class NetworkManager:
    def __init__(self, parameters):
        self.parameters = parameters
        self.network = None
        self.batchHandler = None
        self.sess = None
        self.device = None

        return

    def build_model(self):
        self.device = tf.device('gpu:0')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
        model = Seq2SeqModel(self.parameters)


        return
