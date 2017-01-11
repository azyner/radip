#class that holds JUST THE MODEL
#Its not the trainer/runner. Just the model bit.

#The classification head may be simple enough to put in here, but it should be explicitly identified, such that it
# is obvious where the MDN head goes.

class RNN_model:
    def __init__(self, summary_writer=None):
        return

    def step(self,batch_data):
        return

    def run_test(self,batch_data):
        #generate plot, dump to tensorboard
        return

# This sounds complicated, but could I load two model instances, training and testing, so that they both use the exact
# same underlying weights in memory, but have different batch sizes, such that I can quickly run the test batch?
# For more, I should probably read the exact functionality of the tensorflow checkpointer.