# Class that handles the network itself. Defines the training / testing state, manages tensorboard handles.

# Step function should go here, that means it needs to be passed a BatchHandler, so that it can grab data easily

# This should also have the different test types, such as the accuracy graph

class NetworkManager:
    def __init__(self):
        self.network = None
        self.batchHandler = None

        return

    def step(self):
        return
