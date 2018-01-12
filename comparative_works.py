import numpy

# This module is the one to run comparative
# So the input will be important:
# - index numbers of tracks, or the actual train / test data
# - This may be a cache of the results from the network
# - Depending on the methods, I may want to cache this too
# - How about the graph writer? Should that become more global to allow it to be run with data from here?
#   -- Probably.
#
# Have a look at the handles `collate_graphs.py' gets. I may need to add more to the results folder.
# --> This only ran for categorical tests. I'll have to re-do the whole thing.

# In order for me to begin this, I need a standardized graph. I also need a standardized metric at a specific set of
# data points. Do that first, and do it within the standard networkmanager class. I'll spin it out later.
