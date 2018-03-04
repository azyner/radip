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

# Models I want:
# CV, CA, CTRA -- All of which should fail miserably at an intersection, really.
# There is a Trivedi HMM VGMM (variational Gaussian Mixture Models) which
# seems absolutely parallel to my current work, and so would be valuable to include

# Metrics I want:
# Horizon based metrics: 0.5, 1.0, 1.5, 2.0, 2.5 sec etc
#   Median and Mean Absolute Error (cartesian)
#   Worst 5% and worst 1%

# Track based metrics:
# Modified Hausdorff Distance
# Euclidean Summation?

# Implementing ALL of the above is more work than I've seen in a lot of journals.
# So it should be very thorough.
