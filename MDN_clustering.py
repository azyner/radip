import numpy as np
import scipy
import sklearn.cluster
import pandas as pd
import copy

def KL_divergence():
    return None


def bhattacharyya_distance():
    return None


def euclid_distance(a, b):
    return np.sqrt(np.square(a[1]-b[1]) + np.square(a[2]-b[2]))


def cluster_MDN_into_sets(MDN_model_output):
    mdn = MDN_model_output[0]
    old_clusterer = None
    clusterer = sklearn.cluster.DBSCAN(eps=3, min_samples=0, metric=euclid_distance, algorithm='brute')
    cluster_centroids = []  # Dims: groups, timestep, [x, y]
    MDN_groups = []         # Dims: groups, timesteps, [MDN PARAMS]
    for t in range(len(mdn)):
        if t == 0:
            groupings = clusterer.fit_predict(mdn[t])
            timestep_groups = []
            for group_idx in range(max(groupings) + 1):
                # Create a list that only contains the Mixes for group number IDX
                # Add them to the timetep set.
                timestep_groups.append(mdn[t][groupings==group_idx])
            MDN_groups.append(timestep_groups)
        else:
            timestep_groups = [[] for i in range(len(MDN_groups))]
            for mdn_output_idx in range(len(mdn[t])):
                for mdn_group_idx in range(len(MDN_groups)):
                    # This matches what to what?
                    groupings = clusterer.fit_predict(np.append(MDN_groups[mdn_group_idx][t-1],
                                                                [mdn[t][mdn_output_idx]], axis=0))
                    if max(groupings) == 0:
                        # There is only one group, we have a match
                        timestep_groups[mdn_group_idx].append(mdn[t][mdn_output_idx])
                        break
                    else:
                        ideas = None
                else:
                    # if it does not match any existing group (i.e. two Mixes formed a new group)
                    for timestep_group_idx in range(len(timestep_groups)):
                        groupings = clusterer.fit_predict(np.append(timestep_groups[timestep_group_idx],
                                                                    [mdn[t][mdn_output_idx]], axis=0))
                        if max(groupings) == 0:
                            timestep_groups[timestep_group_idx].append(mdn[t][mdn_output_idx])
                            break
                        else:
                            ideas = None
                    else:
                        # make a new group
                        timestep_groups.append([mdn[t][mdn_output_idx]])
                        # figure out which group it matches from the previous clusters
                        # Use min euclid distance to any of the mixes in the group
                        # Then copy that track's history.
                        closest_idx = None
                        closest_val = 999999
                        for group_idx in range(len(MDN_groups)):
                            for gauss in MDN_groups[group_idx]:
                                if euclid_distance(gauss, mdn[t][mdn_output_idx]) < closest_val:
                                    closest_val = euclid_distance(gauss, mdn[t][mdn_output_idx])
                                    closest_idx = group_idx
                        # Copy all the history from group_idx t-- to new group
                        # Do I have to deepcopy this? Appending an element of an array to the same array is unkwn
                        MDN_groups.append(copy.deepcopy(MDN_groups[closest_idx])) #TODO What aobut timestep groups
            #MDN_groups.append(timestep_groups)
            for i in range(len(timestep_groups)):
                MDN_groups[i].append(np.array(timestep_groups[i]))

        # I then either have to figure out how the new clusters map to the old clusters,
        # Or run the clustering algorithm again individually.
        # I have to run them individually. If I don't, the new data might `join' the old together,
        # That's easier then, for each Mix, see if it groups with any t-1 group.

        #    for group_idx in range(max(groupings)):
        # If they split, the cluster gets copied such that they have the same history
        # new_clusterer = sklearn.cluster.DBSCAN(min_samples=1, metric=euclid_distance)
        # groupings = new_clusterer.fit_predict(mdn[t])
        # for group_idx in range(max(groupings)):
        #
        #     ideas = None

    return {}