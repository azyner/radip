import numpy as np
import scipy
import sklearn.cluster
import pandas as pd
import copy
import scipy.spatial

def KL_divergence():
    return None


def bhattacharyya_distance():
    return None


def euclid_distance(a, b):
    return np.sqrt(np.square(a[1]-b[1]) + np.square(a[2]-b[2]))


def cluster_MDN_into_sets(MDN_model_output):
    mdn = MDN_model_output[0]
    old_clusterer = None
    clusterer = sklearn.cluster.DBSCAN(eps=1, min_samples=1, metric=euclid_distance)
    cluster_centroids = []  # Dims: groups, timestep, [x, y]
    MDN_hypotheses = []         # Dims: groups, timesteps, [MDN PARAMS]
    for t in range(len(mdn)):
        groupings = clusterer.fit_predict(mdn[t])
        timestep_groups = []
        for group_idx in range(max(groupings) + 1):
            # Create a list that only contains the Mixes for group number IDX
            # Add them to the timetep set.
            timestep_groups.append(mdn[t][groupings==group_idx])

        # If first run, there is no t-1 to match to
        if t == 0:
            MDN_hypotheses = [timestep_groups]
            continue

        # else now find the closest group in t-1 for t
        # if there are two groups in t that match t-1, split
        if max(groupings) > 0:
            # List of the closest group from t-1 for group[idx] at time t
            closest_group = []
            # The distance of the above
            distance_to_group = []
            for group_idx in range(max(groupings)+1):
                #Consoladate all mixes into one centroid average
                group_mixes = mdn[t][groupings == 0]
                group_centroid = np.mean(group_mixes[:, 1:3], axis=0)

                hypo_idx = 0
                while True:
                    try:
                        closest_group = MDN_hypotheses[hypo_idx][t - 1]
                        closest_distance = MDN_hypotheses[hypo_idx][t - 1]
                        break
                    except IndexError:
                        hypo_idx += 1

                for hypo_idx in range(len(MDN_hypotheses)):
                    try:
                        # check distance between hypo_idx and group_centroid
                        group_distance = scipy.spatial.distance.euclidean(group_centroid, np.mean(MDN_hypotheses[hypo_idx][t][:, 1:3], axis=0))
                        if group_distance < closest_distance:
                            closest_group = hypo_idx
                            closest_distance = group_distance
                    except IndexError:
                        continue

                # I now have a list of groups and their closest t-1 group.
                ideas = None
                for mix in groupings[groupings == group_idx]:
                    ideas = None
                    # Find closest t-1 group and distance to group
                    # of all mixes, keep closest
        else:
            for hypo_idx in range(len(MDN_hypotheses)):
                MDN_hypotheses[hypo_idx].append(timestep_groups[hypo_idx])
            ideas = None
        # break



            #
            # timestep_groups = [[] for i in range(len(MDN_hypotheses[t-1]))]
            # for mdn_output_idx in range(len(mdn[t])):
            #     #
            #
            #
            #     try:
            #         for mdn_group_idx in range(len(MDN_hypotheses[t-1])):
            #             groupings = clusterer.fit_predict(np.append(mdn[t-1],[mdn[t][mdn_output_idx]], axis=0))
            #             if max(groupings) == 0:
            #                 # There is only one group, we have a match
            #                 timestep_groups[mdn_group_idx].append(mdn[t][mdn_output_idx])
            #                 break
            #         else:
            #             # if it does not match any existing group (i.e. two Mixes formed a new group)
            #             # TODO what about the case where none match?
            #             for timestep_group_idx in range(len(timestep_groups)):
            #                 if len(timestep_groups[timestep_group_idx]) == 0:
            #                     continue
            #                 groupings = clusterer.fit_predict(np.append(timestep_groups[timestep_group_idx],
            #                                                             [mdn[t][mdn_output_idx]], axis=0))
            #                 if max(groupings) == 0:
            #                     timestep_groups[timestep_group_idx].append(mdn[t][mdn_output_idx])
            #                     break
            #             else:
            #                 # make a new group
            #                 timestep_groups.append([mdn[t][mdn_output_idx]])
            #                 # figure out which group it matches from the previous clusters
            #                 # Use min euclid distance to any of the mixes in the group
            #                 # Then copy that track's history.
            #                 closest_idx = None
            #                 closest_val = 999999
            #                 for group_idx in range(len(MDN_hypotheses)):
            #                     for gauss in MDN_hypotheses[group_idx]:
            #                         if euclid_distance(gauss, mdn[t][mdn_output_idx]) < closest_val:
            #                             closest_val = euclid_distance(gauss, mdn[t][mdn_output_idx])
            #                             closest_idx = group_idx
            #                 # Copy all the history from group_idx t-- to new group
            #                 # Do I have to deepcopy this? Appending an element of an array to the same array is unkwn
            #                 # TODO THIS CODE IS WRONG, I have to extend each list at dim 1, not dim 0
            #                 MDN_hypotheses.append(copy.deepcopy(MDN_hypotheses[closest_idx]))
            #         MDN_hypotheses.append(timestep_groups)
            #
            #     except IndexError:
            #         wtf = 0
            # #
            # # I then either have to figure out how the new clusters map to the old clusters,
            # # Or run the clustering algorithm again individually.
            # # I have to run them individually. If I don't, the new data might `join' the old together,
            # # That's easier then, for each Mix, see if it groups with any t-1 group.
            #
            # #    for group_idx in range(max(groupings)):
            # # If they split, the cluster gets copied such that they have the same history
            # new_clusterer = sklearn.cluster.DBSCAN(min_samples=1, metric=euclid_distance)
            # groupings = new_clusterer.fit_predict(mdn[t])
            # for group_idx in range(max(groupings)):
            #
            #     ideas = None

    return {}