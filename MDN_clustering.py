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
    MDN_group_padding = []
    live_MDN_groups = []
    for t in range(len(mdn)):
        if t == 0:
            groupings = clusterer.fit_predict(mdn[t])
            timestep_groups = []
            for group_idx in range(max(groupings) + 1):
                # Create a list that only contains the Mixes for group number IDX
                # Add them to the timetep set.
                timestep_groups.append(mdn[t][groupings==group_idx])
            MDN_groups.append(timestep_groups)
            live_MDN_groups = [0]
        else:
            # We run the grouper again, and then match each group to its closest in t-1. If there are two matches, the
            # tree has split, and so we must copy out history.
            timestep_groups = [[] for i in live_MDN_groups] #range(len(MDN_groups))]

            groupings = clusterer.fit_predict(mdn[t])
            if max(groupings) > 0:
                ideas = None
            temp_groups_this_timestep = []
            for mdn_output_group_idx in range(max(groupings + 1)):
                # Collect all groups at that idx
                this_group_idxs = np.array(range(len(groupings)))[groupings == mdn_output_group_idx]
                # FIXME Assumption groupings are in the same order.
                temp_groups_this_timestep.append(mdn[t][this_group_idxs])

            # Now I have this timestep grouped, I need to find their closest group from t-1 (MDN_groups)
            # If there are two with the same group, the closer wins, and the further spawns a new group, copying
            # its history
            # So now I need a map between temp_groups and MDN t-1, and distances

            ideas = None
            temp_groups_closest = []
            temp_groups_distances = []
            for i in range(len(temp_groups_this_timestep)):
                temp_group_centroid = np.average(temp_groups_this_timestep[i][:, 1:3],
                                                 weights=temp_groups_this_timestep[i][:, 0], axis=0)
                closest_group = None
                closest_distance = 999999999
                for MDN_group_idx in live_MDN_groups:
                    try:
                        MDN_group_centroid = np.average(MDN_groups[MDN_group_idx][t-1][:, 1:3],
                                                    weights=MDN_groups[MDN_group_idx][t-1][:, 0], axis=0)
                    except IndexError:
                        ideas = None
                    this_distance = scipy.spatial.distance.euclidean(temp_group_centroid,MDN_group_centroid)
                    if closest_group is None:
                        closest_group = MDN_group_idx
                        closest_distance = this_distance
                        continue
                    elif this_distance < closest_distance:
                        closest_group = MDN_group_idx
                        closest_distance = this_distance
                temp_groups_closest.append(closest_group)
                temp_groups_distances.append(closest_distance)

            temp_groups_closest = np.array(temp_groups_closest)
            temp_groups_distances = np.array( temp_groups_distances)
            used_idxs = []
            live_idxs_to_remove = []
            for MDN_group_idx in live_MDN_groups:
                temp_group_idxs_that_match = np.array(range(len(temp_groups_closest)))[temp_groups_closest == MDN_group_idx]
                # Because I used the index to remap to a limited range of idxs, I have to un map it again
                if len(temp_groups_distances[temp_group_idxs_that_match]) < 1:
                    # If a track dies
                    live_idxs_to_remove.append(MDN_group_idx)
                    continue
                closest_group_idx = temp_group_idxs_that_match[np.argmin(temp_groups_distances[temp_group_idxs_that_match])]
                # Now that I have found the closest group, I can assign it
                timestep_groups[MDN_group_idx] = temp_groups_this_timestep[closest_group_idx]
                used_idxs.append(closest_group_idx)

            # It is undefined to remove when iterating over a list
            for idx in live_idxs_to_remove:
                live_MDN_groups.remove(idx)

            unused_idxs = np.delete(np.array(range(len(temp_groups_closest))), used_idxs)
            if len(unused_idxs) > 0:
                for unused_idx in unused_idxs:
                    MDN_group_idx += 1
                    MDN_groups.append(copy.deepcopy(MDN_groups[temp_groups_closest[unused_idx]]))
                    timestep_groups.append(temp_groups_this_timestep[unused_idx])
                    live_MDN_groups.append(len(MDN_groups)-1)


            #timestep_groups[mdn_output_group_idx] = mdn[t][this_group_idxs]
            # TODO This is wrong. The assumption that the gourps are linear no longher hold
            # Groups lengths are not equal?
            if len(timestep_groups) is not len(live_MDN_groups):
                help = None
            for i in range(len(timestep_groups)):
                if len(timestep_groups[i]) == 0:
                    continue
                try:
                    MDN_groups[live_MDN_groups[i]].append(np.array(timestep_groups[i]))
                except IndexError:
                    ideas = None






            # for mdn_output_idx in range(len(mdn[t])):
            #     for mdn_group_idx in range(len(MDN_groups)):
            #         # If the mix from current timestep does not match to
            #         groupings = clusterer.fit_predict(np.append(MDN_groups[mdn_group_idx][t-1],
            #                                                     [mdn[t][mdn_output_idx]], axis=0))
            #         if max(groupings) == 0:
            #             # There is only one group, we have a match
            #             timestep_groups[mdn_group_idx].append(mdn[t][mdn_output_idx])
            #             break
            #         else:
            #             ideas = None
            #     else:
            #         # if it does not match any existing group (i.e. two Mixes formed a new group)
            #         for timestep_group_idx in range(len(timestep_groups)):
            #             groupings = clusterer.fit_predict(np.append(timestep_groups[timestep_group_idx],
            #                                                         [mdn[t][mdn_output_idx]], axis=0))
            #             if max(groupings) == 0:
            #                 timestep_groups[timestep_group_idx].append(mdn[t][mdn_output_idx])
            #                 break
            #             else:
            #                 ideas = None
            #         else:
            #             # make a new group
            #             timestep_groups.append([mdn[t][mdn_output_idx]])
            #             # figure out which group it matches from the previous clusters
            #             # Use min euclid distance to any of the mixes in the group
            #             # Then copy that track's history.
            #             closest_idx = None
            #             closest_val = 999999
            #             for group_idx in range(len(MDN_groups)):
            #                 for gauss in MDN_groups[group_idx]:
            #                     if euclid_distance(gauss, mdn[t][mdn_output_idx]) < closest_val:
            #                         closest_val = euclid_distance(gauss, mdn[t][mdn_output_idx])
            #                         closest_idx = group_idx
            #             # Copy all the history from group_idx t-- to new group
            #             # Do I have to deepcopy this? Appending an element of an array to the same array is unkwn
            #             MDN_groups.append(copy.deepcopy(MDN_groups[closest_idx])) #TODO What aobut timestep groups
            # #MDN_groups.append(timestep_groups)
            # for i in range(len(timestep_groups)):
            #     MDN_groups[i].append(np.array(timestep_groups[i]))

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