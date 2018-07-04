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


def cluster_MDN_into_sets(MDN_model_output, mix_weight_threshold=0.5, eps=1.0, min_samples=1):
    np.set_printoptions(precision=1)
    # In hindsight this should have been done via a tree structure. I mean, it is, just the representation is poor
    mdn = MDN_model_output[0]
    old_clusterer = None
    clusterer = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=euclid_distance, algorithm='brute')
    cluster_centroids = []  # Dims: groups, timestep, [x, y]
    MDN_groups = []         # Dims: groups, timesteps, [MDN PARAMS]
    MDN_group_padding = []
    live_MDN_groups = []
    try:
        for t in range(len(mdn)):
            model_out_this_timestep = mdn[t]
            # Delete all mixtures that have a weight smaller than threshhold * (1/num_mixes)
            strong_mixes = model_out_this_timestep[:, 0] > ((mix_weight_threshold) / model_out_this_timestep.shape[0])
            model_out_this_timestep = model_out_this_timestep[strong_mixes, :]
            if t == 0:
                groupings = clusterer.fit_predict(model_out_this_timestep)
                if (groupings == -1).all():
                    raise AssertionError
                timestep_groups = []
                for group_idx in range(max(groupings) + 1):
                    # Create a list that only contains the Mixes for group number IDX
                    # Add them to the timetep set.
                    timestep_groups.append(model_out_this_timestep[groupings==group_idx])
                MDN_groups = [[g] for g in timestep_groups]
                live_MDN_groups = range(len(timestep_groups))
            else:
                # We run the grouper again, and then match each group to its closest in t-1. If there are two matches, the
                # tree has split, and so we must copy out history.
                timestep_groups = [[] for i in live_MDN_groups] #range(len(MDN_groups))]

                groupings = clusterer.fit_predict(model_out_this_timestep)
                if (groupings == -1).all():
                    raise AssertionError
                if max(groupings) > 0:
                    ideas = None
                temp_groups_this_timestep = []
                for mdn_output_group_idx in range(max(groupings + 1)):
                    # Collect all groups at that idx
                    this_group_idxs = np.array(range(len(groupings)))[groupings == mdn_output_group_idx]
                    temp_groups_this_timestep.append(model_out_this_timestep[this_group_idxs])

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
                temp_groups_distances = np.array(temp_groups_distances)

                # We now have the map of the closest parent to these children. But only 1 child per parent, unless a new
                # track has spawned.

                used_parent_idxs = []
                live_idx_vals_to_remove = []
                for idx in range(len(live_MDN_groups)):
                    temp_group_idxs_that_match_parent = np.array(range(len(temp_groups_closest)))[temp_groups_closest == live_MDN_groups[idx]]
                    # Because I used the index to remap to a limited range of idxs, I have to un map it again
                    if len(temp_groups_distances[temp_group_idxs_that_match_parent]) < 1:
                        # If a track dies
                        live_idx_vals_to_remove.append(live_MDN_groups[idx])
                        continue
                    closest_parent_group_idx = temp_group_idxs_that_match_parent[np.argmin(temp_groups_distances[temp_group_idxs_that_match_parent])]
                    # Now that I have found the closest group, I can assign it
                    timestep_groups[idx] = temp_groups_this_timestep[closest_parent_group_idx]
                    used_parent_idxs.append(closest_parent_group_idx)

                ################ PATH DIES
                # It is undefined to remove when iterating over a list
                for idx_val in live_idx_vals_to_remove:
                    idx = live_MDN_groups.index(idx_val)  # First, find the mapping_idx in the live_MDN_groups list (NOT a regular idx)
                    del(live_MDN_groups[idx])             # Then delete those from the list
                    del(timestep_groups[idx])             # As a group is deleted in the master tree, the addition tree must also be deleted
                ################ /PATH DIES

                unused_temp_group_idxs = np.delete(np.array(range(len(temp_groups_closest))), used_parent_idxs)
                if len(unused_temp_group_idxs) > 0:
                    ######## PATH SPAWNS
                    for unused_temp_group_idx in unused_temp_group_idxs:
                        MDN_group_idx += 1
                        MDN_groups.append(copy.deepcopy(MDN_groups[temp_groups_closest[unused_temp_group_idx]]))
                        timestep_groups.append(temp_groups_this_timestep[unused_temp_group_idx])
                        live_MDN_groups.append(len(MDN_groups)-1)

                for i in range(len(timestep_groups)):
                    # if i not in live_MDN_groups:
                    #     continue
                    # if len(timestep_groups[i]) == 0:
                    #     continue
                    try:
                        MDN_groups[live_MDN_groups[i]].append(np.array(timestep_groups[i]))
                    except IndexError:
                        ideas = None
                if len(MDN_groups) > 6:
                    WTFisGOINGon = None
                if len(MDN_groups[0]) > t+1:
                    ohno = None

        # Return 2 things: The raw MDNS and centroid clusters for simplicity
        centroid_groups = []
        centroid_weights = []
        for path in MDN_groups:
            simple_path = []
            path_weights = 0
            for vals_at_timestep in path:
                simple_path.append(np.average(vals_at_timestep[:, 1:3], weights=vals_at_timestep[:, 0], axis=0))
                path_weights += np.sum(vals_at_timestep[:, 0])

            # Normalize according to length of path
            # This avoid bias where longer paths sum more weight and are therefore more important
            #path_weights /= len(path)
            centroid_weights.append(path_weights)
            centroid_groups.append(simple_path)
        #Normalize
        centroid_weights = np.array(centroid_weights) / np.sum(centroid_weights)
    except AssertionError:
        # Oh no! At least one timestep had no groupings! This could be because threshold or min points is set too high
        # The only thing to do now is return the strongest single track as a sort of placeholder solution
        print "Warning: The clustering algorithm failed to find a solution. Either threshold or min_samples is too high"
        MDN_groups = None
        for MDN_timestep_idx in range(len(MDN_model_output[0])):
            MDN_timestep = MDN_model_output[0,MDN_timestep_idx]
            strongest_MDN_idx = np.argmax(MDN_timestep[:, 0])
            strongest_MDN = MDN_timestep[strongest_MDN_idx]
            if MDN_timestep_idx == 0:
                MDN_groups = [strongest_MDN]
            else:
                MDN_groups.append(strongest_MDN)
        MDN_groups = [np.array(MDN_groups)]
        centroid_weights = [1.0]
        centroid_groups = [MDN_groups[0][:, 1:3]]

    return MDN_groups, centroid_groups, centroid_weights