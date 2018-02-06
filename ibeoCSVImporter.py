import numpy as np
import scipy as sp
import random
import bokeh
import csv
import sys
import pandas as pd
import struct
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import os
import dill as pickle
import utils

class ibeoCSVImporter:
    def __init__(self, parameters, csv_name):
        self.unique_id_idx = int(1) #Made object global as disambig is called multiple times now.
        if isinstance(csv_name,str):
            csv_name = [csv_name]
        self.labelled_track_list = []
        self._cumulative_dest_list = []
        self._cumulative_origin_list = []
        # Check if I have cached this already
        # name it after the last csv in csv_name
        cache_name = abs(hash(tuple(csv_name)) + utils.get_library_hash(['ibeoCSVImporter.py']))
        file_path = 'data/' + str(cache_name) + ".pkl"
        if not os.path.isfile(file_path):
            for csv_file in csv_name:
                print "Reading CSV " + csv_file
                input_df = pd.read_csv('data/' + csv_file)
                input_df['csv_name'] = [csv_file]*len(input_df)
                self.lookup_intersection_extent(csv_file)
                parsed_df = self._parse_ibeo_df(input_df)
                input_df = None
                #print "Disambiguating tracks"
                disambiguated_df = self._disambiguate_df(parsed_df)
                parsed_df = None
                labelled_track_list = self._label_df(disambiguated_df)
                #print "Calculating intersection distance"

                sub_track_list = self._calculate_intersection_distance(labelled_track_list)
                trimmed_tracks = self._trim_tracks(sub_track_list)
                related_tracks = self._add_relative_tracks(trimmed_tracks)
                self.labelled_track_list.extend(trimmed_tracks)
                print "#### CUMULATIVE SUMMARY ####"
                self._print_collection_summary()
                self._print_collection_summary(relative=True)
                # write pkl
            with open(file_path, 'wb') as pkl_file:
                pickle.dump(self.labelled_track_list, pkl_file)
        else:
            with open(file_path, 'rb') as pkl_file:
                self.labelled_track_list = pickle.load(pkl_file)
            # Grab some labels for the summary printer
            for csv in csv_name:
                self.lookup_intersection_extent(csv)

        self._print_collection_summary()
        self._print_collection_summary(relative=True)

    def lookup_intersection_extent(self,csv_name):
        #        format: x_min,x_max,y_min,y_max
        if '20170427-stationary-2-leith-croydon' in csv_name:
            top_exit = [-33, -30, 3, 4]
            top_enter = [-23, -20, 6, 7]
            right_exit = [-16, -15, -3, 2]
            right_enter = [-16, -15, -12, -8]
            low_exit = [-23, -21, -19, -18]
            low_enter = [-33, -30, -16, -15]
            intersection_centre = [-25.8, -5]
            intersection_rotation = 0
            self.dest_gates = {"north": top_exit, "east": right_exit, "south": low_exit}
            self.origin_gates = {
                "north": top_enter,
                "east": right_enter,
                "south": low_enter
            }
            self.relative_ring = ['north', 'east', 'south', 'west'] # Clockwise relation of each gate
        if (('20170601-stationary-3-leith-croydon' in csv_name) or
            ('stationary-4-leith-croydon' in csv_name) or
            ('stationary-5-leith-croydon' in csv_name)):
            # left right bottom top
            top_exit = [-25, -16, 0, 0.5]
            top_enter = [-12, -6, 0, 0.5]
            right_exit = [-1.0, -0, -6, 0]
            right_enter = [-1.0, 0.0, -17, -10]
            left_exit = [-25, -24.1, -16, -9]
            left_enter = [-26, -25, -7, -0]
            intersection_centre = [-14.4, -7.5]
            intersection_rotation = 0  # 90 degree?
            self.dest_gates = {"north": left_exit, "east": top_exit, "south": right_exit}
            self.origin_gates = {
                "north": left_enter,
                #"east": top_enter,
                "south": right_enter
            }
            self.relative_ring = ['north', 'east', 'south', 'west']
        if 'queen-hanks' in csv_name:
            # left right bottom top
            top_exit = [26, 30, 6, 7]
            right_exit = [41, 42, -3, 3]
            right_enter = [41, 42, -9, -6]
            left_enter = [18, 20, -2, 2]
            left_exit = [18, 20, -10, -6]
            bottom_exit = [30, 35, -14, -13]
            self.dest_gates = {"north": right_exit, "east": bottom_exit, "west": top_exit, "south": left_exit}
            self.origin_gates = {
                "south": left_enter,
                "north": right_enter
            }
            self.relative_ring = ['north', 'east', 'south', 'west']
        if 'roslyn-crieff' in csv_name:
            # left right bottom top
            right_exit = [-12,-10,-4,2]
            right_enter = [-12,-10,-12,-8]
            left_enter = [-31,-29,-3,1]
            left_exit = [-31,-29,-13,-8]
            top_exit = [-27,-24,3,5]
            bottom_exit = [-18,-14,-15,-13]
            self.dest_gates = {"NW": right_exit, "NE": bottom_exit, "SW": top_exit, "SE": left_exit}
            self.origin_gates = {
                "SE": left_enter,
                "NW": right_enter
            }
            self.relative_ring = ['NW', 'NE', 'SE', 'SW']
        if 'oliver-wyndora' in csv_name:
            # left right bottom top
            right_exit = [-10, -8, -5, 0]
            right_enter = [-10, -8, -12, -6]
            bottom_exit = [-17, -13, -16, -14]
            left_exit = [-28, -26, -12, -8]
            left_enter = [-28, -26, -3, 0]
            top_exit = [-22, -18, 4, 6]
            self.dest_gates = {"north": right_exit, "east": bottom_exit, "west": top_exit, "south": left_exit}
            self.origin_gates = {
                "south": left_enter,
                "north": right_enter
            }
            self.relative_ring = ['north', 'east', 'south', 'west']
        if 'orchard-mitchell' in csv_name:
            # left right bottom top
            right_enter = [-11, -9, -13, -5]
            right_exit = [-11, -9, -4, -2]
            bottom_exit = [-23, -16, -16, -14]
            left_exit = [-30, -28, -13, -5]
            left_enter = [-30, -28, -5, 1]
            top_exit = [-24, -18, 4, 6]
            self.dest_gates = {"north": right_exit, "east": bottom_exit, "west": top_exit, "south": left_exit}
            self.origin_gates = {
                "south": left_enter,
                "north": right_enter
            }
            self.relative_ring = ['north', 'east', 'south', 'west']
        for key, value in self.dest_gates.iteritems():
            self._cumulative_dest_list.append(key)
        for key, value in self.origin_gates.iteritems():
            self._cumulative_origin_list.append(key)
        self._cumulative_origin_list = list(set(self._cumulative_origin_list))
        self._cumulative_dest_list = list(set(self._cumulative_dest_list))

    def _get_relative_exit(self, origin_label, dest_label):
        o_idx = self.relative_ring.index(origin_label)
        d_idx = self.relative_ring.index(dest_label)
        relative_map = {-1: "right",
                        0: "u-turn",
                        1: "left",
                        -2: "straight",
                        2: "straight",
                        3: "right",
                        -3: "left"}
        return relative_map[d_idx-o_idx]

    def get_track_list(self):
        return self.labelled_track_list

    def _print_collection_summary(self, relative=False):
        # Here I want to print a origin/destination matrix
        # Preferably with summary margins
        if relative:
            dest_key_list = ['left', 'straight', 'right', 'u-turn']
        else:
            dest_key_list = self._cumulative_dest_list #[key for key, value in self.dest_gates.iteritems()]
        orig_key_list = self._cumulative_origin_list #[key for key, value in self.origin_gates.iteritems()]

        summary_df = pd.DataFrame(np.zeros([len(orig_key_list), len(dest_key_list)]),
                                  index=orig_key_list, columns=dest_key_list)
        for single_track in self.labelled_track_list:
            if relative:
                summary_df.loc[single_track.iloc[0]["origin"], single_track.iloc[0]["relative_destination"]] += 1
            else:
                summary_df.loc[single_track.iloc[0]["origin"], single_track.iloc[0]["destination"]] += 1
        # Add marginals
        summary_df["total"] = summary_df.sum(1)
        summary_df = summary_df.append(pd.Series(summary_df.sum(0), name="total"))

        print "origin | destination"
        print summary_df

    def _in_box(self, point, extent):
        """Return if a point is within a spatial extent."""
        return ((point[0] >= extent[0]) and
                (point[0] <= extent[1]) and
                (point[1] >= extent[2]) and
                (point[1] <= extent[3]))

    def _parse_ibeo_df(self, input_df):
        """ This function is used to clean up some of the many parameters inside the dataframe."""

        # Clean up the cs and add some labels
        # No longer works with the split csv
        #input_df.Timestamp = input_df.Timestamp - input_df.iloc[0].Timestamp
        # motionModelValidated is always true
        input_df.trackedByStationaryModel = (input_df.trackedByStationaryModel == 1)
        input_df.mobile = (input_df.mobile == 1)
        input_df['ObjPrediction'] = (input_df.ObjectPredAge > 0)

        # Here I can inject some more robust object tracking methods. I am just using object centre for now, which can
        # be noisy as the estimate of the object size changes significantly

        input_df["Object_X"] = input_df["ObjBoxCenter_X"]
        input_df["Object_Y"] = input_df["ObjBoxCenter_Y"]
        #TODO Resolve AbsVelocity X/Y into one magnitude scalar

        return input_df

    def _lookup_intersection_origin_per_entrance(self, csv_name, entrance):
        if 'oliver-wyndora' in csv_name:
            if entrance == 'north':
                origin = [-10.3, -5.3]
                rotation = [-1, 0]
            if entrance == 'south':
                origin = [-26.7, -5.3]
                rotation = [1, 0]

        if 'roslyn-crieff' in csv_name:
            if entrance == 'NW':
                origin = [-11.8, -5.3]
                rotation = [-1, 0]
            if entrance == 'SE':
                origin = [-30.5, -5.3]
                rotation = [1, 0]

        if 'queen-hanks' in csv_name:
            if entrance == 'north':
                origin = [38.5, -3.0]
                rotation = [-1, 0]
            if entrance == 'south':
                origin = [21.7, -3.8]
                rotation = [1, 0]

        if 'orchard-mitchell' in csv_name:
            if entrance == 'north':
                origin = [-11.5, -5.3]
                rotation = [-1, 0]
            if entrance == 'south':
                origin = [-28.5, -5.3]
                rotation = [1, 0]

        if 'leith-croydon' in csv_name:
            if entrance == 'north':
                origin = [-23.3, -7.8]
                rotation = [1, 0]
            if entrance == 'south':
                origin = [-5, -7.8]
                rotation = [-1, 0]

        return origin, rotation


# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _add_relative_tracks(self, tracks):
        new_tracks = []
        for track in tracks:
            # Subtract origin first, then rotate.
            # This means the new origin does not need to be rotated
            x = track['Object_X']
            y = track['Object_Y']
            origin_coords, orig_vec = self._lookup_intersection_origin_per_entrance(track.csv_name.iloc[0],
                                                                                    track.origin.iloc[0])
            x_z = x - origin_coords[0]
            y_z = y - origin_coords[1]
            dest_vec = [0, 1]
            a = self._angle_between(orig_vec, dest_vec)
            new_x = x_z*np.cos(a) - y_z*np.sin(a)
            new_y = x_z*np.sin(a) + y_z*np.cos(a)
            new_a = track.ObjBoxOrientation + a
            # shift to -pi and pi
            new_a = ((new_a + 2*np.pi) % 2*np.pi) - np.pi
            track['relative_x'] = new_x
            track['relative_y'] = new_y
            track['relative_angle'] = new_a
            new_tracks.append(track)

        return new_tracks

    def _disambiguate_df(self,input_df):
        drop_list = []
        DROP_INDEX = -1
        object_id_list = np.sort(input_df.ObjectId.unique())
        # Classes 4 and 5 are car, truck (maybe in that order)
        # 3 might be bike, have to check.
        # I only care about cars and trucks right now.

        vehicle_df = input_df # .loc[input_df.Classification > 3, :]

        disambiguated_df_list = []

        for ID in object_id_list:
            sys.stdout.write("\rDisambiguating track: %04d of %04d" % (ID, len(object_id_list)))
            sys.stdout.flush()
            obj_data = vehicle_df.loc[vehicle_df.ObjectId == ID, :]

            # Some objects have no data at all.
            if len(obj_data) < 5:
                continue

            obj_data = obj_data.reset_index() # Create copy, and reset the index to be contiguous
            obj_data.rename(columns={'index': "file_index"}, inplace=True)
            obj_data = obj_data.sort_values(['Timestamp'], ascending=True)

            # Diff in its current format gives an off-by one error when passed to my splitter function.
            # insert/delete used to shift the whole array over by one.
            obj_diff = np.diff(obj_data['Timestamp'])
            obj_diff = np.insert(np.delete(obj_diff, -1), [0], [0])
            cuts = np.where(obj_diff > 1)[0]

            prev_cut = 0
            cuts = np.append(cuts, len(obj_data)-1)
            for cut in cuts:
                #print obj_data.loc[prev_cut:cut, :]

                # Now that we have isolated the tracks, drop any that do not meet classification requirements
                # Note that the class may begin as `unknown', and later become classified, so do not check only for
                #  any class > 3, but instead check all class > 3

                if (obj_data.loc[prev_cut:cut].Classification < 4).all():
                    obj_data.loc[prev_cut:cut, "uniqueId"] = DROP_INDEX
                else:
                    obj_data.loc[prev_cut:cut, "uniqueId"] = self.unique_id_idx
                    self.unique_id_idx += 1

                #print("ObjId:" + str(ID) + " UniqueId:" + str(unique_id_idx) +
                #      " Prev:" + str(prev_cut) + " end:" + str(cut))
                prev_cut = cut

            if len(cuts) == 0:
                cuts = [0]

            obj_data.uniqueId = obj_data.uniqueId.astype(int) # Why is this a float64?

            disambiguated_df_list.append(obj_data)

        disambiguated_df = pd.concat(disambiguated_df_list)
        disambiguated_df = disambiguated_df[disambiguated_df.uniqueId != DROP_INDEX]
        sys.stdout.write("\t\t\t\t%4s" % "[ OK ]")
        sys.stdout.write("\r\n")
        return disambiguated_df

    ############################################################################
    # labelling code below

    # The gates used to label will be different depending on the recording used.
    # The below code will use Leith-Croydon recording 2

    # I need to label this by entrances and exits.
    # I can then do a simplification for some labels as threat, safe etc.

    # So I want 6 gates
    # Boxes will be in metres, in format [X X Y Y]
    # For whatever reason, the forward back measurement seems to be X, and the other Y

    # This code differs to the RAV4 dataset as I do not want to trim the tracks based on the intersection box.
    # Instead, I only want to label them according to the gates, and calculate the distance from these gates.

    def _label_df(self, disambiguated_df):

        clean_tracks = []

        for uID in disambiguated_df.uniqueId.unique():
            # obj_data = vehicle_df[vehicle_df.ObjectId==ID]
            obj_data = disambiguated_df.loc[disambiguated_df.uniqueId == uID, :]
            #print("Sorting track: " + str(uID))
            if all(obj_data.trackedByStationaryModel):
                #print "Don't know what's going on here, but the data is incomplete for this track, skipping"
                continue
            sys.stdout.write("\rSorting track: %04d of %04d " % (uID,max(disambiguated_df.uniqueId.unique())))
            sys.stdout.flush()
            if len(obj_data) < 1:
                continue

            # I do want to be copying out my slice and adding them individually to a new collection, as this reduces size
            obj_data = obj_data.sort_values(['Timestamp'], ascending=True).reset_index()

            intersection_flag = False
            origin_label = None
            dest_label = None
            for time_idx in range(0, len(obj_data)):
                data = obj_data.iloc[time_idx]
                o_X = data["Object_X"]
                o_Y = data["Object_Y"]
                if intersection_flag == False:
                    for label, gate in self.origin_gates.iteritems():
                        if self._in_box([o_X, o_Y], gate):
                            origin_label = label
                            intersection_flag = True

                else:
                    for label, gate in self.dest_gates.iteritems():
                        if self._in_box([o_X, o_Y], gate):
                            # if label == origin_label:
                            #     #Skip U turns. HACK This is because in the first dataset, we have only 1 u-turn.
                            #     # This breaks the stratifier.
                            #     continue
                            dest_label = label

                if origin_label is not None and dest_label is not None:
                    # Stop checking, we have categorized this track
                    break

            if origin_label is None or dest_label is None:
                # If we never categorized this track, its garbage, skip this and check next track
                continue
            obj_data["origin"] = [origin_label] * len(obj_data)
            obj_data["destination"] = [dest_label] * len(obj_data)
            #print("ID: " + str(uID) + " Origin: " + origin_label + " Destination: " + dest_label)
            obj_data["relative_destination"] = [self._get_relative_exit(origin_label,dest_label)] * len(obj_data)
            obj_data = obj_data.assign(AbsVelocity=np.sqrt(np.power(obj_data['AbsVelocity_X'], 2)
                                                           + np.power(obj_data['AbsVelocity_Y'], 2)))

            clean_tracks.append(obj_data)
        sys.stdout.write(" Found %d clean tracks" % (len(clean_tracks)))
        sys.stdout.write("\t\t%4s" % "[ OK ]")
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        #print("Number of tracks in collection: " + str(len(clean_tracks)))

        return clean_tracks

    def _trim_tracks(self,long_tracks):
        #Intersection extent buffer.
        intersection_limits = 15
        dis_after_exit = 5
        trimmed_tracks = []

        intersection_xs = []
        intersection_ys = []
        for cardinal, box in self.dest_gates.iteritems():
            intersection_xs.extend(box[0:1])
            intersection_ys.extend(box[2:3])
        for cardinal, box in self.origin_gates.iteritems():
            intersection_xs.extend(box[0:1])
            intersection_ys.extend(box[2:3])
        int_x_max = max(intersection_xs)
        int_x_min = min(intersection_xs)
        int_y_max = max(intersection_ys)
        int_y_min = min(intersection_ys)

        for track in long_tracks:
            debug_track = track.copy()
            #Cut out cars that park if they are visible
            last_moving_idx = max(track[track['AbsVelocity'] > 0.1].index)
            last_observed_idx = max(track[track['ObjectPredAge'] == 0].index)
            # trim_track = track.iloc[0:last_moving_idx]
            # If they are outside the roundabout and have stopped i.e. parked
            track.drop(track[(track.index > last_moving_idx) & (
                             (track.Object_Y > int_y_max) |
                             (track.Object_Y < int_y_min) |
                             (track.Object_X > int_x_max) |
                             (track.Object_X < int_x_min))
                             ].index, inplace=True)
            # If the system is guessing
            track.drop(track[(track.index > last_observed_idx) & (
                             (track.Object_Y > int_y_max) |
                             (track.Object_Y < int_y_min) |
                             (track.Object_X > int_x_max) |
                             (track.Object_X < int_x_min))
                             ].index, inplace=True)
            # Or they are not in the roundabout proximity
            track.drop(track[(track.Object_Y > ( intersection_limits + int_y_max)) |
                             (track.Object_Y < (-intersection_limits + int_y_min)) |
                             (track.Object_X > ( intersection_limits + int_x_max)) |
                             (track.Object_X < (-intersection_limits + int_x_min))
                       ].index, inplace=True)
            # Or they have left the roundabout
            track.drop(track[(track.distance_to_exit > dis_after_exit)].index, inplace=True)
            #track.drop('level_0', axis=1,inplace=True)
            # And now trim everything that is not continuous around distance zero.
            track.reset_index(inplace=True)
            track.drop('level_0', axis=1, inplace=True)

            # and finally, check if any of the above filters have split a track in two. Keep the track that contains
            # distance zero
            cuts = np.where(np.diff(track.Timestamp) > 1)[0]
            if len(cuts) > 0:
                cuts += 1
                cuts = np.append(np.insert(cuts, 0, 0), len(track))
                for i in range(len(cuts)-1):
                    #check if zero distance is in this sequence
                    if not 0.0 in list(track.iloc[cuts[i]:cuts[i+1]].distance):
                        track.drop(range(cuts[i],cuts[i+1]),inplace=True)
                track.reset_index(inplace=True)
                track.drop('level_0', axis=1, inplace=True)

            if len(track) < 30:
                continue

            trimmed_tracks.append(track)
        return trimmed_tracks

    def _calculate_intersection_distance(self, labelled_track_list):
        base_idx = len(self.labelled_track_list)
        for track_idx in range(len(labelled_track_list)):
            single_track = labelled_track_list[track_idx]
            track_origin = single_track.iloc[0]['origin']
            track_dest = single_track.iloc[0]['destination']

            # At this point the tracks are ordered, so I should find the index where the object leaves its already
            # designated origin box.
            #print("Calculating distance metric for track: " + str(track_idx))
            sys.stdout.write("\rCalculating distance metric for track: %04d of %04d " % (base_idx + track_idx,
                                                                                         base_idx + len(labelled_track_list)))
            sys.stdout.flush()
            d = np.sqrt((single_track.loc[1:, "Object_X"].values -
                         single_track.loc[0:len(single_track)-2, "Object_X"].values) ** 2  # [0:len-2] is equiv. to [0:-1].
                        +
                        (single_track.loc[1:, "Object_Y"].values -
                         single_track.loc[0:len(single_track)-2, "Object_Y"].values) ** 2)
            #d = np.sqrt((single_track[1:, 0] - single_track[0:-1, 0]) ** 2 +
            #            (single_track[1:, 1] - single_track[0:-1, 1]) ** 2)
            d = np.cumsum(np.append(0.0, d))

            # Find the last point in which the car is still in the origin box.
            enter_ref_step = 0
            for step in range(len(single_track)):
                if self._in_box([single_track.iloc[step]["Object_X"],
                                 single_track.iloc[step]["Object_Y"]], self.origin_gates[track_origin]):
                    enter_ref_step = step
                    # Do not break as it will keep overriding with the latest point in box

            for step in range(len(single_track)):
                if self._in_box([single_track.iloc[step]["Object_X"],
                                 single_track.iloc[step]["Object_Y"]], self.dest_gates[track_dest]):
                    exit_ref_step = step
                    break  # I want the first step, not the last step.

            dis_from_enter = d - d[enter_ref_step]
            dis_from_exit = d - d[exit_ref_step]
            single_track["distance"] = dis_from_enter
            single_track["distance_to_exit"] = dis_from_exit

        sys.stdout.write("\t\t%4s" % "[ OK ]")
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        return labelled_track_list

        # Traversals:
        # iloc[]
        # ObjBoxCente_X
        # ObjBoxCenter_Y

        # dest_1_hot                                           [1.0, 0.0, 0.0]
        # destination                                                     east
        # destination_vec                                                    0
        # origin                                                          west
        # track_class                                                west-east
        # track_idx                                                          0
        # encoder_sample     [[-59.742669988, 1.02318042263, 69.2177803932,...
        # decoder_sample                                                    []
        # distance                                                    -39.3501
        # time_idx                                                           1

        # Above is a sample from the 'master pool'. The formatter will have to adjust for the destination label, as it needs to
        # be aware of the list of labels. Check with the label_generator. The encoder_sample is formatted according to a global,
        # namely
        # parameters["input_columns"] = ['easting', 'northing', 'heading', 'speed']
        # so the ibeoformatter will have to read this accordinly.
        # track_idx ~ uID
        # Distance will have to be generated at a per-track basis, and I will probably need to better define the intersection.
        #

    # Debug code below clipped from an ipython notebook. Non-functional right now.
    def _plot_tracks(self):
        df = pd.DataFrame();
        Nlines = 260
        from itertools import permutations

        color_lvl = 8
        rgb = np.array(list(permutations(range(0, 256, color_lvl), 3)))
        rgb_m = np.array(list(permutations(range(0, 256, color_lvl), 3))) / 255.0

        from random import sample

        colors = sample(rgb, Nlines)
        colors_m = sample(rgb_m, Nlines)

        # struct.pack('BBB',*rgb).encode('hex')

        # df['color'] = [tuple(colors[i]) for i in df.ObjectId]
        df['color'] = ["#" + struct.pack('BBB', *colors[i]).encode('hex') for i in df.ObjectId]
        df['color_m'] = [tuple(colors_m[i]) for i in df.ObjectId]

        fill_color = [color if flag else None for color, flag in zip(df['color'], df['mobile'])]

        from bokeh.plotting import figure
        from bokeh.layouts import layout, widgetbox
        from bokeh.models import ColumnDataSource, HoverTool, Div
        from bokeh.models.widgets import Slider, Select, TextInput
        from bokeh.io import curdoc, push_notebook
        from ipywidgets import interact
        import ipywidgets

        source_df = ColumnDataSource(data=dict(x=[], y=[]))
        slider_objectId = ipywidgets.IntSlider(value=2, min=min(df['ObjectId']), max=max(df['ObjectId']),
                                               step=1, description="Time", slider_color="red")
        p = figure(plot_width=800, plot_height=800)
        color = tuple(colors[1])
        # data = df[(df.ObjectId==4)&(df.Classification > 3)]
        p.circle(x='x', y='y', source=source_df, size=2, color=color)
        show(p, notebook_handle=True)

        def select_data(selected_ObjectId):
            return df[(df.ObjectId == selected_ObjectId) & (df.Classification > 3)]

        def update(ObjectID_value=slider_objectId):
            data = select_data(ObjectID_value)
            source_df.data = dict(x=data["Timestamp"], y=data["ObjectAge"])
            push_notebook()

        interact(update, selected_ObjectId=slider_objectId)
