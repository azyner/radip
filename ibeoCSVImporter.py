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
import os, pickle

class ibeoCSVImporter:
    def __init__(self, parameters, csv_name):
        self.unique_id_idx = int(1) #Made object global as disambig is called multiple times now.
        if isinstance(csv_name,str):
            csv_name = [csv_name]
        self.labelled_track_list = []
        # Check if I have cached this already
        # name it after the last csv in csv_name
        cache_name = csv_name[-1]
        file_path = 'data/' + cache_name + ".pkl"
        if not os.path.isfile(file_path):
            for csv_file in csv_name:
                print "Reading CSV " + csv_file
                input_df = pd.read_csv('data/' + csv_file)
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
                self.labelled_track_list.extend(trimmed_tracks)
                # write pkl
                with open(file_path, 'wb') as pkl_file:
                    pickle.dump(self.labelled_track_list, pkl_file)
        else:
            with open(file_path, 'wb') as pkl_file:
                self.labelled_track_list = pickle.load(pkl_file)

        self._print_collection_summary()

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
            self.origin_gates = {"north": top_enter, "east": right_enter, "south": low_enter}
        if  '20170601-stationary-3-leith-croydon' in csv_name:
            top_exit = [-25,-5,-0.5,0.5]
            top_enter = top_exit
            right_exit = [-4,-2,-16,-1]
            right_enter = right_exit
            left_exit = [-26,-25,-16,-2]
            left_enter = left_exit
            intersection_centre = [-14.4,-7.5]
            intersection_rotation = 0 # 90 degree?
            self.dest_gates = {"north": left_exit, "east": top_exit, "south": right_exit}
            self.origin_gates = {"north": left_enter, "east": top_enter, "south": right_enter}
        if 'stationary-4-leith-croydon' in csv_name:
            top_exit = [-25, -5, -0.5, 0.5]
            top_enter = top_exit
            right_exit = [-4, -2, -16, -1]
            right_enter = right_exit
            left_exit = [-26, -25, -16, -2]
            left_enter = left_exit
            intersection_centre = [-14.4, -7.5]
            intersection_rotation = 0  # 90 degree?
            self.dest_gates = {"north": left_exit, "east": top_exit, "south": right_exit}
            self.origin_gates = {"north": left_enter, "east": top_enter, "south": right_enter}

    def get_track_list(self):
        return self.labelled_track_list

    def _print_collection_summary(self):
        # Here I want to print a origin/destination matrix
        # Preferably with summary margins
        key_list = [key for key,value in self.dest_gates.iteritems()]

        summary_df = pd.DataFrame(np.zeros([len(key_list), len(key_list)]),columns=key_list,index=key_list)
        for single_track in self.labelled_track_list:
            summary_df.loc[single_track.iloc[0]["origin"],single_track.iloc[0]["destination"]] += 1
        summary_df["total"] = summary_df.sum(1)
        summary_df= summary_df.append(pd.Series(summary_df.sum(0), name="total"))

        print "origin | destination"
        print summary_df
        #TODO Write function that strips out any classes of size 1 as they break the stratitifer. Eg. rare u-turns.


    def _in_box(self, point, extent):
        """Return if a point is within a spatial extent."""
        return ((point[0] >= extent[0]) and
                (point[0] <= extent[1]) and
                (point[1] >= extent[2]) and
                (point[1] <= extent[3]))

    def _parse_ibeo_df(self, input_df):
        """ This function is used to clean up some of the many parameters inside the dataframe."""

        # Clean up the cs and add some labels
        input_df.Timestamp = input_df.Timestamp - input_df.iloc[0].Timestamp
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

    def _disambiguate_df(self,input_df):

        object_id_list = np.sort(input_df.ObjectId.unique())
        # Classes 4 and 5 are car, truck (maybe in that order)
        # 3 might be bike, have to check.
        # I only care about cars and trucks right now.

        vehicle_df = input_df.loc[input_df.Classification > 3, :]

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
            for cut in cuts:
                # print obj_data.loc[prev_cut:cut, :]
                obj_data.loc[prev_cut:cut, "uniqueId"] = self.unique_id_idx
                #print("ObjId:" + str(ID) + " UniqueId:" + str(unique_id_idx) +
                #      " Prev:" + str(prev_cut) + " end:" + str(cut))
                prev_cut = cut
                self.unique_id_idx += 1
            if len(cuts) == 0:
                cuts = [0]

            # Do this once more as there are more segments than there are cuts
            obj_data.loc[cuts[-1]:len(obj_data) - 1, "uniqueId"] = self.unique_id_idx
            self.unique_id_idx += 1

            obj_data.uniqueId = obj_data.uniqueId.astype(int) # Why is this a float64?
            disambiguated_df_list.append(obj_data)

        disambiguated_df = pd.concat(disambiguated_df_list)
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
            if any(obj_data.trackedByStationaryModel):
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
                            if label == origin_label:
                                #Skip U turns. HACK This is because in the first dataset, we have only 1 u-turn.
                                # This breaks the stratifier.
                                continue
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
            obj_data = obj_data.assign(AbsVelocity=np.sqrt(np.power(obj_data['AbsVelocity_X'],2)
                                                           + np.power(obj_data['AbsVelocity_Y'],2)))
            clean_tracks.append(obj_data)
        sys.stdout.write(" Found %d clean tracks" % (len(clean_tracks)))
        sys.stdout.write("\t\t%4s" % "[ OK ]")
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        #print("Number of tracks in collection: " + str(len(clean_tracks)))

        return clean_tracks

    def _trim_tracks(self,long_tracks):
        #Intersection extent buffer.
        buf = 40
        trimmed_tracks = []

        for track in long_tracks:
            debug_track = track.copy()
            #Cut out cars that park if they are visible
            last_moving_idx = max(track[track['AbsVelocity'] > 0.1].index)
            # trim_track = track.iloc[0:last_moving_idx]
            # If they are outside the roundabout and have stopped i.e. parked
            track.drop(track[(track.index>last_moving_idx) |
                             (track.Object_Y > (self.dest_gates['east'][3])) |
                             (track.Object_X > (self.dest_gates['south'][1])) |
                             (track.Object_X < (self.dest_gates['north'][0]))
                             ].index,inplace=True)
            # Or they have left the roundabout proximity
            track.drop(track[(track.Object_Y > (buf + self.dest_gates['east'][3])) |
                             (track.Object_X > (buf + self.dest_gates['south'][1])) |
                             (track.Object_X < (-buf + self.dest_gates['north'][0]))].index,inplace=True)
            #track.drop('level_0', axis=1,inplace=True)
            track.reset_index(inplace=True)
            if len(track)  < 20:
                print "WTF?"
            trimmed_tracks.append(track)
        return trimmed_tracks

    #TODO I want forward distance from entrance, and distance to exit.
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
