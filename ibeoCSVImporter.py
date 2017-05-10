import numpy as np
import scipy as sp
import random
import bokeh
import csv
import pandas as pd
import struct
from bokeh.plotting import figure, show
from bokeh.io import output_notebook


class ibeoCSVImporter:
    def __init__(self, parameters, csv_name):
        # df = pd.read_csv('long.csv')
        # sf = pd.read_csv('short.csv')
        # d1 = pd.read_csv('test1.csv')
        # d2 = pd.read_csv('test2.csv')
        # d3 = pd.read_csv('test3.csv') #This is the one I have notes for
        # d4 = pd.read_csv('test4.csv') #Driving Home

        # s1 = pd.read_csv('round1.csv')
        # s2 = pd.read_csv('round2.csv')
        # df = pd.read_csv('round3.csv')
        input_df = pd.read_csv(csv_name)
        if csv_name == 'data/20170427-stationary-2-leith-croydon.csv':
            top_exit = [-33, -30, 3, 4]
            top_enter = [-23, -20, 6, 7]
            right_exit = [-16, -15, -3, 2]
            right_enter = [-16, -15, -12, -8]
            low_exit = [-23, -21, -19, -18]
            low_enter = [-33, -30, -16, -15]
        self.dest_gates = {"north": top_exit, "east": right_exit, "south": low_exit}
        self.origin_gates = {"north": top_enter, "east": right_enter, "south": low_enter}


        # Clean up the cs and add some labels
        input_df.Timestamp = input_df.Timestamp - input_df.iloc[0].Timestamp

        # motionModelValidated is always true
        input_df.trackedByStationaryModel = (input_df.trackedByStationaryModel == 1)
        input_df.mobile = (input_df.mobile == 1)
        input_df['ObjPrediction'] = (input_df.ObjectPredAge > 0)
        disambiguated_df = self._disambiguate_df(input_df)
        self._label_df(disambiguated_df)

    def _disambiguate_df(self,input_df):
        # ts_diff = np.diff(df[(df.ObjectId == 17) & (df.Classification > 3)].Timestamp)
        # print max(ts_diff)
        # print min(ts_diff)
        # print np.average(ts_diff)

        object_id_list = input_df.ObjectId.unique()
        # Classes 4 and 5 are car, truck (maybe in that order)
        # 3 might be bike, have to check.
        # I only care about cars and trucks right now.

        vehicle_df = input_df.loc[input_df.Classification > 3, :]

        unique_id_idx = int(1)
        disambiguated_df_list = []

        for ID in object_id_list:
            obj_data = vehicle_df.loc[vehicle_df.ObjectId == ID, :]

            # Some objects have no data at all.
            if len(obj_data) < 5:
                continue

            obj_data = obj_data.reset_index() # Create copy, and reset the index to be contiguous
            obj_data = obj_data.sort_values(['Timestamp'], ascending=True)

            # Diff in its current format gives an off-by one error when passed to my splitter function.
            # insert/delete used to shift the whole array over by one.
            obj_diff = np.diff(obj_data['Timestamp'])
            obj_diff = np.insert(np.delete(obj_diff, -1), [0], [0])
            cuts = np.where(obj_diff > 1)[0]

            prev_cut = 0
            for cut in cuts:
                # print obj_data.loc[prev_cut:cut, :]
                obj_data.loc[prev_cut:cut, "uniqueId"] = unique_id_idx
                print("ObjId:" + str(ID) + " UniqueId:" + str(unique_id_idx) +
                      " Prev:" + str(prev_cut) + " end:" + str(cut))
                prev_cut = cut
                unique_id_idx += 1
            if len(cuts) == 0:
                cuts = [0]

            # Do this once more as there are more segments than there are cuts
            obj_data.loc[cuts[-1]:len(obj_data) - 1, "uniqueId"] = unique_id_idx
            unique_id_idx += 1

            obj_data.uniqueId = obj_data.uniqueId.astype(int) # Why is this a float64?
            disambiguated_df_list.append(obj_data)

        disambiguated_df = pd.concat(disambiguated_df_list)
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

    def _label_df(self, disambiguated_df):

        def in_box(point, extent):
            """Return if a point is within a spatial extent."""
            return ((point[0] >= extent[0]) and
                    (point[0] <= extent[1]) and
                    (point[1] >= extent[2]) and
                    (point[1] <= extent[3]))

        clean_tracks = []

        for uID in disambiguated_df.uniqueId.unique():
            # obj_data = vehicle_df[vehicle_df.ObjectId==ID]
            obj_data = disambiguated_df.loc[disambiguated_df.uniqueId == uID, :]
            print("Sorting track: " + str(uID))
            if len(obj_data) < 1:
                continue

            # I do want to be copying out my slice and adding them individually to a new collection, as this reduces size
            obj_data = obj_data.sort_values(['Timestamp'], ascending=True)

            intersection_flag = False
            origin_label = None
            dest_label = None
            for time_idx in range(0, len(obj_data)):
                data = obj_data.iloc[time_idx]
                o_X = data["ObjBoxCente_X"]
                o_Y = data["ObjBoxCenter_Y"]
                if intersection_flag == False:
                    for label, gate in self.origin_gates.iteritems():
                        if in_box([o_X, o_Y], gate):
                            origin_label = label
                            intersection_flag = True

                else:
                    for label, gate in self.dest_gates.iteritems():
                        if in_box([o_X, o_Y], gate):
                            dest_label = label

                if origin_label is not None and dest_label is not None:
                    # Stop checking, we have categorized this track
                    break

            if origin_label is None or dest_label is None:
                # If we never categorized this track, its garbage
                continue

            obj_data["origin"] = [origin_label] * len(obj_data)
            obj_data["destination"] = [dest_label] * len(obj_data)

            print("ID: " + str(uID) + " Origin: " + origin_label + " Destination: " + dest_label)
            clean_tracks.append(obj_data)

        print("Number of tracks in collection: " + str(len(clean_tracks)))

        self.labelled_df = pd.concat(clean_tracks)

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