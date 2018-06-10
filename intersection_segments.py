# Import notebook utilities (helper functions to reduce verbosity of script)
import pylab
import sys
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict

def in_box(point, extent):
    """Return if a point is within a spatial extent."""
    return ((point[0] >= extent[0]) and
            (point[0] <= extent[1]) and
            (point[1] >= extent[2]) and
            (point[1] <= extent[3]))

def manoeuvre_bounds(intersection_extent, terminal_dims):
    """Define extents to locate segment start/end."""

    # Define extent for segments starting/ending in the west.
    west = [intersection_extent[0], intersection_extent[0] + terminal_dims[0],
            -terminal_dims[1], terminal_dims[1]]

    # Define extent for segments starting/ending in the east.
    east = [intersection_extent[1] - terminal_dims[0], intersection_extent[1],
            -terminal_dims[1], terminal_dims[1]]

    # Define extent for segments starting/ending in the south.
    south = [-terminal_dims[1], terminal_dims[1],
             intersection_extent[2], intersection_extent[2] + terminal_dims[0]]

    return west, east, south

def transform(x, y, translation=None, rotation=None):
    # type: (object, object, object, object) -> object
    """Translate and rotate points."""

    # Translate points.
    if translation is not None:
        x = x + translation[0]
        y = y + translation[1]

    # Rotate points.
    if rotation is not None:
        phi = rotation * (np.pi/180)
        xt = (x * np.cos(phi) - y * np.sin(phi))
        yt = (x * np.sin(phi) + y * np.cos(phi))
        x = xt
        y = yt

    return x, y

def get_intersection_segments(eastings, northings, dataset,
                              intersection, rotation,
                              intersection_extent, terminal_dims,
                              verbose=False):
    """Retrieve driving segments which enter 'intersection'."""

    # Get unique data sets.
    unique_datasets = OrderedDict()
    for i, d in enumerate(np.unique(dataset)):
        unique_datasets[d] = i

    # Shift intersection to the origin.
    intersection = -np.array(intersection)

    # Rotate intersection about the origin so the T-intersection is axis
    # aligned.
    x, y = transform(eastings, northings, intersection, rotation)

    # Define extents for segment start/end.
    west, east, south = manoeuvre_bounds(intersection_extent, terminal_dims)

    # Define list of possible manoeuvre categories for T-intersection.
    categories = [[west,  east,  'west-east'],
                  [west,  south, 'west-south'],
                  [east,  west,  'east-west'],
                  [east,  south, 'east-south'],
                  [south, west,  'south-west'],
                  [south, east,  'south-east']]

    # Find values in the intersection.
    idx = np.logical_and(np.logical_and(x > intersection_extent[0],
                                        x < intersection_extent[1]),
                         np.logical_and(y > intersection_extent[2],
                                        y < intersection_extent[3]))

    # Pre-allocate objects for containing segments.
    k = 0
    manoeuvres = OrderedDict()
    for category in categories:
        manoeuvres[category[2]] = list()

    # Pre-allocate memory for counting segments.
    num_datasets = len(unique_datasets)
    segment_matrix = np.zeros((len(categories), num_datasets))

    # Find intersection segments.
    in_intersection = False
    for i, intersection_flag in enumerate(idx):

        # Intersection segment has started.
        if intersection_flag and not in_intersection:
            start_idx = i
            in_intersection = True

        # Intersection segment has ended.
        elif not intersection_flag and in_intersection:
            in_intersection = False
            end_idx = i - 1

            # Create index for segment. Note that the intersection ended one
            # observation previously. This is taken into in the range function.
            index = np.arange(start_idx, i)

            # Create starting and ending points for segment.
            start_point = [x[start_idx], y[start_idx]]
            end_point = [x[end_idx], y[end_idx]]

            # Iterate through categories
            for j, (start, end, name) in enumerate(categories):
                if (in_box(start_point, start) and
                    in_box(end_point, end)):
                    k += 1

                    # Calculate the distance between pairs of points in
                    # segment.
                    d = np.sqrt((x[index][1:] - x[index][0:-1])**2 +
                                (y[index][1:] - y[index][0:-1])**2)
                    d = np.cumsum(np.append(0.0, d))

                    # Store segment data.
                    dataset_ID = unique_datasets[dataset[start_idx]]
                    manoeuvres[name].append({'ID': k,
                                             'index': index,
                                             'distance': d,
                                             'dataset': dataset[start_idx],
                                             'dataset_ID': dataset_ID
                                            })

                    # Record counts of segment origin.
                    segment_matrix[j, dataset_ID] += 1

                    break

    # Summarise data.
    if verbose:
        print 'Summary:'
        for i, (key, value) in enumerate(manoeuvres.iteritems()):
            sys.stdout.write('    {0:>10}: '.format(key))
            for j in range(num_datasets):
                sys.stdout.write('{0:>3n} | '.format(segment_matrix[i, j]))
            print '{0:>3n} |'.format(segment_matrix[i, :].sum())
        print '    -----------' + '-' * 6 * (num_datasets + 1)
        sys.stdout.write('    {0:>10}: '.format('Total'))
        for j in range(num_datasets):
            sys.stdout.write('{0:>3n} | '.format(segment_matrix[:, j].sum()))
        print '{0:>3n} |'.format(segment_matrix.sum())

    return manoeuvres

def load_filtered_position_messages(fname=None):
    """Load filtered position messages from CSV file."""

    if fname is None:
        fname = ('data/20150323T041228_ivssg-2_EKF_All_GNSS.csv.gz',
                 'data/20150414T041122_ivssg-2_EKF_All_GNSS.csv.gz',
                 'data/20150423T032510_ivssg-2_EKF_ALL_GNSS.csv.gz')

    frame = pd.DataFrame()

    for filename in fname:
        df = pd.io.parsers.read_csv(filename, compression='gzip')
        df.columns = [name.strip() for name in list(df.columns.values)]

        # Force heading to be [-pi, pi].
        heading = np.pi * df['heading'].values / 180.0
        heading = np.arctan2(np.sin(heading), np.cos(heading)) * 180.0 / np.pi
        df.heading = heading
        df['dataset'] = filename

        frame = pd.concat([frame, df], ignore_index=True)

    return frame

def get_manouvre_sequences(columns,extent=60):
    # Use higher resolution, in-line images.
    # %pylab inline
    pylab.rcParams['savefig.dpi'] = 150

    # Format plots for LaTeX.
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 11,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    # plt.rcParams.update(params)

    intersection = [332387, 6248042]  # Location of T-intersection
    intersection_orientation = -21  # Orientation of T-intersection (in degrees)
    intersection_extent = [-extent, extent, -extent, extent]  # Define extent of intersection
    #intersection_extent = [-90, 90, -90, 90]  # Define extent of intersection
    terminal_dims = [2, 10]  # Width and height of terminal nodes
    resolution = 1.0  # Spacing between models (in metres)
    states = ['easting', 'northing', 'heading', 'speed']  # States used in models.

    # url = "https://maps.google.com/maps?q=-33.89471+151.187335&z=20&t=k&output=embed"
    # display(IFrame(url, '800', '600'))
    #
    # # Load driving data as Pandas object.
    fpm = load_filtered_position_messages()
    # display(fpm.loc[:, ['timestamp', 'easting', 'northing', 'heading', 'speed']])

    # Plot driving data and intersection.
    # Mask the values that have large timestamp jumps
    import numpy.ma as ma

    manoeuvres = get_intersection_segments(fpm['easting'].values,
                                                 fpm['northing'].values,
                                                 fpm['dataset'].values,
                                                 intersection, intersection_orientation,
                                                 intersection_extent,
                                                 terminal_dims)

    # del manoeuvres['east-south']
    # del manoeuvres['east-west']
    # del manoeuvres['west-south']
    # del manoeuvres['west-east']

    # Shift intersection to the origin.
    intersection = -np.array(intersection)

    track_collection = []
    class_collection = []

    for name, manoeuvre_segments in manoeuvres.iteritems():
        eastings = fpm['easting'].values
        northings = fpm['northing'].values
        timestamps = fpm['timestamp'].values
        speeds = fpm['speed'].values
        headings = fpm['heading'].values
        manoeuvres = manoeuvre_segments
        rotation = intersection_orientation
        extent = intersection_extent
        cols = 6

        # Rotate intersection about the origin so the T-intersection is axis
        # aligned.
        eastings, northings = transform(eastings, northings,
                                              intersection, rotation)

        # for h in range(0,len(headings)):
        headings -= rotation

        headings += rotation
        for element in np.nditer(headings, op_flags=['readwrite']):
            while element > 360:
                element -= 360
            while element <= 0:
                element += 360

        # normalize timestamps


        # Calculate number of rows and columns in plot.
        num_manoeuvres = len(manoeuvres)

        # CSV WRITE timestamp, heading. speed, easting northing

        # i is the iterator for all ~30 manoeuvres in a segment
        for i in range(num_manoeuvres):
            indices = manoeuvres[i]['index']
            dataset_ID = manoeuvres[i]['dataset_ID']
            distance = manoeuvres[i]['distance'][-1]

            file_timestamps = timestamps[indices]
            file_headings = headings[indices]
            file_speeds = speeds[indices]
            file_eastings = eastings[indices]
            file_northings = northings[indices]

            # Scrub timestamp data here
            start_time = file_timestamps[0]
            for element in np.nditer(file_timestamps, op_flags=['readwrite']):
                element -= start_time
            d = {'easting': eastings[indices],
                 'northing': northings[indices],
                 'speed': speeds[indices],
                 'heading': headings[indices],
                 'timestamp': timestamps[indices]}
            df = pd.DataFrame(d)
            track_collection.append(df.as_matrix(columns))
            class_collection.append(name)

    #Need to convert from data_frame to np array

    return track_collection, class_collection

