from unittest import TestCase
import intersection_segments
import sys
import copy
import numpy as np
import pandas as pd
import SequenceWrangler
import BatchHandler
import parameters

class TestBatchHandler(TestCase):
    def test_get_minibatch(self):

        Wrangler = SequenceWrangler.SequenceWrangler(parameters)

        if not Wrangler.load_from_checkpoint():
            #This call copied from the dataset paper. It takes considerable time, so ensure it is run once only
            print "reading data"
            raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
            Wrangler.generate_master_pool_naturalistic_2015(raw_sequences, raw_classes)

        Wrangler.split_into_evaluation_pools()
        cf_pool, test_pool = Wrangler.get_pools()
        cf_fold = -1

        for train_pool, val_pool in cf_pool:
            cf_fold += 1
            training_batch_handler = BatchHandler.BatchHandler(train_pool,parameters.parameters,True)
            validation_batch_handler = BatchHandler.BatchHandler(val_pool,parameters.parameters,False)

            batch_frame = training_batch_handler.get_minibatch()
            train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(batch_frame['encoder_sample'],
                                                                                     batch_frame['dest_1_hot'],
                                                                                     batch_frame['padding'])

            if len(train_x[0]) != parameters.parameters['batch_size']:
                self.fail()

            # TEST padding condition when d_thresh is set
            if cf_fold == 0:
                # Get number of unique tracks in val pool
                num_val_tracks = len(val_pool['track_idx'].unique())
                validation_batch_handler.set_distance_threshold(d_thresh=-20)
                complete = False
                pad_array = []
                while not complete:
                    batch_frame, complete = validation_batch_handler.get_sequential_minibatch()
                    train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(
                        batch_frame['encoder_sample'],
                        batch_frame['dest_1_hot'],
                        batch_frame['padding'])
                    pad_array.extend(batch_frame['padding'])
                if (sum(np.array(weights[0]).astype(int))) != num_val_tracks:
                    self.fail()

            # TEST padding when d_thresh not set
            if cf_fold == 1:
                # Get number of unique tracks in val pool
                num_samples = len(val_pool)
                complete = False
                pad_array = []
                total_valid = 0
                while not complete:
                    batch_frame, complete = validation_batch_handler.get_sequential_minibatch()
                    train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(
                        batch_frame['encoder_sample'],
                        batch_frame['dest_1_hot'],
                        batch_frame['padding'])
                    total_valid += np.sum(np.logical_not(batch_frame['padding'])*1)
                    pad_array.extend(batch_frame['padding'])
                if total_valid != num_samples:
                    self.fail()

