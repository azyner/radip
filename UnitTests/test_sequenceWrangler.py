from unittest import TestCase
import SequenceWrangler
import numpy as np
import pandas as pd
import parameters

class TestSequenceWrangler(TestCase):

    #Test the function that creates all possible
    def test__rnn_data_np_trivial(self):
        #generate garbage data
        data = np.arange(10)
        data = data.reshape([5,2])  #Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(parameters,data,training=1.0,test=0,val=0)
        data_collection = dataWrangler._track_slicer(data, 5, 0,pd.DataFrame())
        self.assertEqual(len(data_collection),1)
        #self.fail()

    #Test the function that creates all possible
    def test__rnn_data_np_trivial2(self):
        #generate garbage data
        data = np.arange(10)
        data = data.reshape([5,2])  #Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(parameters,data,training=1.0,test=0,val=0)
        template = pd.DataFrame({'label': 'east','origin':'west'},index=[0])
        data_collection = dataWrangler._track_slicer(data, 4, 0,template)
        self.assertEqual(len(data_collection),2)
        self.assertEqual(data_collection.iloc[0]['label'], 'east')
        #self.fail()

    def test__rnn_data_np_track_too_short(self):
        # generate garbage data
        data = np.arange(10)
        data = data.reshape([5, 2])  # Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(parameters,data, training=1.0, test=0, val=0)
        with self.assertRaises(ValueError):
            encoder_data, decoder_data = dataWrangler._track_slicer(data, 6, 0,pd.DataFrame())


    def test__dis_from_ref_line(self):
        a = np.arange(-10,0)
        z = np.zeros(len(a))
        data = np.array([a,z]).transpose()
        dataWrangler = SequenceWrangler.SequenceWrangler(parameters,data, training=1.0, test=0, val=0)
        dis = dataWrangler._dis_from_ref_line(data,4)
        self.assertTrue((dis==[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]).all())

    def test__track_slicer_w_dis(self):
        a = np.arange(-10, 0)
        z = np.zeros(len(a))
        data = np.array([a, z]).transpose()
        dataWrangler = SequenceWrangler.SequenceWrangler(parameters,data, training=1.0, test=0, val=0)
        template = pd.DataFrame({'label': 'east','origin':'west'},index=[0])
        data_collection = dataWrangler._track_slicer(data, 4, 0,template,4)
        self.assertTrue((data_collection['distance'].values == [-4, -3, -2, -1, 0, 1, 2]).all())


