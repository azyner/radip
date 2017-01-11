from unittest import TestCase
import SequenceWrangler
import numpy as np
import pandas as pd

class TestSequenceWrangler(TestCase):

    #Test the function that creates all possible
    def test__rnn_data_np_trivial(self):
        #generate garbage data
        data = np.arange(10)
        data = data.reshape([5,2])  #Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(data,training=1.0,test=0,val=0)
        encoder_data, decoder_data = dataWrangler._track_slicer(data, 5, 0)
        self.assertEqual(len(encoder_data),1)
        #self.fail()

    #Test the function that creates all possible
    def test__rnn_data_np_trivial2(self):
        #generate garbage data
        data = np.arange(10)
        data = data.reshape([5,2])  #Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(data,training=1.0,test=0,val=0)
        encoder_data, decoder_data = dataWrangler._track_slicer(data, 4, 0)
        self.assertEqual(len(encoder_data),2)
        #self.fail()

    def test__rnn_data_np_track_too_short(self):
        # generate garbage data
        data = np.arange(10)
        data = data.reshape([5, 2])  # Data is now 5 units long, of data vectors of length 2
        dataWrangler = SequenceWrangler.SequenceWrangler(data, training=1.0, test=0, val=0)
        with self.assertRaises(ValueError):
            encoder_data, decoder_data = dataWrangler._track_slicer(data, 6, 0)


    def test__dis_from_ref_line(self):
        a = np.arange(-10,0)
        z = np.zeros(len(a))
        data = np.array([a,z]).transpose()
        dataWrangler = SequenceWrangler.SequenceWrangler(data, training=1.0, test=0, val=0)
        dis = dataWrangler._dis_from_ref_line(data,4)
        self.assertTrue((dis==[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]).all())

