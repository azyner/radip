from scipy.spatial import distance
import numpy as np
from numpy.core.umath_tests import inner1d

class ReportWriter:
    def __init__(self,
                 training_batch_handler,
                 validation_batch_handler,
                 test_batch_handler,
                 report_df):
        error = self._score_model_on_metric(report_df)
        ideas = None

    # Here, there are many options
    # A) metric variance. LCSS, Hausdorff, etc
    # B) Statistical variance:
        # best mean
        # best worst 5% / 1% / 0.1% <-- It took me ages to get data for a reasonable 0.1% fit!
    def _score_model_on_metric(self, report_df, metric=None):
        scores_list = []
        for track in report_df.iterrows():
            track_scores = {}
            track = track[1]

            preds = track.outputs[np.logical_not(track.trackwise_padding)]
            gts = track.decoder_sample[np.logical_not(track.trackwise_padding)]
            #TODO Trim tracks here based on trackwise padding.
            ### EUCLIDEAN ERROR -- Average
            euclid_error = []
            for pred, gt in zip(preds[:,0:2], gts[:,0:2]):
                # Iterates over each time-step
                euclid_error.append(distance.euclidean(pred, gt))

            ### MODIFIED HAUSDORFF DISTANCE
            # Pulled shamelessly from https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py
            # Thanks sapphire008!
            (A, B) = (preds[:, 0:2], gts[:, 0:2])
            # Find pairwise distance
            D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T +
                            inner1d(B, B) - 2 * (np.dot(A, B.T)))
            # Calculating the forward HD: mean(min(each col))
            FHD = np.mean(np.min(D_mat, axis=1))
            # Calculating the reverse HD: mean(min(each row))
            RHD = np.mean(np.min(D_mat, axis=0))
            # Calculating mhd
            MHD = np.max(np.array([FHD, RHD]))

            track_scores['euclidean'] = np.mean(np.array(euclid_error))
            track_scores['MHD'] = MHD


            scores_list.append(track_scores)
        return scores_list