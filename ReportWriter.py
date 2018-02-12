from scipy.spatial import distance

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
        # best worst 5% / 1% / 0.1% <-- It tooke me ages to get dat for a reasonable 0.1% fit!
    def _score_model_on_metric(self, report_df, metric=None):
        error = 0
        for track in report_df.iterrows():
            track = track[1]
            preds = track.outputs
            gts = track.decoder_sample


            for pred, gt in zip(preds[:,0:2], gts[:,0:2]):
                error += distance.euclidean(pred, gt)
            ideas = None
        return error