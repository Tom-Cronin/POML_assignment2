from unittest import TestCase
import Metrics as m

class Test(TestCase):

    def test_accuracy(self):
        label = [1, 0, 1, 0]
        preds = [1, 1, 0, 0]
        self.assertEquals(m.accuracy(preds, label), .5)

    def test_confusion_matrix(self):
        label = [1, 0, 1, 0]
        preds = [1, 1, 0, 0]
        confusion = [[1, 1], [1, 1]]
        self.assertEquals(m.confusion_matrix(preds, label), confusion)

    def test_precision(self):
        label = [1, 1, 0, 1]
        preds = [1, 1, 1, 1]
        self.assertEquals(m.precision(preds, label), .75)

    def test_recall(self):
        label = [1, 0, 0, 1]
        preds = [1, 0, 0, 0]
        self.assertEquals(m.recall(preds, label), .5)
    def test_f1_score(self):
        label = [0, 1, 0, 1]
        preds = [0, 1, 1, 1]
        self.assertEquals(m.f1_score(preds, label), .8)

