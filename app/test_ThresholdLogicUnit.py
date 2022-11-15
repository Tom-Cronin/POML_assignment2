from unittest import TestCase
from ThresholdLogicUnit import ThresholdLogicUnit


class TestThresholdLogicUnit(TestCase):
    def test_initialise_weights_should_have_n_weights_to_n_inputs(self):
        tlu = ThresholdLogicUnit(learning_rate=0.001)
        inputs = [1,2,3]
        tlu.initialise_weights(inputs)
        print(len(tlu.weights))
        self.assertEquals(len(inputs), len(tlu.weights))

    def test_heaviside(self):
        tlu = ThresholdLogicUnit(learning_rate=0.001)
        self.assertEquals(tlu.heaviside(5), 1)
        self.assertEquals(tlu.heaviside(0), 1)
        self.assertEquals(tlu.heaviside(-1), 0)

    def test_relu(self):
        tlu = ThresholdLogicUnit(learning_rate=0.001)
        self.assertEquals(tlu.relu(5), 5)
        self.assertEquals(tlu.relu(0), 0)
        self.assertEquals(tlu.relu(-1), 0)

    def test_sigmoid(self):
        tlu = ThresholdLogicUnit(learning_rate=0.001)
        self.assertEquals(tlu.sigmoid(5), 1)
        self.assertEquals(tlu.sigmoid(0), 1)
        self.assertEquals(tlu.sigmoid(-1), 0)

