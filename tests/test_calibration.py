import unittest

from calibration import average_offsets, update_training


class CalibrationTests(unittest.TestCase):
    def test_repeated_same_correction_does_not_decay(self):
        history = {}
        raw = {'Nasion': (10, 20)}
        reviewed = {'Nasion': (15, 17)}
        for _ in range(3):
            update_training(history, raw, reviewed)
            self.assertEqual(average_offsets(history)['Nasion'], (5, -3))

    def test_independent_landmarks_keep_independent_offsets(self):
        history = {}
        update_training(history, {'A': (0, 0), 'B': (3, 3)}, {'A': (2, 4)})
        self.assertEqual(average_offsets(history), {'A': (2, 4), 'B': (0, 0)})


if __name__ == '__main__':
    unittest.main()
