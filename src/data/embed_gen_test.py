import unittest, os, sys
import numpy as np

from .embed_gen import embed_gen

script_dir = os.path.dirname(__file__)
test_file = "../../data/embed/test.npy"
test_file = os.path.join(script_dir, test_file)

class TestEmbedGen(unittest.TestCase):

    def test_embed_gen(self):
        if os.path.isfile(test_file):
            os.remove(test_file)

        test_messsge = ["This", "Is", "A", "Test"]
        embed_gen(test_messsge, test_file)

        file_created = os.path.isfile(test_file)
        np_load = np.load(test_file)

        self.assertTrue(file_created, "Failed to create .npy file at {}.".format(test_file))
        self.assertIsNotNone(np_load, "Failed to load .npy file at {}.".format(test_file))

        os.remove(test_file)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            embed_gen(10, test_file)
        with self.assertRaises(TypeError):
            embed_gen("None", test_file)
        with self.assertRaises(TypeError):
            embed_gen(["Test"], 10)

if __name__ == '__main__':
    unittest.main()
