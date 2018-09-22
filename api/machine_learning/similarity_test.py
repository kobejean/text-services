from similarity import similarity as sim
from subprocess import call
import unittest

class TestSimilarity(unittest.TestCase):

    def test_similarity(self):
        diff = sim("Up", "Above") - sim("Up", "Below")
        self.assertTrue(diff, "Up is more similar to Below than Above.")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            sim(10, "Word")
        with self.assertRaises(TypeError):
            sim("None", None)
        with self.assertRaises(TypeError):
            sim([], [])

    def test_shell(self):
        ret = call("python3 similarity.py -a \"Hello\" -b \"Bye\"", shell=True)
        print(ret)
        self.assertTrue(ret != None, "")

if __name__ == '__main__':
    unittest.main()
