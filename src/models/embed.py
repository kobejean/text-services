import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
script_dir = os.path.dirname(__file__)
var_data_file = "../../models/embed/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables.data-00000-of-00001"
var_data_file = os.path.join(script_dir, var_data_file)
if (os.path.isfile(var_data_file)):
    module_url = "../../models/embed/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"
    module_url = os.path.join(script_dir, module_url)

print("loading module from {}".format(module_url))

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
