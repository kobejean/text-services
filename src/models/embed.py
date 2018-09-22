import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
var_data_file = "../../use_module/\
                1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/\
                variables/variables.data-00000-of-00001"
if (os.path.isfile(var_data_file)):
    module_url = "../../use_module/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
