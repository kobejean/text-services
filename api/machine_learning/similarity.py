import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys, getopt, os

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
var_data_file = "/universal_sencence_encoder_module/\
                1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/\
                variables/variables.data-00000-of-00001"
if (os.path.isfile(var_data_file)):
    module_url = "universal_sencence_encoder_module/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def similarity(astring, bstring):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    similarities = 1 - tf.losses.cosine_distance(embed([astring]), embed([bstring]), axis=1)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        computed_similarity = session.run(similarities)
        return computed_similarity

def main(argv):
    astring = ''
    bstring = ''
    try:
        opts, args = getopt.getopt(argv,"ha:b:",["astring=","bstring="])
    except getopt.GetoptError:
        print('similarity.py -a <string1> -b <string2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('similarity.py -a <string1> -b <string2>')
            sys.exit()
        elif opt in ("-a", "--astring"):
            astring = arg
        elif opt in ("-b", "--bstring"):
            bstring = arg

    print(similarity(astring, bstring))

if __name__ == "__main__":
    main(sys.argv[1:])
