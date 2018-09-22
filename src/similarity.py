import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys, getopt

from models.embed import embed

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
