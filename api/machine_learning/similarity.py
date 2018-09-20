import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys, getopt

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
module_url = "universal_sencence_encoder_module/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# # Import the Universal Sentence Encoder's TF Hub module
# embed = hub.Module(module_url)
#
# # Compute a representation for each message, showing various lengths supported.
# word = "Elephant"
# sentence = "I am a sentence for which I would like to get its embedding."
# paragraph = (
#     "Universal Sentence Encoder embeddings also support short paragraphs. "
#     "There is no hard limit on how long the paragraph is. Roughly, the longer "
#     "the more 'diluted' the embedding will be.")
# messages = [word, sentence, paragraph]
#
#
# # Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)

# with tf.Session() as session:
#   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#   message_embeddings = session.run(embed(messages))
#
#   for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#     print("Message: {}".format(messages[i]))
#     print("Embedding size: {}".format(len(message_embedding)))
#     message_embedding_snippet = ", ".join(
#         (str(x) for x in message_embedding[:3]))
#     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

def similarity(astring, bstring):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    similarities = 1 - tf.losses.cosine_distance(embed([astring]), embed([bstring]), axis=1)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        computed_similarity = session.run(similarities)
        print("Similarity: {}\n".format(computed_similarity))
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

    similarity(astring, bstring)

if __name__ == "__main__":
    main(sys.argv[1:])
