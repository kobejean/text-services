import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys, getopt

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
module_url = "universal_sencence_encoder_module/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.

health_embeddings = np.load('data/embed/health_embeddings.npy')
happy = health_embeddings[0]
sad = health_embeddings[1]
feminine = health_embeddings[2]
masculine = health_embeddings[3]
wealthy = health_embeddings[4]
poor = health_embeddings[5]
yin = health_embeddings[6]
yang = health_embeddings[7]
liberal = health_embeddings[8]
conservative = health_embeddings[9]
future = health_embeddings[10]
past = health_embeddings[11]
anger = health_embeddings[12]
peace = health_embeddings[13]

def similarity(a, b):
    return 1 - tf.losses.cosine_distance(a, b, axis=0)

def health_from_file(file_path):
    text = []
    with open(file_path) as file:
        for line in file:
            text.append(line)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        ave_embed = tf.reduce_mean(embed(text), 0)
        happy_stat = similarity(happy, ave_embed) - similarity(sad, ave_embed)
        feminine_stat = similarity(feminine, ave_embed) - similarity(masculine, ave_embed)
        wealthy_stat = similarity(wealthy, ave_embed) - similarity(poor, ave_embed)
        yin_stat = similarity(yin, ave_embed) - similarity(yang, ave_embed)
        liberal_stat = similarity(liberal, ave_embed) - similarity(conservative, ave_embed)
        future_stat = similarity(future, ave_embed) - similarity(past, ave_embed)
        anger_stat = similarity(anger, ave_embed) - similarity(peace, ave_embed)
        stat = tf.stack([happy_stat, feminine_stat, wealthy_stat, yin_stat, liberal_stat, future_stat, anger_stat])
        computed_stat = session.run(stat)
        print("happy feminine wealthy yin liberal future anger")
        print(computed_stat)
        return computed_stat



def main(argv):
    file_path = ''
    try:
        opts, args = getopt.getopt(argv,"hf:",["file="])
    except getopt.GetoptError:
        print('text_health.py -f')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('text_health.py -f <file>')
            sys.exit()
        elif opt in ("-f", "--file"):
            file_path = arg

    health_from_file(file_path)

if __name__ == "__main__":
    main(sys.argv[1:])
