import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys, getopt

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
module_url = "universal_sencence_encoder_module/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# happy = "The world is a happy, wonderful place of heaven."
# sad = "The world is a sad, dreadful place of hell."
# feminine = "The world is feminine, motherly and a place for women."
# masculine = "The world is masculine, fatherly and a place for men."
# wealthy = "The world is rich, wealthy and full of prosperity."
# poor = "The world is poor, deprived and full of poverty."
# yin = "The world is yin a symbol of earth, femaleness, darkness, passivity, and absorption."
# yang = "The world is yang a symbol of heaven, maleness, brightness, activity, and penetration."
# liberal = "The world should be more liberal and open."
# conservative = "The world should be more conservative and traditional."
# future = "The world is new and the future is good."
# past = "The world is old and the past is good."
# anger = "The world is angry and a place of hate."
# peace = "The world is peaceful and a place of love."
# health_strs = [happy, sad, feminine, masculine, wealthy, poor, yin, yang, liberal,
#             conservative, future, past, anger, peace]

happy = "happy"
sad = "sad"
feminine = "feminine"
masculine = "masculine"
wealthy = "wealthy"
poor = "poor"
yin = "yin"
yang = "yang"
liberal = "liberal"
conservative = "conservative"
future = "future"
past = "past"
anger = "anger"
peace = "peace"
health_strs = [happy, sad, feminine, masculine, wealthy, poor, yin, yang, liberal,
            conservative, future, past, anger, peace]

synaesthesia_str = ["red", "orange", "yellow", "green", "blue", "purple"
                    "black", "white", "grey"
                    "circle", "square", "triangle"]

def generate_embeddings(messages, file_path):

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(messages))

        for i, embedding in enumerate(np.array(embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(embedding)))
            embedding_snippet = ", ".join(
                (str(x) for x in embedding[:3]))
            print("Embedding: [{}, ...]\n".format(embedding_snippet))

        np.save(file_path, embeddings)

generate_embeddings(health_strs, "data/embedding/health")
generate_embeddings(synaesthesia_str, "data/embedding/synaesthesia")
