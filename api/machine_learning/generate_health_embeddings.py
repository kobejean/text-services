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
# messages = [happy, sad, feminine, masculine, wealthy, poor, yin, yang, liberal,
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
messages = [happy, sad, feminine, masculine, wealthy, poor, yin, yang, liberal,
            conservative, future, past, anger, peace]


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed(messages))

    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join(
            (str(x) for x in message_embedding[:3]))
        print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

    np.save("data/health_embeddings", message_embeddings)
