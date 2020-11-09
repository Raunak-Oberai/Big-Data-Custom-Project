# Import libraries:
import pandas as pd
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api
from bert_serving.client import BertClient
import numpy

# Main Code to load and process the FAQ Dataset

df = pd.read_csv("faq_database.csv",header=0)

glove_model = None
try:
    glove_model = gensim.models.KeyedVectors.load("./glovemodel.mod")
    print("Loaded Glove Model")
except:
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")

v2w_model = None
try:
    v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
    print("Loaded W2Vec model")
except:
    v2w_model = api.load('word2vec-google-news-300')
    glove_model.save("./w2vecmodel.mod")

bc = BertClient()


# Function to return Answer to User's Question
def data(search):

    if len(search.split()) < 2:
        words_gl, words_vw = similar_words(search)
        answer = "Most Similar Words: "
        answer1 = str(words_gl).strip('[]')
        answer2 = str(words_vw).strip('[]')
        answer3 = "*BERT does not have similar word functionality*"
    else:
        answer = "Most Related Answer: "
        answer1, answer2, answer3 = faq_answer(search)

    return answer, answer1, answer2, answer3




# Function to clean dataset
def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

    if stopwords:
        sentence = remove_stopwords(sentence)

    return sentence


def get_cleaned_sentences(df, stopwords=False):
    sents = df[['Questions']]
    cleaned_sentences = []

    for index, row in df.iterrows():
        cleaned = clean_sentence(row["Questions"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences


#Function to Retrieve Answer
def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf,sentences):
    max_sim = -1
    index_sim = -1
    for index,faq_embedding in enumerate(sentence_embeddings):
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0]
        if sim>max_sim:
            max_sim = sim
            index_sim = index
    closest_ques = FAQdf.iloc[index_sim, 0]
    faq_answer = FAQdf.iloc[index_sim, 1]

    return closest_ques, faq_answer

#Function to create embeddings:
def getWordVec(word,model):
    samp = model['computer']
    vec = [0]*len(samp)
    try:
        vec = model[word]
    except:
        vec = [0]*len(samp)
    return(vec)

def getPhraseEmbedding(phrase,embeddingmodel):
    samp = getWordVec('computer',embeddingmodel)
    vec = numpy.array([0]*len(samp))
    den=0
    for word in phrase.split():
        den = den + 1
        vec = vec + numpy.array(getWordVec(word,embeddingmodel))
    return vec.reshape(1,-1)



#Function to return list of similar words from each model:
def similar_words(search):
    word = search
    words_gl = []
    words_vw = []
    for words,_ in glove_model.most_similar(word, topn=5):
        words_gl.append(words)
    for words, _ in v2w_model.most_similar(word, topn=5):
        words_vw.append(words)
    return words_gl,words_vw

#Function to Return the closest answer
def faq_answer(search):

    question_orig = search
    question = clean_sentence(question_orig, stopwords=True)

    # Answer using Glove Model:
    sent_embeddings = sent_embeddings_gl

    question_embedding = getPhraseEmbedding(question, glove_model)

    closest_ques_gl, faq_answer_gl = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, df,
                                                               cleaned_sentences)

    # Answer using Word2Vec Model:
    sent_embeddings = sent_embeddings_vw

    question_embedding = getPhraseEmbedding(question, v2w_model)

    closest_ques_vw, faq_answer_vw = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, df,
                                                               cleaned_sentences)

    # Answer using BERT:

    question = clean_sentence(question_orig, stopwords=False)

    sent_embeddings= sent_bertphrase_embedding

    question_embedding = bc.encode([question])

    closest_ques_bert, faq_answer_bert = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, df,
                                                                   cleaned_sentences)

    answer1 = "Question: " + closest_ques_gl + "             " + "Answer: " + faq_answer_gl
    answer2 = "Question: " + closest_ques_vw + "             " + "Answer: " + faq_answer_vw
    answer3 = "Question: " + closest_ques_bert + "             " + "Answer: " + faq_answer_bert

    return answer1, answer2, answer3

cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)


cleaned_sentences_bert = get_cleaned_sentences(df, stopwords=False)

sent_bertphrase_embedding = []

for sent in cleaned_sentences_bert:
    sent_bertphrase_embedding.append(bc.encode([sent]))

sent_embeddings_vw = []
for sent in cleaned_sentences:
    sent_embeddings_vw.append(getPhraseEmbedding(sent, v2w_model))

sent_embeddings_gl = []
for sent in cleaned_sentences:
    sent_embeddings_gl.append(getPhraseEmbedding(sent, glove_model))

print("Ready to Go!")