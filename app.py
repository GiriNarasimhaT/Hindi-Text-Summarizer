import imp
from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import networkx as nx
from nltk.cluster.util import cosine_distance
import math
import nltk
from nltk.tokenize import SpaceTokenizer
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt')




def remove_redundant_sentences(sentences):
    cleaned=[]
    for s in sentences:
        if s in cleaned or s.strip()=='':
            continue
        else:
            cleaned.append(s)
    return cleaned

def clean_corpus(corpus):
    corpus=corpus.replace('।','.')
    corpus=corpus.replace('\xa0','')
    corpus=corpus.replace('\n','')
    corpus=corpus.replace('\r','')
    return corpus

def get_clean_sentences(doc):
    cleaned_doc=clean_corpus(doc)
    sentences=cleaned_doc.split('.')
    sentences=remove_redundant_sentences(sentences)
    return sentences

stopwords=pd.read_csv('final_stopwords.txt')
stopwords=list(stopwords[stopwords])

# Method I : Text summarization using Term frequency scores of sentences

def create_frequency_table(sentences):
    word_freq={}
    for sentence in sentences:
        words=SpaceTokenizer().tokenize(sentence)
        for word in words:
            if word in stopwords:
                continue
            if word in word_freq:
                word_freq[word]+=1
            else:
                word_freq[word]=1
    return word_freq

def score_sentences(sentences,word_freq):
    sentenceValue={}
    for sentence in sentences:
        words_in_sentence=SpaceTokenizer().tokenize(sentence)
        word_count=len(words_in_sentence)
        for word in word_freq:
            if word in words_in_sentence:
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]]+=word_freq[word]
                else:
                    sentenceValue[sentence[:10]]=word_freq[word]
        sentenceValue[sentence[:10]]=sentenceValue[sentence[:10]]/word_count
    return sentenceValue
        

def average_sentence_value(sentenceValues):
    sum_values=0
    for sent_id in sentenceValues:
        sum_values+=sentenceValues[sent_id]
    sent_count=len(sentenceValues)
    return sum_values/sent_count

def generate_summary(avg_score,sentences,sentenceValue,identifier):
    sentence_count=0
    summary=''
    for sentence in sentences:
        if sentence[:identifier] in sentenceValue and sentenceValue[sentence[:identifier]]>=avg_score:
            summary+=sentence.strip()+"| "
            sentence_count+=1
    return summary,sentence_count

def summarise_term_frequency_sentence_weighing(sentences):
    word_freq_table=create_frequency_table(sentences)
    scores=score_sentences(sentences,word_freq_table)
    avg_score=average_sentence_value(scores)
    summary,sent_count=generate_summary(avg_score,sentences,scores,10)
    return summary

# Method II : Text Summarization using TF-IDF scores of sentences

def calc_term_frequency_sentence_wise(sentences):
    freq_matrix={}
    for sentence in sentences:
        words=SpaceTokenizer().tokenize(sentence)
        sent_freq_table={}
        for word in words:
            if word in stopwords:
                continue
            if word in sent_freq_table:
                sent_freq_table[word]+=1
            else:
                sent_freq_table[word]=1
        freq_matrix[sentence[:15]]=sent_freq_table
    return freq_matrix
            
def calc_tf_matrix(freq_matrix):
    for sentence,sent_freq_table in freq_matrix.items():
        tf_sent={}
        sent_len=len(sent_freq_table)
        for word in sent_freq_table:
            sent_freq_table[word]/=sent_len
    return freq_matrix

def calc_sentence_frequency(freq_matrix):
    sent_freq={}
    for sentence,freq_matrix_sent in freq_matrix.items():
        for word,count in freq_matrix_sent.items():
            if word in sent_freq:
                sent_freq[word]+=1
            else:
                sent_freq[word]=1
    return sent_freq

def calc_idf_score(total_sentences,sent_freq,freq_matrix):
    idf={}
    for sentence,freq_matrix_sent in freq_matrix.items():
        idf_sent={}
        for word in freq_matrix_sent:
            idf_sent[word]=math.log10(total_sentences / float(sent_freq[word]))
        idf[sentence]=idf_sent
    return idf
    
def calc_tf_idf_score(tf,idf):
    tf_idf={}
    for (sentence1,tf_sent),(sentence2,idf_sent) in zip(tf.items(),idf.items()):
        tf_idf_sent={}
        for (word1,tf_score),(word2,idf_score) in zip(tf_sent.items(),idf_sent.items()):
            tf_idf_sent[word1]=tf_score*idf_score
        tf_idf[sentence1]=tf_idf_sent
    return tf_idf

def calc_tf_idf_score_sentence_wise(tf_idf_matrix):
    tf_idf={}
    for sentence,tf_idf_sent in tf_idf_matrix.items():
        tf_idf_sentence=0
        for word,tf_idf_score in tf_idf_sent.items():
            tf_idf_sentence+=tf_idf_score
        tf_idf[sentence]=tf_idf_sentence
    return tf_idf

def get_tf_idf(sentences):
    sent_freq_matrix=calc_term_frequency_sentence_wise(sentences)
    freq_matrix=calc_tf_matrix(sent_freq_matrix)
    total_sentences=len(sentences)
    sent_freq=calc_sentence_frequency(freq_matrix)
    idf=calc_idf_score(total_sentences,sent_freq,freq_matrix)
    tf_idf_matrix=calc_tf_idf_score(freq_matrix,idf)
    tf_idf=calc_tf_idf_score_sentence_wise(tf_idf_matrix)
    return tf_idf

def summarise_tf_idf_sentence_weighting(sentences):
    tf_idf=get_tf_idf(sentences)
    sentences=remove_redundant_sentences(sentences)
    avg_tf_idf_score=average_sentence_value(tf_idf)
    tf_idf_summary,sent_count=generate_summary(avg_tf_idf_score,sentences,tf_idf,15)
    return tf_idf_summary

# Method III : Text Summarization using Bag of Words(BOW) and PageRank algorithm

def calculate_sentence_similarity(sentence1, sentence2):
    words1 = [word for word in nltk.word_tokenize(sentence1)]
    words2 = [word for word in nltk.word_tokenize(sentence2)]
    all_words = list(set(words1 + words2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for word in words1:
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1
    return 1 - cosine_distance(vector1, vector2)

def calculate_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
        similarity_matrix[i][j] = calculate_sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix
            
def summarize(clean_sentences, percentage):
    similarity_matrix = calculate_similarity_matrix(clean_sentences)
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    ordered_scores = sorted(((scores[i], score) for i, score in enumerate(clean_sentences)), reverse=True)
    number_of_sentences=int(len(clean_sentences))
    print(number_of_sentences)
    if percentage > 0:
        number_of_sentences = int(number_of_sentences * percentage)
    best_sentences = []
    for sentence in range(number_of_sentences):
        best_sentences.append(ordered_scores[sentence][1])
    return best_sentences, ordered_scores

def generate_summary_textrank(clean_sentences,best_sentences):
    sent_dict={}
    ordered_list_of_sentences=[]
    for sent in clean_sentences:
        if sent[:15] in sent_dict:
            pass
        else:
            sent_dict[sent[:15]]=sent
            ordered_list_of_sentences.append(sent)
    summary_text=""
    for sent in ordered_list_of_sentences:
        if sent in best_sentences:
            summary_text+=sent+". "
    return summary_text


# Generating summaries
def hindisummarizer(data):
    clean_sentences=get_clean_sentences(data)
    summary_tf_weight=summarise_term_frequency_sentence_weighing(clean_sentences)
    summary_tf_idf_weight=summarise_tf_idf_sentence_weighting(clean_sentences)
    best_sentences,textrank_scores=summarize(clean_sentences, 0.4)
    summary_textrank=generate_summary_textrank(clean_sentences,best_sentences)

    summaries=[]
    summaries.append(summary_tf_idf_weight)
    summaries.append(summary_tf_weight)
    summaries.append(summary_textrank)
    return summaries

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/project")
def project():
    return render_template('project.html')

@app.route("/project", methods=['POST','GET'])
def getSummary():
    body=request.form['data']
    technique = int(request.form['techniques'])
    
    result = hindisummarizer(body)

    if technique == 1:
        result1=result[0]
        return render_template('project.html',input=body,result=result1,method="TF-IDF")
    elif technique == 2:
        result1=result[1]
        return render_template('project.html',input=body,result=result1,method="Term Frequency")
    elif technique == 3:
        result1=result[2]
        return render_template('project.html',input=body,result=result1,method="BOW and Text Rank")
    else:
        return render_template('project.html')
    

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ =="__main__":
    app.run(host='0.0.0.0', port=5000)