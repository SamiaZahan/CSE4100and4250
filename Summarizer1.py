# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:12:19 2020

@author: nowshad
"""
#import heapq
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import rouge
from yellowbrick.cluster import KElbowVisualizer


def preprocessing(proc_sentences):
    #Filtering the data for if-idf
    for i in range(len(proc_sentences)):
        proc_sentences[i] = re.sub(r"\W"," ",proc_sentences[i])
        proc_sentences[i] = re.sub(r"\d"," ",proc_sentences[i])
        proc_sentences[i] = re.sub(r"\s+[a-z]\s+"," ",proc_sentences[i],flags=re.I)
        proc_sentences[i] = re.sub(r"\s+"," ",proc_sentences[i])
        proc_sentences[i] = re.sub(r"^\s","",proc_sentences[i])
        proc_sentences[i] = re.sub(r"\s$","",proc_sentences[i])
        proc_sentences[i] = re.sub(r"[^a-zA-Z0-9 -]","",proc_sentences[i])
        proc_sentences[i] = proc_sentences[i].lower()
        
    lemmatizer = WordNetLemmatizer()

    # Lemmatization
    for i in range(len(proc_sentences)):
        words = nltk.word_tokenize(proc_sentences[i])
        words = [lemmatizer.lemmatize(word) for word in words]
        proc_sentences[i] = ' '.join(words)               

    return proc_sentences    

def tf_idf_scores(proc_sentences):
    # Creating the BOW model
    vectorizer = CountVectorizer(max_features = 2000, min_df = 1, max_df = 1, stop_words = stopwords.words('english'))
    X = vectorizer.fit_transform(proc_sentences).toarray()
    
    # Creating the Tf-Idf Model
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X).toarray()
    return X

def sent_weighting(tf_idf):
    size = tf_idf.shape
    sum_All_tfidf=0
    for i in range(size[0]):
        sum_All_tfidf += sum(tf_idf[i])
    sent_weights = []
    for j in range(size[0]):
        sent_weights.append(sum(tf_idf[j])/sum_All_tfidf)
    return sent_weights

def Elbow_Method(df):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,df.shape[0]))

    visualizer.fit(df)        
    visualizer.show()
    
    
def Silhouette_Method(df):
    from sklearn.metrics import silhouette_score
    no_of_clusters = df.shape[0]

    for n_clusters in range (no_of_clusters): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster.fit(df)
        cluster_labels = cluster.labels_.tolist() 

    	# The silhouette_score gives the 
        #verage value for all the samples. 
        silhouette_avg = silhouette_score(df, cluster_labels) 

        print("For no of clusters =", n_clusters, 
		" The average silhouette_score is :", silhouette_avg) 


def getBestCluster(X,_min,_max):
	selected_cluster = 0
	previous_sil_coeff = 0.001 #some random small number not 0
	sc_vals = []
	for n_cluster in range(_min, _max):
	    kmeans = KMeans(n_clusters=n_cluster).fit(X)
	    label = kmeans.labels_

	    sil_coeff = silhouette_score(X, label, metric='euclidean', sample_size=10)
	    sc_vals.append(sil_coeff)
	    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

	    percent_change = (sil_coeff-previous_sil_coeff)*100/previous_sil_coeff

	    # return when below a threshold of 1%
	    if percent_change<1:
	    	selected_cluster = n_cluster-1

	    previous_sil_coeff = sil_coeff

	return selected_cluster or _max, sc_vals 





def KMean_Clustering(df):
    num_clusters = int(input("Enter Elbow point form graph: ")) #Change it according to your data.
    km = KMeans(n_clusters=num_clusters)
    km.fit(df)
    clusters = km.labels_.tolist()
    
    df["Clus_km"] = clusters
    result = df.groupby('Clus_km').size()
    print(result)    

    return clusters

def extract_summary(sentences, clusters):
    cluster = int(input("Select the cluster which have most frequency: "))
    summary = ""
    for i in range(len(sentences)):
        if clusters[i] == cluster:
            summary += sentences[i]
            summary += " "
    return summary




#evaluation start

def prepare_results(metric,p, r, f):
    return '{}:   {}: {:5.2f}   {}: {:5.2f}   {}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def evaluation(summary,reference):
    for aggregator in ['Avg', 'Best', 'Individual']:
        #print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=3,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=1, # Default F1_score
                                weight_factor=1.2,
                                stemming=True)


        hypothesis_1 = summary
        references_1 = [reference,]

     #   hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
     #   references_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

        all_hypothesis = [hypothesis_1]
        all_references = [references_1]

        scores = evaluator.get_scores(all_hypothesis, all_references)
        #print(scores)
        precision = []
        recall= []
        F1_scores = []
        
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        #print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print(prepare_results(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                        precision.append(results_per_ref['p'][reference_id])
                        recall.append(results_per_ref['r'][reference_id])
                        F1_scores.append(results_per_ref['f'][reference_id])
                #print()
            else:
                pass
            #print(prepare_results(results['p'], results['r'], results['f']))
        #print()
    
    
    Evaloation_Graph(precision,recall,F1_scores)    
    
#end evaluation    

def Evaloation_Graph(precision,recall,F1_scores):
    x = np.arange(5)
    plt.figure(figsize = (10,8))
    rouge_ = "ROUGE_1","ROUGE_2","ROUGE_3","ROUGE_l","ROUGE_w"
    plt.bar(x+0.00,precision,color='b',width=0.25,label="Precision")
    plt.bar(x+0.25,recall,color='r',width=0.25,label="Recall")
    plt.bar(x+0.50,F1_scores,color='g',width=0.25,label="F1 scores")
    plt.title("Evaluation",fontsize=15)
    plt.ylabel("scores")
    plt.xticks(x,rouge_)
    plt.legend(fontsize=10, loc=(1.0,0.07))
    plt.show()








    
#Main
filename=input("Enter File Name: ")        
news = open("Dataset/News Articles/business/"+filename,"r")
text = news.read()

#Tokenization
sentences = nltk.sent_tokenize(text)
words = nltk.word_tokenize(text)

proc_sentences = nltk.sent_tokenize(text)
proc_sentences = preprocessing(proc_sentences)

tf_idf=tf_idf_scores(proc_sentences)
sent_weights = sent_weighting(tf_idf)

df = pd.DataFrame({
        'sent_weight': sent_weights,
        'sent_index' : [i for i in range(len(sent_weights))]
        })

Elbow_Method(df)
#Silhouette_Method(df)
"""
from sklearn.metrics import silhouette_score
range_n_clusters = list (range(2,10))
print ("Number of clusters from 2 to 9: \n", range_n_clusters)
score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters).fit(df)
    preds = clusterer.predict(df)
    centers = clusterer.cluster_centers_

    score.append (silhouette_score (df, preds, metric='euclidean'))
    #print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))


plt.plot(range_n_clusters, score)
plt.show()

"""



from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X = df

range_n_clusters = [2, 3, 4, 5, 6]
s = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    s.append(silhouette_avg)
    

plt.plot(range_n_clusters,s)
plt.show()









#k-means
clusters = KMean_Clustering(df)
while(True):
    summary = extract_summary(sentences, clusters)

    print("\nSummary:\n")
    print(summary)

    print("\nEvaluation: ")

    ref_summary = open("Dataset/Summaries/business/"+filename,"r")
    reference = ref_summary.read()
    evaluation(summary,reference)

    #find summary size
    summ_sentences = nltk.sent_tokenize(summary)
    summary_size = ((len(summ_sentences))/(len(sentences)))*100
    print("Number of sentences")
    print("In Document =",len(sentences))
    print("In Summary  =",len(summ_sentences))
    print("\nSummary size : ",summary_size,"%") 
    
    #if high frequency cluster is more than one

    print()
    answer = input("Want to choose another cluster?\nAnswer: ")
    if answer == "yes":
        continue
    else:
        break