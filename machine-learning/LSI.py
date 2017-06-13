import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.utils.extmath import randomized_svd
import pandas as pd


stop_words = set(stopwords.words('english')).difference(set(['all']))

def pre_process(data):
    
    data_p = []
    
    for text in data:
        
        cur_rev = []
        cur_text = [x.lower() for x  in re.findall(r"[\w]+|[^\s\w]",text)]   
        M = len(cur_text)
        st, sp = 0,0
        found_start = False
        
        for k in range(M):
            
            if cur_text[k].isalpha() and not found_start:
                
                st = k
                found_start = True
            
            elif (found_start and not cur_text[k].isalpha()) or k == M-1:
                
                if not cur_text[k].isalpha():
                    sp = k
                    found_start = False
                    cur_rev.append(cur_text[st:sp])
                    
                else:
                    
                    cur_rev.append(cur_text[st:])
                    
        data_p.append(cur_rev)
            
            
    return data_p
    
    
def get_n_grams(n, data):
    
    res = {}
    
    for k in range(len(data)):
        cur_rev = data[k] 
        cur_seen = []
        for sent in cur_rev:
            N = len(sent)
            if N < n:
                pass
            else:
                for j in range(N-n+1):
                    if sent[j] not in stop_words and sent[j+n-1] not in stop_words and sent[j].isalpha() and sent[j+n-1].isalpha():
                        cur = tuple(sent[j:j+n])
                        if cur in res:
                            res[cur][0] += 1
                            if cur not in cur_seen:
                                res[cur][1] += 1
                        else:
                            res[cur] = [1,1]
                        cur_seen.append(cur)
    return res
    

def summary_extract(data, n_min=2, n_max=5):
    
    dict_list = []
    
    for k in range(n_min,n_max+1):
        
        dict_list.append(get_n_grams(k,data))
        
    
    sum_list = []
    tot_grams = len(data)
    
    for dic in dict_list:
        
        for n_gram in dic:
            
            text = ' '.join([x.upper() for x in n_gram])
            sum_list.append([text, dic[n_gram][0], dic[n_gram][1], 
                             str(round(100*dic[n_gram][1]/tot_grams,2))+'%',
                             dict_list.index(dic)+n_min,
                            0.434 * dic[n_gram][0] * np.log(tot_grams/dic[n_gram][1])])
            
    res = pd.DataFrame(sum_list)
    res.columns = ['Phrase', 'Frequency', 'No. Cases', '% of Cases', 'Length', 'TF-IDF']
    
    
    return res.sort_values('Frequency', ascending=False)
    
def n_gram_comp(ng):
    
    return '_'.join(ng)
  

def get_topics_and_reviews(data,text_column_name, n_g_min, n_g_max, num_topics=20, review_tol = 0.08):
    
    
    data_text = data[text_column_name].as_matrix()
    vectorizer = TfidfVectorizer(min_df=5,
                                max_features=2000,
                                stop_words='english', 
                                ngram_range= (n_g_min,n_g_max))
    tf_idf = vectorizer.fit_transform(data_text)
    tf_idf_feature_names = vectorizer.get_feature_names()
    U_t, Sigma_t, VT_t = randomized_svd(tf_idf.T, 
                              n_components=num_topics,
                              n_iter=5,
                              random_state=7)
    
    topics = []
    
    for k in range(U_t.shape[-1]):
        topics.append(tf_idf_feature_names[U_t[:,k].argmax()])
        
    review_info = {}
    
    for k in range(VT_t.T.shape[-1]):
        
        cur =1 
        j = -1
        args = VT_t.T[:,k].argsort(axis=0)
        tol_ind = []
        while cur > review_tol*VT_t.T[:,k].max() and j > -12940:
            cur = VT_t.T[:,k][args[j]]
            tol_ind.append(args[j])
            j -= 1
            
        review_info[topics[k]] = {'text':data_text[tol_ind], 'df':data.ix[tol_ind]}
        
        
        
    return topics, review_info
    
filename = askopenfilename()



dryer_data = pd.read_csv(filename, encoding="ISO-8859-1")
text_data = dryer_data['Review'].as_matrix()
text_data_p = pre_process(text_data)

freq_df = summary_extract(text_data_p)

topics, rev_inf = get_topics(dryer_data,'Review', 2, 2)

topics_df = pd.DataFrame(topics)
topics_df.columns =['Topics']

