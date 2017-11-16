import nltk
from nltk.sentiment import vader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
import pandas as pd
import re
from spyre import server
from matplotlib import pyplot as plt
import seaborn as sns
from tkinter.filedialog import askopenfilename
from tkinter import *
from tqdm import tqdm

plt.style.use('fivethirtyeight')

    
###############################################################################  
############### Functions Used In Execution of the Program.####################
###############################################################################

def get_topics_and_reviews(data,text_column_name, n_g_min, n_g_max,
                           num_topics=25, review_tol = 0.08):
    '''This function performs the actual topic modelling using singular value
       decomposition of the tf-idf document matrix (Latent Semantic Analysis).
       This document matrix has a row for each tokenized word/n-gram and its
       index is preserved by factorization, tf_idf_feature names gives a map
       of this index to the corresponding terms.
       First it tokenizes and creates the tf-idf document matrix using 
       scikit-learn's TfidfVectorizer()  function, it then factorizes the
       transpose of this matrix into U S V^t using scikit-learn's fast 
       randomized SVD. The columns of matrix U are the topics and extracting 
       the argmax of each column and fnding corresponding term is how how to
       extract the topics. To get the reviews associated with the topics we 
       use V^t which has one row for each topic, we take the column indices 
       that are review_tol percent as large as the maximum for the row, those 
       indices are the documents associated with that topic.
       
       Inputs:
               data: The dataframe containg the reviews and brand info.
               
               text_column_name: The column name containing review text.
               
               n_g_min: Integer of the minimum n_gram size.
               
               n_g_max: Integer of the maximum n_gram size.
               
               num_topics: The number of topics to extract from the reviews.
               
               review_tol: The  minimal percentage of the maximum association
                           a review can have to be considered as dealing with
                           a topic. Float between 0-1.
                           
       Outputs:
               topics: A list of the topics found by the model.
               
               rev_inf: A dictionary with keys equal to the topics and value
                        a dictionary with keys 'text' and 'df' where 
                        rev_inf[topic]['text'] is a datframe with a single 
                        column containing all the reviews associated with
                        the given topic, while rev_inf[topic]['df'] gives the
                        full slice of the original data including review,
                        brand, etc. associated with the given topic.        '''
                          
    
    # Get the review text data as an array.
    data_text = data[text_column_name].as_matrix()
    
    # Create vectorizer object using scikit-learn's TfidfVectorizer() function.
    vectorizer = TfidfVectorizer(min_df=5,
                                max_features=2000,
                                stop_words='english', 
                                ngram_range= (n_g_min,n_g_max))
    
    # Use the vectorizer to create our tf-idf document matrix D.
    tf_idf = vectorizer.fit_transform(data_text)
    
    # Create the mapping of (row) index to word/ n-gram.
    tf_idf_feature_names = vectorizer.get_feature_names()
    
    # Factor the transpose of our tf-idf matrix using SVD: D = USV^t .
    # If there are N total words/n-grams and M total documents then our tf-idf 
    # matrix is of size NxM, and the factors U, S, V^t are of shapes: NxT, TxT,
    # and TxM respectively (where T is number of topics).
    U, Sigma, VT = randomized_svd(tf_idf.T, 
                              n_components=num_topics,
                              n_iter=5,
                              random_state=7)
    
    # Extract the most associated word/n-gram for each topic by finding the
    # argmax of each column of U and using the tf_idf_feature_name function.
    topics = []
    
    for k in tqdm(range(U.shape[-1])):
        top = tf_idf_feature_names[U[:,k].argmax()]
        if top not in topics:
            topics.append(top)
        else:
            topics.append(top+str(topics.count(top)))
	       
    # Create the review info dictionary by looping over the columns of V = 
    # (V^t)^t and extracting in descending order the rows with the strongest 
    # association (value), stopping once the association is less than or equal
    # to review_tol * max_association. We then extract the corresponding rows 
    # and reviews from the original dataframe.        
    review_info = {}
    
    # VT.T has shape (M,T) (since VT = V^t is of size TxM so transpose
    # has size MxT) thus we are looping over T (num topics) in  this for loop.
    for k in range(VT.T.shape[-1]):
        
        cur =1 
        j = -1
        args = VT.T[:,k].argsort(axis=0)
        tol_ind = []
        top_rel = []
        while cur > review_tol*VT.T[:,k].max() and j > -VT.T.shape[0]:
            cur = VT.T[:,k][args[j]]
            top_rel.append(cur / VT.T[:,k].max())
            tol_ind.append(args[j])
            j -= 1
            
        review_info[topics[k]] = {'text':data_text[tol_ind], 
                                  'df':data.ix[tol_ind]}
        review_info[topics[k]]['df']['Topic Relevance'] = top_rel
        
        
    return topics, review_info
    
def get_summ_by_brand(review_inf, data, topic):
    '''This function breaks down the topic statistics by brand returning
        the percentage of all reviews in the entered dataframe a brands makes
        up, as well as the percetage of the entered topic reviews a brand makes
        up.
        
        Inputs:
                review_info: The output dictionary of get_topics_and_reviews()
                             run on the dataframe entered next.
                data:  A dataframe containing all the review and brand info.
                       Must be the same datframe from which review_info above
                       was generated. 
                       
                topic: A topic contained in the topics output generated by
                       get_topics_and_reviews() run on 'data' dataframe.
                       
        Outputs:
                 brands: The list of unique brands in the review data.
                 
                 percts: A list whose i-th entry is the percent of all reviews
                         for the i-th brand (in brands list) that are 
                         associated with the input topic.
                         
                 pop_v_top: A list whose i-th entry is the percent of all 
                            reviews associated with input topic the i-th brand
                            makes up.                                       '''
    
    # Create list of the unique() brands present in the data,.
    df = review_inf[topic]['df']
    brands = data['Brand'].unique()
   
    # Finds the percentage of reviews of each brand which are associated with
    # the input topic.
    percts = np.zeros(brands.shape[0])
    for k in range(brands.shape[0]):
        tot = data[data['Brand']==brands[k]]['Brand'].shape[0]
        percts[k] = 100* df[df['Brand']==brands[k]]['Brand'].shape[0] / tot

    # Finds the percentage of reviews each brand makes up of the total reviews
    # associated with the input topic.    
    pop_v_top = np.zeros(brands.shape[0])
    
    for k in range(brands.shape[0]):
        
        pop_v_top[k] = (df[df['Brand']==brands[k]].shape[0] / 
                                                       df.shape[0])
        
    return brands, percts, pop_v_top
    
def get_per_1000(topic, data, rev_inf):
    ''' This function creates a bar plot of a brands rate of occurence of 
        reviews per 100 for a given topic, i.e. it plots the percts output
        from the get_summ_by_brand() function.
        
        Inputs:
                topic: The topic to use. Must appear in the topics output of 
                       get_topics_and_reviews() run on the 'data' entry.
                
                data: Dataframe containing all the reviews and brand info.
                
                rev_inf: The output dictionary of get_topics_and_reviews() run
                         on 'data' entry.
                         
        Outputs: A matplotlib bar chart.                                    '''
    
    brands, brand_sum, pop_rev = get_summ_by_brand(rev_inf, data,topic)
    fig = plt.figure(figsize=(12,12))
    plt.bar(np.arange(len(brand_sum)), brand_sum,color=(0/255,174/255,77/255),
            alpha=0.8, align='center', width=1)
    plt.xticks(np.arange(len(brand_sum)), brands, rotation='vertical')
    plt.ylabel('Issues / 100 Reviews')
    plt.title( topic.upper() + ' Issues per 100 Brand Reviews'.upper())
    
    return fig
    
def plot_pop_vs_rev(topic, data, rev_inf, sent):
    ''' Function to plot the fraction of topic reviews a brand constitutes
        versus the fraction of overall reviews that a brand constitutes.
        These fractions are plotted as an overlay bar chart.
        
        Inputs:
                topic: The topic to use.
                
                data: Dataframe containing the brand and review data.
                
                rev_inf: The output dictionary of get_topics_and_reviews()
                         run on the 'data' dataframe.
                         
                sent: The sentiment of the reviews in input data. Must be one
                      of the following strings: 'pos', 'neg', 'neu'.        
                      
        Output: A matplotlib overlay bar chart as desribed above. The bars are
                color coded in a way that blue showing on top is good, red  on
                top is bad, green/yellow on top are neutral.'''

    brands, brand_sum, pop_rev = get_summ_by_brand(rev_inf, data,topic)
    pop_hist = np.zeros(len(brands))
    
    # Getting the fraction of total reviews in 'data' that a brand makes up.
    for k in range(len(brands)):
    
        pop_hist[k] =  (data[data['Brand']==brands[k]].shape[0] / 
                        data.shape[0])
        
    #Setting the RGB color values for the bar plots.
    if sent == 'neu' or sent == 'none':
        
        c_rev = (0/255,173/255, 77/255)    # Green chosen from CR style guide.
        c_pop = (255/255, 221/255, 0/255)  # Yellow chosen from CR style guide.
        
    
    elif sent == 'neg':
        
        c_rev = (236/255, 28/255, 36/255)  # Red from CR brand style guide.
        c_pop = (43/255,228/255, 255/255)  # A blue randomly chosen.
       
    else:
        
        c_pop = (236/255, 28/255, 36/255)
        c_rev = (43/255,228/255, 255/255)
        
        


    ind = np.arange(len(brands))

    fig = plt.figure(figsize=(12,12))
    
    # Change order so that blue colored bar always comes second which makes the
    # mixed color in overlay bars look better than if red comes second.
    if sent == 'pos':
        
        p2 = plt.bar(ind, pop_hist, width=1,color=c_pop, alpha=0.8,
                     align='center')
        p1 = plt.bar(ind, pop_rev, width=1, color=c_rev, alpha=0.6, 
                 align='center')
    else:
        
        p1 = plt.bar(ind, pop_rev, width=1, color=c_rev, alpha=0.8, 
                 align='center')
        p2 = plt.bar(ind, pop_hist, width=1,color=c_pop, alpha=0.6,
                 align='center')

    plt.ylabel('Percent of Population')
    plt.title('Topic vs Total Review Frequencies' +
              '   TOPIC: ' + topic.upper() )
    plt.xticks(ind, brands, rotation='vertical')
    plt.legend((p1[0], p2[0]), ('Topic Reviews', 'Total Reviews'))
    
    return fig

    


def topic_brand_hm(review_inf, brands, topics, data):
    ''' This function creates a heatmap of the topic review rates of all the 
        topics vs all the brands.
        
        Inputs:
                review_inf: The dictionary with the text and dataframe for 
                            each topic, the second output of 
                            get_topics_and_reviews().
                            
                brands: A list of all the unique brands that appear in the 
                        data.
                        
                topics: A list of all the topics output by 
                        get_topics_and_reviews().
                        
                data: The dataframe from which all the topics and reviews
                      were generated.
                      
        Outputs: A seaborn heatmap described above.'''
    
    df = pd.DataFrame()
    for brand in brands:
        topic_pcts = []
        for topic in topics:
            df_t = review_inf[topic]['df']
            tot = data[data['Brand']==brand]['Brand'].shape[0]
            
            if tot > 0:
                topic_pcts.append(100* df_t[df_t['Brand']==brand]['Brand'].shape[0]
                / tot)
            else:
                topic_pcts.append(0)
                              
        df[brand] = topic_pcts
    df.index = topics
    cmap = sns.light_palette((147, 100, 39), input="husl",as_cmap=True)
    return sns.heatmap(df,cmap=cmap)
    
# Expressions for the split into sentences regex based function below.   
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
conjs = ['but', 'however', 'except', 'although', 'otherwise',
             'with the exception', 'the only problem', 'other than',
             'my only complaint', 'My only complaint']

def split_into_sentences(text):
    '''This function takes a string and splits it into it's component sentences
        works by splitting on punctuation with carefully chosen exceptions.
        Not written by James D., found here:
    https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    '''
    text = " " + str(text) + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "e.g." in text: text = text.replace("e.g.", "for example")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]",
                  "\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    for wrd in conjs:
        text = text.replace(wrd, '<stop>'+wrd)
    text = text.replace(".",".<stop>")
    text = text.replace(",",",<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    #sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
    
# Importing the pre-built sentiment analysis model VADER from nltk.
# http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
v_sent = vader.SentimentIntensityAnalyzer()

def sentiment_score(text):
    ''' This function does a bit of preprocessing to encorporate some
        bi/tri-grams into the vader model'''
        
    bigrams = {'no problems': 'no_problems',
                   'no complaints': 'no_complaints',
                   'no negatives': 'no_negatives',
                   'no concerns': 'no_concerns',
                   'no issues': 'no_issues',
                   'not very': 'not_very',
                   'no major': 'no_major',
                   'no negative': 'no_negative',
                   'no real': 'no_real',
                   'no troubles': 'no_troubles',
                   'no trouble': 'no_trouble',
                   'no worries': 'no_worries',
                   'no mechanical': 'no_mechanical',
                   'no problem': 'no_problem',
                   'could be': 'could_be',
                   'could use': 'could_use',
                   'does not': 'does_not',
                   'difficult to': 'difficult_to',
                   'not as': 'not_as',
                   'hard to': 'hard_to'}

    trigrams = {'no problems whatsoever': 'no_problems_whatsoever',
                    'no issues whatsoever': 'no_issues_whatsoever',
                    'no complaints whatsoever': 'no_complaints_whatsoever', 
                    'no trouble whatsoever': 'no_trouble_whatsoever',
                    'no troubles whatsoever':'no_troubles_whatsoever',
                    'no worries whatsoever': 'no_worries_whatsoever',
                    'no negatives whatsoever': 'no_negatives_whatsoever',
                    'no concerns whatsoever': 'no_concerns_whatsoever'}
        
    cur_text = text.lower()
        
    for tri in trigrams:
            
        if tri in cur_text:
                
            cur_text = cur_text.replace(tri, trigrams[tri])
        
    for bi in bigrams:
            
        if bi in cur_text:
                
            cur_text = cur_text.replace(bi, bigrams[bi])
                
    return v_sent.polarity_scores(cur_text)
        
        
def pos_neg_review_split(text_data, tol):
    ''' This function takes an array of text, splits each component into its
        sentences and then partitions them based on sentiment using the VADER
        model. It also records the sentiment strength of each partitioned 
        block.
        
        Input:
                text_data: An array or list of text strings to be partitioned 
                            by sentiment.
                            
        Outputs: Three arrays (pso, neg, neu) of the same length as input 
                 array whose entries are the sentences of a certain sentiment 
                 from original entry. Three more arrays (pos_s, neg_s, neu_s)
                 which contain the sentiment score asscociated to the text.
                 '''
    
    print('Running Sentiment Analysis...')
    pos, neg, neu = [], [], []
    pos_s, neg_s, neu_s, = [], [], []
    L = len(text_data)
    for k in tqdm(range(L)):
        
        cur_pos, cur_neg, cur_neu = [], [], []
        cur_pos_s, cur_neg_s, cur_neu_s = [], [], []
        
        # Split the k-th text entry into individual sentences.
        cur_text = [x for x in split_into_sentences(text_data[k])
                    if len(x) > 1]

        # Analyze the sentiment of each sentence and place it into the 
        # corresponding sentiment partition.
        for sent in cur_text:
            
            sentiment = sentiment_score(sent)
            
            # Consider neutral iff if the neutral score is > 95%.
            # Else choose the larger of the positive or negative sentiment.
            if sentiment['neu'] > tol:
                
                cur_neu.append(sent)
                cur_neu_s.append(sentiment['neu'])
                
            elif sentiment['pos'] >= sentiment['neg']:
                
                cur_pos.append(sent)
                cur_pos_s.append(sentiment['pos'])
                
            else:
                
                cur_neg.append(sent)
                cur_neg_s.append(sentiment['neg'])
                
        if len(cur_pos) > 0: 
            
            pos.append(' '.join(cur_pos))
            pos_s.append(np.asarray(cur_pos_s).mean())
        
        # If an entry's partition for a certain sentiment is empty fill it with
        # the '.' just to avoid errors in processing. Punctuation is removed
        # for topic modelling so this will not affect anything.
        else:
            
            pos.append('.')
            pos_s.append(0)
            
        if len(cur_neg) > 0:
            
            neg.append(' '.join(cur_neg))
            neg_s.append(np.asarray(cur_neg_s).mean())
            
        else:
            
            neg.append('.')
            neg_s.append(0)
            
        if len(cur_neu) > 0:
            
            neu.append(' '.join(cur_neu))
            neu_s.append(np.asarray(cur_neu_s).mean())
            
        else:
            
            neu.append('.')
            neu_s.append(0)
        
    return pos, neg, neu, pos_s, neg_s, neu_s
    
def sentiment_split_df(data, rev_column_name, tol):
    ''' This function uses the sentiment based sentence partitioning on the
        reviews contained in our dataframe and then splits the original data
        into three new datframes with the review section replaced with 
        sentiment partitioned reviews.
        
        Inputs:
                data: Original dataframe to be partioned by sentiment.
                
                rev_column_name: The name of the column containing the 
                                 review data.
                                 
        Outputs: Three new dataframes which are partitions of the original
                 based on sentiment.
                 '''
    
    pos, neg, neu, pos_s, neg_s, neu_s = pos_neg_review_split(
                                         data[rev_column_name].as_matrix(),
                                         tol)
    pos_df, neg_df, neu_df = data.copy(), data.copy(), data.copy()
    pos_df[rev_column_name] = pos
    neg_df[rev_column_name] = neg
    neu_df[rev_column_name] = neu
    pos_df['Sentiment Strength'] = pos_s
    neg_df['Sentiment Strength'] = neg_s
    neu_df['Sentiment Strength'] = neu_s

    cols = list(data.columns)
    N = cols.index(rev_column_name)
    
    pos_df = pos_df[cols[:N+1] + ['Sentiment Strength'] + cols[N+1:-1]]
    neg_df = neg_df[cols[:N+1] + ['Sentiment Strength'] + cols[N+1:-1]]
    neg_df = neg_df[cols[:N+1] + ['Sentiment Strength'] + cols[N+1:-1]]

    return pos_df, neg_df, neu_df
    
def topic_summ(pos_top, neg_top, neu_top, pos_data, neg_data, neu_data,
             pos_rev_inf, neg_rev_inf, neu_rev_inf):
    ''' This function takes all the topics found for all three sentiments
        (positive, negative, neutral) as well as the corresponding sentiment
        partitoned data and returns a dataframe which gives the percent of 
        reviews accross all brands a certain topic makes up, both by sentiment
        and total accross all sentiments.
        
        Inputs:
                pos_top: The list of topics found from the positive sentiment
                         data.
                        
                neg_top: The list of topics found from negative sentiment
                         data.
                         
                neu_top: The list of topics found from the neutral sentiment
                         data.
                         
                pos_data: The postive sentiment partition dataframe.
                
                neg_data: The negative sentiment partition dataframe.
                
                neu_data: The neutral sentiment partition dataframe.
                
                pos_rev_inf: The dictionary from get_topics_and_reviews()
                             run on pos_data.
                              
                neg_rev_inf: The dictionary from get_topics_and_reviews()
                             run on neg_data.
                             
                neu_rev_inf: The dictionary from get_topics_and_reviews()
                             run on neu_data.
                             
        Outputs: A dataframe giving the sentiment split percentage breakdown 
                of reviews associated with a certain topics accross all brands.
                '''
    
    tot_top = list(set(pos_top).union(set(neg_top)).union(set(neu_top)))
    N = len(tot_top)
    pos_pct, neg_pct, neu_pct = np.zeros(N), np.zeros(N), np.zeros(N)
    
    for k in range(N):
        
        if tot_top[k] in pos_top:
            
            cur_df = pos_rev_inf[tot_top[k]]['df']
            pos_pct[k] = 100 * cur_df.shape[0] / pos_data.shape[0]
            
        if tot_top[k] in neg_top:
            
            cur_df = neg_rev_inf[tot_top[k]]['df']
            neg_pct[k] = 100 * cur_df.shape[0] / neg_data.shape[0]
        
        if tot_top[k] in neu_top:
            
            cur_df = neu_rev_inf[tot_top[k]]['df']
            neu_pct[k] = 100 * cur_df.shape[0] / neu_data.shape[0]

    W = (np.array([pos_data.shape[0], neg_data.shape[0],
                        neu_data.shape[0]]) / 
                (pos_data.shape[0] + neg_data.shape[0] + neu_data.shape[0]))
    tot_pct = W[0]*pos_pct + W[1]*neg_pct + W[2]*neu_pct
            
    res_df = pd.DataFrame()
    res_df['Topic'] = tot_top
    res_df['Positive Percentage'] = [str(round(x,2)) + '%' for x in pos_pct]
    res_df['Negative Percentage'] = [str(round(x,2)) + '%' for x in neg_pct]
    res_df['Neutral Percentage'] = [str(round(x,2)) + '%' for x in neu_pct]
    res_df['Total Percentage'] = [str(round(x,2)) + '%' for x in tot_pct]

    return res_df
    
def top_rel(rel):
    
    if rel >= 0.8:
        
        return 'Very Likely'
    
    elif rel < 0.8 and rel >= 0.2:
        
        return 'Likely'
        
    else:
        
        return 'Somewhat Likely'

###############################################################################  
############### Execution of the Program Happens Here.#########################
###############################################################################


if __name__ == '__main__':
    
    root = Tk()
    filename = askopenfilename()
    root.destroy()
    
    master = Tk()
    def show_entry_fields():
    
        print("Sentiment Tolerance:",  e1.get())
        master.quit()

    capt0 = "Sentiment tolerance:\n Enter a number from 0 to 0.99. \n"
    capt1 = "A good default is 0.75."
    capt = capt0 + capt1
    Label(master,
      text=capt).grid(row=0)
    e1 = Entry(master)
    e1.grid(row=0, column=1)
    Button(master, text='Run',
       command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=4)
    mainloop( )
    tol = float(e1.get())
    master.destroy()

    data = pd.read_csv(filename, encoding="ISO-8859-1")
    data.fillna('NA', inplace=True)
    pos_df_r, neg_df_r, neu_df_r = sentiment_split_df(data, 'Review', tol)
    pos_df = pos_df_r[pos_df_r['Review']!='.'].copy()
    neg_df = neg_df_r[neg_df_r['Review']!='.'].copy()
    neu_df = neu_df_r[neu_df_r['Review']!='.'].copy()
    
    neu_df.index = np.arange(neu_df.shape[0])
    neg_df.index = np.arange(neg_df.shape[0])
    pos_df.index = np.arange(pos_df.shape[0])

    #text_data = data['Review'].as_matrix()
    #text_data_p = pre_process(text_data)

    #freq_df = summary_extract(text_data_p)
    print('Running topics modeler 1/4')
    topics, rev_inf = get_topics_and_reviews(data,'Review', 2, 2)
    print('Running topics modeler 2/4')
    topics_pos, rev_inf_pos = get_topics_and_reviews(pos_df,'Review', 2, 2)
    print('Running topics modeler 3/4')
    topics_neg, rev_inf_neg = get_topics_and_reviews(neg_df,'Review', 2, 2)
    print('Running topics modeler 4/4')
    topics_neu, rev_inf_neu = get_topics_and_reviews(neu_df,'Review', 2, 2)
    brands = data['Brand'].unique()
    topics_df = pd.DataFrame(topics)
    topics_df.columns =['Topics']

    options = {'none': [], 'pos': [], 'neg': [], 'neu': []}
    data_frames = {'none': data, 'pos': pos_df, 'neg': neg_df, 'neu': neu_df}
    top_d = {'none': (topics, rev_inf, 'top'),
             'pos': (topics_pos, rev_inf_pos, 'top_p'),
             'neg': (topics_neg, rev_inf_neg, 'top_n'),
             'neu': (topics_neu, rev_inf_neu, 'top_ne')}           
             
    for sents in options:
        for tops in top_d[sents][0]:
            options[sents].append({"label": tops, "value": tops})
            
    import cherrypy
    from cherrypy import engine, tree
    import webbrowser
    
    def launch_app(app, host="local",port=8080,prefix='/'):
        app.prefix = prefix
        webapp = app.getRoot()
        if host!="local":
            cherrypy.server.socket_host = '0.0.0.1'
        cherrypy.server.socket_port = port
        tree.mount(webapp,prefix)
        if hasattr(engine, "signal_handler"):
            engine.signal_handler.subscribe()
        if hasattr(engine, "console_control_handler"):
            engine.console_control_handler.subscribe()
        engine.start()
        webbrowser.open("http://127.0.0.1:8080")    
        engine.block()

    class ViewTopics(server.App):
        title = "Topics"

        inputs = [{
                    'type': 'dropdown',
                    'label': 'Sentiment', 
                    'options':[
                                {'label':'None', 'value': 'none'},
                                {'label': 'Positive', 'value': 'pos'},
                                {'label': 'Negative', 'value': 'neg'},
                                {'label': 'Neutral', 'value': 'neu'},
                                ],
                     'key': 'sent', 
                     'action_id': "update_data"
                     
                },
                  {
                    'type': 'dropdown',
                    'label': 'Topics with no Sentiment Analysis', 
                    'options': options['none'],
                     'key': 'top', 
                     'action_id': "update_data"
                     
                },
                {
                    'type': 'dropdown',
                    'label': 'Topics from Positive Sentiment Reviews', 
                    'options': options['pos'],
                     'key': 'top_p', 
                     'action_id': "update_data"
                     
                },
                {
                    'type': 'dropdown',
                    'label': 'Topics from Negative Sentiment Reviews', 
                    'options': options['neg'],
                     'key': 'top_n', 
                     'action_id': "update_data"
                     
                },
                {
                    'type': 'dropdown',
                    'label': 'Topics from Neutral Sentiment Reviews', 
                    'options': options['neu'],
                     'key': 'top_ne', 
                     'action_id': "update_data"
                     
                },
                
                {
                    'type':'dropdown',
                    'label': 'Sort Tables By',
                    'options':[
                                {'label':'Topic Relevance', 'value': 'top'},
                                {'label':'Sentiment Strength','value': 'sent'}
                                ],
                    'key': 'sort',
                    'action_id': 'update_data'
                    },
                    {
                    'type':'dropdown',
                    'label': 'Text, SpreadSheet or Topic Summary',
                    'options':[
                                {'label':'Text', 'value': 'text'},
                                {'label': 'Spreadsheet', 'value': 'df'},
                                {'label': 'Topic Summary', 'value': 'summ'}
                                ],
                    'key': 'ToS',
                    'action_id': 'update_data'
                    }, 
                    {
                    'type':'dropdown',
                    'label': 'Summary Visualization Choice',
                    'options':[
                                {'label':'Topic vs Total', 'value': 'TvT'},
                                {'label':'Topic Rate per 100 ',
                                'value':'p1000'},
                                {'label': 'Brand vs Topic Heatmap', 'value':
                                    'hm'}
                                ],
                    'key': 'summ',
                    'action_id': 'update_data'
                    }
                    ]
                     
        controls = [{
                        'type': "hidden",
                        'id' : "update_data"
                    },
                    {
                        'type': "hidden",
                        'id' : "update_data"
                    }
                    ]
                             
        tabs = ["Table", 'Plot']

        outputs = [{
                    "type": "table",
			       "id": "table_id",
			       "control_id": "update_data",
			       "tab": "Table",
                   'on_page_load' : True
		     },{
                     'type': 'plot',
                     'id': 'plot',
                     'control_id': 'update_data',
                     'tab': 'Plot'
                     }
              
	               ]

        def getData(self, params):
            
            sent = params['sent']
            rev_infs = top_d[sent][1]
            top_key = top_d[sent][2]
            top = params[top_key]
            df = rev_infs[top]['df']
            top_rels = df['Topic Relevance'].as_matrix()
            top_rels_t = [top_rel(rel) for rel in top_rels]
            df['Topic Likelihood'] = top_rels_t
              
            if params['ToS'] == 'df':
                
                df_s = df.copy()
                pd.set_option('display.max_colwidth', 50)
                
                if params['sort'] != 'sent' or sent == 'none':
                    
                    return df_s
                else:
                    
                    return  df_s.sort_values(by='Sentiment Strength',
                                            ascending=False)
                
            elif params['ToS'] == 'text':
                
                if sent != 'none':
                    
                    df_s = df[['Review', 'Topic Relevance', 
                    'Topic Likelihood', 'Sentiment Strength']].copy()
                    
                else:
                    
                    df_s = df[['Review', 'Topic Relevance', 
                               'Topic Likelihood']].copy()
                
                pd.set_option('display.max_colwidth', 300)  
                
                if params['sort'] != 'sent' or sent == 'none':
                    
                    return df_s
                else:
                    
                    return  df_s.sort_values(by='Sentiment Strength',
                                            ascending=False)
            else:
                
                df_s = topic_summ(topics_pos, topics_neg, topics_neu,
                                  pos_df, neg_df, neu_df, rev_inf_pos,
                                  rev_inf_neg, rev_inf_neu)
                return df_s
                
        
        def getPlot(self, params):
            
            sent = params['sent']
            data_frame = data_frames[sent]
            topics_c = top_d[sent][0]
            rev_infs = top_d[sent][1]
            top_key = top_d[sent][2]
            top = params[top_key]
            if params['summ'] == 'p1000':
                return get_per_1000(top, data_frame, rev_infs)
                
            elif params['summ'] == 'TvT':
                return plot_pop_vs_rev(top, data_frame, rev_infs, sent)
                
            else:
                return topic_brand_hm(rev_infs,brands,topics_c,data_frame)
            


    app = ViewTopics()
    launch_app(app)
