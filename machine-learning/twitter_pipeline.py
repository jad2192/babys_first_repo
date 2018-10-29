# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:27:56 2018

@author: DIOTJA

TO DO:
      
    - Continue tuning models
     
    - On going bug monitoring.
"""

import pandas as pd
import numpy as np
import twitter
import pyodbc
import time
from datetime import datetime, timedelta
import tqdm
import twitter_dnn
import phase2_model 
from timeit import default_timer
import logging
import logging.handlers
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pickle
from time_series_analysis import HMM, EARS, get_counts, discretize2

###############################################################################
#####       Set up twitter API, SQL Database and SMTP Connections         #####
###############################################################################

consumer_key = 'JdI7kqED9hckbSGCEtzXX42yM'
consumer_secret = 'C08YWRgXcyg55Jo4kVDprwUhxw8GZY5Ofn0snu0e31Ap6kwDqa'
api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret,
                 application_only_auth=True,
                 tweet_mode='extended')

conn = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER=sql12ynk08vp;'
    r'DATABASE=boca;'
    r'Trusted_Connection=yes'
    )
cursor = conn.cursor()

#email_add = str(input('Enter Consumer.org Email address: '))
#pw = getpass.getpass()
#creds = (email_add, pw)
creds = 'EWS_Pipeline_Script@consumer.org'
smtp_handler = logging.handlers.SMTPHandler(mailhost=("smtp.consumer.org"),
                                        fromaddr=creds, 
                                        toaddrs='james.diotte@consumer.org',
                                        subject= u"EWS Pipeline Error",
                                        secure=())
logger = logging.getLogger()
logger.addHandler(smtp_handler)

###############################################################################
#####                Import the Brands and Convert to Query               #####
###############################################################################
def get_brands(fp):
    
    brand_df = pd.read_csv(fp)
    brand_df.fillna('BLANK', inplace=True)
    brand_m = brand_df.as_matrix()
    brands = {}
    
    for k in range(brand_m.shape[0]):
        
        cur_brand = str(brand_m[k,0])
        cur_query_s = set(brand_m[k,2:]).difference(set(['BLANK']))
        
        if str(brand_m[k,1]) not in ['JAD', 'CR-GOVT', 'CR-PRODUCT']:
             
             q_raw = ' '.join(cur_query_s)
             q_fin = ' OR '.join([x for x in q_raw.split()
                                               if 'NONE' not in x])
        else:
            
            q_fin = ' OR '.join(cur_query_s)
            
        if q_fin:
                
            brands[cur_brand] = q_fin
    
    return brands
###############################################################################
#####                             ETL Functions                           #####
###############################################################################

def get_tweets(query, since_date, until_date):
    '''This function takes a search term and queries the twitter api
       to find all tweets related to that search from the previous day.
       It has a built in error exception to handle rate limiting.'''
    
    unique_tweets = {}
    tweet_ids = []
    
    try:
        
        statuses = api.GetSearch(query, count=100, lang='en',
                             locale='US', result_type='recent',
                             since=since_date,
                             until=until_date)
        
    except twitter.TwitterError:
        
        print('Rate limited, waiting 15 minutes...')
        print('Paused at: ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        time.sleep(60 * 15)
        print('Starting up...')
        statuses = api.GetSearch(query, count=100, lang='en',
                             locale='US', result_type='recent',
                             since=since_date,
                             until=until_date)
        
    for status in statuses:
        
        if status.id not in unique_tweets:
            
            unique_tweets[status.id] = status
            tweet_ids.append(status.id)
            tweet_ids.sort()
            
    stop = 1
    if tweet_ids:
        while stop != 0:
        
            try:
            
                old_len = len(tweet_ids)
                statuses = api.GetSearch(query,
                                count=100, lang='en', locale='US',
                                result_type='recent',
                                max_id=tweet_ids[0],
                                since=since_date,
                                until=until_date)
                for status in statuses:
                
                    if status.id not in unique_tweets:
                    
                        unique_tweets[status.id] = status
                        tweet_ids.append(status.id)
                        tweet_ids.sort()
            
                stop = len(tweet_ids) - old_len

            except twitter.TwitterError:
            
                print('Rate limited, waiting 15 minutes...')
                print('Paused at: ')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                time.sleep(60 * 15)
                print('Starting up...')
    return unique_tweets

def get_all_tweets(brands, since_date, until_date):
    
    relevant_tweets = []
    brand_set = list(brands)
    brand_set.sort()
    
    for brand in brand_set:
        print('Getting tweet for brand: ', brand)
        print('Query: ', brands[brand])
        tweets = get_tweets(brands[brand], since_date, until_date)
        temp_rel = []
        
        for x in tweets:
            
            if twitter_dnn.predict(tweets[x].full_text) > 0.71 :
                
                relv = twitter_dnn.predict(tweets[x].full_text)
                garb = phase2_model.predict(tweets[x].full_text)[0]
                nuis = phase2_model.predict(tweets[x].full_text)[1]
                safe = phase2_model.predict(tweets[x].full_text)[2]
                temp_rel.append((str(x), tweets[x].full_text,
                    (datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d'),
                        str(brand), relv, garb, nuis, safe))
                
        relevant_tweets += temp_rel
    
    return relevant_tweets

def save_tweets(tweets, db_name):   #Need to create a db
    
    sql_cmd = '''INSERT INTO %s (
             ID,
             Text,
             Tweet_Date,
             Brand,
             Relevance,
             Garbage,
             Nuisance,
             Safety
             )
             VALUES
             (?, ?, ?, ?, ?, ?, ?, ?);''' % db_name
             
    l = len(tweets)
    
    if l <= 1000:
        
        cursor.executemany(sql_cmd, tweets)
        cursor.commit()
        
    else:
        
        for k in tqdm.tqdm(range(l // 1000)):
            
            cursor.executemany(sql_cmd, tweets[1000*k:1000*(k+1)])
            cursor.commit()
        
        cursor.executemany(sql_cmd, tweets[-(l % 1000):])
        cursor.commit()

def safety_flag_analysis(date_out):
    
    sql_cmd = '''SELECT COUNT(Tweet_Date), Brand, Tweet_Date
             FROM EWS_Tweet_Stream_Ph2
             WHERE Safety > 0.70
             GROUP BY Brand, Tweet_Date;''' 
             
    cursor.execute(sql_cmd)
    rows = [x for x in cursor.fetchall()]

    sql_cmd = '''SELECT DISTINCT Tweet_Date FROM EWS_Tweet_Stream_Ph2
             ;''' 

    cursor.execute(sql_cmd)
    dates_t = [x[0] for x in cursor.fetchall()]

    sql_cmd = '''SELECT DISTINCT Brand FROM EWS_Tweet_Stream_Ph2
             ;''' 

    cursor.execute(sql_cmd)
    brands_t = [x[0] for x in cursor.fetchall()]

    counts = get_counts(rows, dates_t, brands_t)
    
    brand_models = {}

    for brand in brands_t:
        
        brand_models[brand] = (EARS(brand, counts[brand], 3),
                               HMM(brand, discretize2(counts[brand])))
        
    flagged_brands_e = set()
    flagged_brands_h = set()
    
    for brand in brands_t:
    
        x = brand_models[brand][0].predict_anomaly(7)
        y = brand_models[brand][1].anomaly_prob(len(counts[brand])) > 0.5

        if x:
            flagged_brands_e.add(brand)
        if y:
            flagged_brands_h.add(brand)
    
    flags = []

    for b in flagged_brands_e.difference(flagged_brands_h):
        flags.append((date_out, b, 'EARS'))
    for b in flagged_brands_h.difference(flagged_brands_e):
        flags.append((date_out, b, 'HMM'))
    for b in flagged_brands_e.intersection(flagged_brands_h):
        flags.append((date_out, b, 'BOTH'))
            
    sql_cmd = '''
             INSERT INTO EWS_Tweet_Anomaly (
             Flag_Date,
             Brand,
             TS_Model
             )
             VALUES
             (?, ?, ?);'''
             
    cursor.executemany(sql_cmd, flags)
    cursor.commit()
    return flagged_brands_e, flagged_brands_h
    
def job(brands, db_name):
    
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    since_date = str(np.copy(yesterday.strftime('%Y-%m-%d')))
    until_date = str(np.copy(today.strftime('%Y-%m-%d')))
    
    print('Starting Job for: ', datetime.now().strftime('%Y-%m-%d'))
    print('Getting Tweets from: ',
          (datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d'))
    print('------------------------------------------------------------------')
    start = default_timer()
    
    try:
        tweets = get_all_tweets(brands, since_date, until_date)
        
    except Exception as e:
         logger.exception('Unhandled Exception')
         
    time_elp = default_timer() - start
    hours_elp = time_elp // 3600
    min_elp = (time_elp % 3600) // 60
    sec_elp = (time_elp % 3600) % 60
    print('Saving Tweets...')
    print('------------------------------------------------------------------')
    
    saved = True
    try:
        save_tweets(tweets, db_name)
        
    except pyodbc.Error:
         
        saved = False
        now_d = datetime.now().strftime('%Y_%m_%d')
        save_path = 'C:\\Users\\Diotja\\Desktop\\tweets' + now_d + '.pickle'
        with open(save_path, 'wb') as handle:
            pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)

         
    date_out = (datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d')    
    print('Job Complete for: ', date_out)
    print('Total time elapsed: %i Hours %i min and %i seconds'%(hours_elp,
                                                                min_elp,
                                                                sec_elp))
    print(len(tweets), ' relevant tweets found and added.')
    print('Sleeping until tomorrow')
    print('------------------------------------------------------------------')
    
###############################################################################    
    if saved:
        
        flags_e, flags_h = safety_flag_analysis(date_out)
        
###############################################################################    
    me = creds
    to = "james.diotte@consumer.org"
    cc = "dsullivan@consumer.org"
    string =  'This is an auto-generated email\n'
    if not saved:
        string += '!THERE WAS A SAVE ERROR, PLEASE RUN BACKUP SCRIPT! \n'
        # twitter_pipeline_backup.py
    string += 'Report for %s:\n\nThere were %s total tweets recorded.\n' %(date_out, '{:,}'.format(len(tweets)))
    if saved:
        string += '\nBrands with flagged anomalous behaviour (Only EARS):\n\t'
        string += '\n\t'.join(flags_e.difference(flags_h))
        string += '\nBrands with flagged anomalous behaviour (Only HMM):\n\t'
        string += '\n\t'.join(flags_h.difference(flags_e))
        string += '\nBrands with flagged anomalous behaviour (Mutual):\n\t'
        string += '\n\t'.join(flags_e.intersection(flags_h))
    message_text = string
    rcpt = cc.split(",") + [to]
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "EWS Twitter Pipeline Daily Report"
    msg['To'] = to
    msg['Cc'] = cc
    msg.attach(MIMEText(message_text))
    server = smtplib.SMTP('smtp.consumer.org') # or your smtp server
    server.sendmail(me, rcpt, msg.as_string())
    server.quit()

###############################################################################
#####                   Start the Pipeline                                #####
###############################################################################

if __name__ == '__main__':
    
    db_name = 'EWS_Tweet_Stream_Ph2'
    fp = 'C:\\Users\\DIOTJA\\Desktop\\Data\\brands.csv'
    brands = get_brands(fp)
    job(brands, db_name)


