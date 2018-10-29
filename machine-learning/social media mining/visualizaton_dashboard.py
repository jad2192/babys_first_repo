import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import pyodbc
import numpy as np
import matplotlib.colors as mplc
import random
import json
from textwrap import dedent as d
import nltk
from nltk.corpus import stopwords
import datetime

conn = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER=;'
    r'DATABASE=;'
    r'Trusted_Connection=yes'
    )
cursor = conn.cursor()


###############################################################################
#     The following SQL queries get the list of dates for twitter data        #
###############################################################################
sql_cmd = '''
        SELECT DISTINCT Tweet_Date FROM EWS_Tweet_Stream_Ph2
             ;''' 
cursor.execute(sql_cmd)
dates_t = [x[0] for x in cursor.fetchall()]
months_t = []

for date in dates_t:
    
    if date[:-3] not in months_t:
        
        months_t.append(date[:-3])
        
months_t.sort()

# This block creates a partition for creating weekly counts for tweet data
now = datetime.datetime.now().strftime('%Y-%m-%d')
weeks_t = ['2018-05-07']
w_start = datetime.datetime.strptime(weeks_t[-1], '%Y-%m-%d')
k = 1
new_w = weeks_t[-1]
while new_w < now:
    new_w = (w_start + datetime.timedelta(days=7*k)).strftime('%Y-%m-%d')
    if new_w < now:
        weeks_t.append(new_w)
    k +=1

if (datetime.datetime.now() -
    datetime.datetime.strptime(weeks_t[0], '%Y-%m-%d')).days % 7 == 0:
    weeks_t.append(now)
    
sql_cmd = '''
        SELECT DISTINCT Brand FROM EWS_Tweet_Stream_Ph2
             ;''' 

cursor.execute(sql_cmd)
brands_t = [x[0] for x in cursor.fetchall()]

sql_cmd = '''
        SELECT COUNT(DISTINCT ID), Brand, Tweet_Date FROM EWS_Tweet_Stream_Ph2
        WHERE Safety > 0.70
        GROUP BY Brand, Tweet_Date;''' 
cursor.execute(sql_cmd)
rows_t = [x for x in cursor.fetchall()]

###############################################################################
#           Getting date information from CPSC data                           #
###############################################################################
sql_cmd = '''
        SELECT DISTINCT [Report Date] FROM CPSC_Safer_Prod
             ;''' 
cursor.execute(sql_cmd)
dates_all = [x[0] for x in cursor.fetchall()]

months_sp = []

for date in dates_all:
    
    if date[:-3] not in months_sp:
        
        months_sp.append(date[:-3])
months_sp.sort()

sql_cmd = '''
        SELECT DISTINCT [Product Type] FROM CPSC_Safer_Prod
             ;''' 

cursor.execute(sql_cmd)
brands_sp = [x[0] for x in cursor.fetchall()]
###############################################################################
#              Defining functions for use in subsequent code                  #
###############################################################################

def date_str(x):
    ''' Input: An intger or string, x,  representing a piece of a date.
        Output: String representation that adds a leading zero if x
                is only a single digit.
                
        Examples: date_str(3) = '03'
                  date_str(2018) = '2018' '''    
                  
    if len(str(x)) == 1:
        return '0'+str(x)
    else:
        return str(x)

def color_gen():
    ''' Functon which randomly generates a high saturation, 
        high brightness color to associate with a brand or category
        for use in plotting.
        
        Ouput: Hex code of generated color as a string'''
        
    h = np.random.uniform()
    s = np.random.uniform(low=0.85)
    v = np.random.uniform(low=0.85)
    
    return mplc.to_hex(mplc.hsv_to_rgb((h,s,v)))
    
def get_colors(brands):
    '''Input: A list of brands or categories.
       Output: A dictionary with brands/categories as keys and
               a randomly assigned hex color string as value.'''
    res =  {}
    
    for brand in brands:
        
        res[brand] = color_gen()
        
    return res

def get_flag_dates(brand):
    '''Input: Brand name (as a string).
       Output: A list of flagged anomalous date ranges for daily tweets for 
               the given brand. List elements are tuples of the form:
               (start_date, end_date).
       Example output: 
           [('2018-08-04', '2018-08-07'), ('2018-09-01, '2018-09-01')] '''
    
    sql_cmd = '''
           SELECT Flag_Date FROM EWS_Tweet_Anomaly
           WHERE BRAND = '%s' ''' %brand.replace("'", "''")
    
    cursor.execute(sql_cmd)
    fl_d = [row[0] for row in cursor.fetchall()]
    dates_f = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in fl_d]
    dates_f.sort()
    ranges = []
    start, stop = 0, 0
    # This loop extracts flagged anomalous ranges by checking if the 
    # proceeding flagged date is the following day.
    for k in range(1,len(dates_f)):
        
        if (dates_f[k] - dates_f[k-1]).days == 1:
            stop += 1
            if k == len(dates_f)-1:
                ranges.append((dates_f[start].strftime('%Y-%m-%d'),
                               dates_f[stop].strftime('%Y-%m-%d')))
        else:
            ranges.append((dates_f[start].strftime('%Y-%m-%d'),
                           dates_f[stop].strftime('%Y-%m-%d')))
            start, stop = k, k
        
    ranges.append((dates_f[start].strftime('%Y-%m-%d'),
                   dates_f[stop].strftime('%Y-%m-%d')))
    return ranges

# The following are all pieces of the N-gram analyzer.
    
translator = str.maketrans('', '', '''!"$%&\()*+,-./:;<=>?@[\\]^'_`{|}~''')
remove_words = {'rt', '@', 'https'}
add_stops = {'womans', 'mans', 'man', 'woman',
                'via', 'due'}
stop_wds = set(stopwords.words('english')).union(add_stops)

def process_tweet(tweet):
    '''Input: A raw string to pre-process.
       Output: A processed string with certain encoding fixed
               and certain words removed (remove_words).'''
               
    tweet = tweet.replace('&amp;', 'and')
    tweet = tweet.replace('\x92', "'")
    tweet_split = [wd for wd in tweet.lower().split()]
    for k in range(len(tweet_split)):
        if  any([(x in tweet_split[k]) for x in remove_words]):
            tweet_split[k] = ''
        
    tweet = ' '.join(tweet_split)
    return ' '.join(tweet.translate(translator).split())

def get_n_grams(n, data):
    ''' N-gram Tokenizer which counts frequency of N-grams present in input 
        text data. Only N-grams which don't start or end in stop words or 
        special characters are considered. Counts total term frequency (tf)
        as well as document frequency (df).
        
        Inputs: 
               n: An integer representing the size of N-grams to count.
               data: A list of text data from which to count N-grams.
               
        Output: A dictionary with N-gram as key and the list [tf,df] as value
        '''
    
    res = {}
    
    for k in range(len(data)):
        sent = data[k].split()
        cur_seen = []
        N = len(sent)
        if N < n:
            pass
        else:
            for j in range(N-n+1):
                if all([sent[j] not in stop_wds, sent[j+n-1] not in stop_wds,
                        sent[j].isalpha(), sent[j+n-1].isalpha()]):
                
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
    '''Function that creates a table output of top N-grams found in the input
       text data.
       
       Inputs:
              data: A list of text documents on which to analyze.
              n_min: Integer representing the mininum N-gram size to count.
              n_max: Integer representing the maximum N-gram size to count.
              
       Output: A pandas DataFrame listing N-gram total counts (Frequency),
               document frequency (No. Cases), document percentage (% Cases),
               as well as the N-gram itself and its length. Sorted by 
               decreasing Frequency.'''
    
    dict_list = []
    
    for k in range(n_min,n_max+1):
        
        dict_list.append(get_n_grams(k,data))
        
    
    sum_list = []
    tot_grams = len(data)
    
    for dic in dict_list:
        
        for n_gram in dic:
            
            text = ' '.join([x.upper() for x in n_gram])
            
            if (dic[n_gram][1] > 1) or (tot_grams < 10):
                
                sum_list.append([text, dic[n_gram][0], dic[n_gram][1], 
                             str(round(100*dic[n_gram][1]/tot_grams,2))+'%',
                             dict_list.index(dic)+n_min])
    if sum_list:
        res = pd.DataFrame(sum_list)
    else:
        res = pd.DataFram()
    res.columns = ['Phrase', 'Frequency', 'No. Cases', '% of Cases', 'Length']
    return res.sort_values('Frequency', ascending=False)

###############################################################################
#                            Dash Code                                        #
###############################################################################

brand_colors_t = get_colors(brands_t)
brand_colors_sp = get_colors(brands_sp)



xaxis_d=dict(
        rangeselector=dict(
            font=dict(color='#bd1cc9'),
            bgcolor='#111111',
            buttons=list([
                dict(count=1,
                     label='One Week',
                     step='week',
                     stepmode='backward'
                     ),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date',
        font= {'text': '#7FDBFF'}
    )
xaxis_m=dict(
        tickformat='%Y-%m',
        rangeselector=dict(
            font=dict(color='#bd1cc9'),
            bgcolor='#111111',
            buttons=list([
                dict(count=1,
                     label='One Month',
                     step='week',
                     stepmode='backward'
                     ),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date',
        font= {'text': '#7FDBFF'}
    )
layout = dict(
    autosize=True,
    height=600,
    font=dict(color='#CCCCCC'),
    titlefont=dict(color='#CCCCCC', size='20'),
    margin=dict(
        l=35,
        r=35,
        b=25,
        t=50
    ),
    hovermode="closest",
    plot_bgcolor="#191A1A",  ##303333  #191A1A
    paper_bgcolor="#020202",
    legend=dict(font=dict(size=10), orientation='v'),
    title='Product Safety Timeseries: Twitter'
)


css_url = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash()
app.css.append_css({"external_url": css_url})
app.layout = html.Div(children=[
  html.Div([
          html.Div([
          html.Label('Data Source:', style={'fontSize':14,
                                            'fontWeight':'bold'}),
          dcc.Dropdown(id='main-selec',
                  options=[{'label': 'Twitter', 'value': 'twitter'},
                           {'label': 'CPSC', 'value': 'cpsc'}],
                  value='twitter'
                 )],
                    style={'width': '25%', 'display': 'inline-block',
                         'margin':5}   ),
          html.Div([
          html.Label('Count Visualization:', style={'fontSize':14,
                                            'fontWeight':'bold'}),
          dcc.Dropdown(id='count-type',
                  options=[{'label': 'Raw', 'value': 'raw'},
                           {'label': 'Standardized', 'value': 'std'}],
                  value='std',
                 )],
                  style={'width': '25%', 'display': 'inline-block',
                         'margin':5}),
         html.Div([
         html.Label('Count Interval Size:', style={'fontSize':14,
                                            'fontWeight':'bold'}),
         dcc.Dropdown(id='day-month', value='day'
                  )],
                  style={'width': '25%', 'display': 'inline-block',
                         'margin':5}),
         html.Div([
         html.Label('Show Warning Flags:', style={'fontSize':14,
                                            'fontWeight':'bold'}),
         dcc.Slider(id='show-flag',
                    min=0,
                    max=1,
                    step=1,
                    value=0,
                    marks ={0:'Off', 1:'On'}
                  )],
                  style={'width': '15%', 'display': 'inline-block',
                         'margin':15})],
        style={'width': '100%', 'display': 'table'}
           ),           
  html.Div([
     html.Div([
     html.Label('Select Brand(s) and/or Categories:', style={'fontSize':14,
                                            'fontWeight':'bold'}),
     dcc.Dropdown(id='brand-dropdown',
        value=['Product_Safety', 'Product_Safety2', 'Product_Safety3'],
        multi = 'True')], style={'width': '85%',
                                 'display': 'table-cell', 'margin':5})
    ],
     style={'width': '100%', 'display': 'table'}),
     html.Div([
        html.Div([
            dcc.Slider(
        id='tol-slider',
        min=0.5,
        max=0.95,
        step=0.01,
        value=0.7,
        marks = {x / 100: str(x / 100) for x in range(50,100,5)}
    )], style={'width': '68%', 'display': 'inline-block',
                'margin':5}),
        html.Div(id='tol-out'
        , style={'width': '25%', 'display': 'inline-block',
                 'margin':20, 'fontWeight':'bold'})
            ], style= {'width': '100%',
                       'marginBottom': 10}),
    html.Div([                                
    dcc.Graph(id='ts-graph')],
    style={'backgroundColor': '#111111'}),
    html.Div([
        html.Div([
                html.H1(id='table-title', style={
                        'fontSize':32})],
                style={'width': '68%', 'display': 'inline-block',
                'margin':5, 'fontColor':'#FFFFFF'}),
        html.Div([
                html.Label('Raw Text or Text Analysis :',
                           style={'fontSize':14,
                                            'fontWeight':'bold'}),
                dcc.RadioItems(id='table-choice', 
                        value='tweets',
                        labelStyle ={'display': 'inline-block'})],
                        style={'width': '25%', 'display': 'inline-block',
                               'margin':5, 'fontColor':'#FFFFFF'}),
        html.Table(id='tweet-table', style={'backgroundColor':'#020202',
                                            'fontColor':'#FFFFFF'})
        ], style={'backgroundColor': '#111111', 'color': '#FFFFFF'})
    ])

@app.callback(
        dash.dependencies.Output('table-choice', 'options'),
        [dash.dependencies.Input('main-selec', 'value')])
def update_table_choice(ds):
    if ds =='twitter':
        return [{'label': 'Tweets', 'value': 'tweets'},
                {'label': 'Text Analysis', 'value': 'ng'}]
    else:
        return [{'label': 'Incident Descriptions', 'value': 'tweets'},
                {'label': 'Text Analysis', 'value': 'ng'}]
@app.callback(
        dash.dependencies.Output('day-month', 'options'),
        [dash.dependencies.Input('main-selec', 'value')])
def update_day_month(ds):
    if ds == 'twitter':
        return [{'label': 'Daily', 'value': 'day'},
                {'label': 'Weekly', 'value': 'week'},
                {'label': 'Monthly', 'value': 'month'}]
    else:
        return [{'label': 'Monthly', 'value': 'month'}]

@app.callback(
        dash.dependencies.Output('brand-dropdown', 'options'),
        [dash.dependencies.Input('main-selec', 'value')])
def update_brand_dd(ds):
    if ds == 'twitter':
        return [{'label': brand, 'value': brand} for brand in brands_t]
    else:
        return [{'label': brand, 'value': brand} for brand in brands_sp]
@app.callback(
        dash.dependencies.Output('tol-out', 'children'),
        [dash.dependencies.Input('tol-slider', 'value')])
def output_tol(tol):
    return 'Selected Safety Tolerance: %s'%str(tol)

@app.callback(
        dash.dependencies.Output('table-title', 'children'),
        [dash.dependencies.Input('table-choice', 'value'),
         dash.dependencies.Input('main-selec', 'value')])
def update_table_title(tbl, ds):
    
    if tbl == 'tweets' and ds == 'twitter':
        
        return 'Tweets (Click Graph Point to Load):'
    
    elif tbl == 'tweets' and ds == 'cpsc':
        
        return 'Incident Descriptions (Click Graph Point to Load):'
    
    else:
        
        return 'N-Gram Analysis: '
        
    
@app.callback(
    dash.dependencies.Output('ts-graph', 'figure'),
    [dash.dependencies.Input('brand-dropdown', 'value'),
    dash.dependencies.Input('count-type', 'value'),
    dash.dependencies.Input('tol-slider', 'value'),
    dash.dependencies.Input('day-month', 'value'),
    dash.dependencies.Input('main-selec', 'value'),
    dash.dependencies.Input('show-flag', 'value')])
def update_figure(selected_brands, count_type, tol, day_mo, ds, flag):
    
    selected_brands_n = [b.replace("'", "''") for b in selected_brands]
            
    b_list_s = ["'" + brand + "'" for brand in selected_brands_n]
    b_list = '(' + ', '.join(b_list_s) + ')'
    
    if ds == 'twitter':
    
        if (day_mo == 'day' or day_mo == None) and flag == 1:
        
            sql_cmd = '''
            SELECT COUNT(DISTINCT ID), Brand,
            Tweet_Date FROM EWS_Tweet_Stream_Ph2
            WHERE Safety > %s
            AND Brand in %s
            GROUP BY Brand, Tweet_Date;''' %(str(tol), b_list) 
            cursor.execute(sql_cmd)
            rows_c = [x for x in cursor.fetchall()]
    
            counts_c = {}
            flags_c = {}
            for brand in selected_brands:
        
                counts_c[brand] = np.zeros(len(dates_t))
                flags_c[brand] = get_flag_dates(brand)
              
            for row in rows_c:
        
                counts_c[row[1]][dates_t.index(row[2])] = row[0]
    
            data = []

            for brand in selected_brands:
        
                if count_type == 'std':
                    sig = counts_c[brand].std()
                    if sig != 0:
                        cnts = (counts_c[brand]-np.median(counts_c[brand]))/sig
                    else:
                        cnts = counts_c[brand]
                else:
                    cnts = counts_c[brand]
                
                datum = dict(
                        type='scatter',
                        mode='lines+markers',
                        x=dates_t,
                        y=cnts,
                        name = brand,
                        line = dict(color= brand_colors_t[brand],
                            shape= 'linear'
                           ),
                            opacity = 1,
                            marker=dict(symbol='circle-open-dot'))
                data.append(datum)
            layout_d = layout
            layout_d['shapes'] = []
            for brand in selected_brands:
                if flags_c[brand]:
                    for rng in flags_c[brand]:
                        if rng[-1] != rng[0]:
                            shape_c = {
                                        'type': 'rect',
                                        'xref': 'x',
                                        'yref': 'paper',
                                        'x0': rng[0],
                                        'y0': 0,
                                        'x1': rng[1],
                                        'y1': 1,
                                        'fillcolor': brand_colors_t[brand],
                                        'opacity': 0.2,
                                        'line': {
                                                'width': 0,
                                                }
                                        }
                            layout_d['shapes'].append(shape_c)
                        else:
                            shape_c = {
                                        'type': 'line',
                                        'xref': 'x',
                                        'yref': 'paper',
                                        'x0': rng[0],
                                        'y0': 0,
                                        'x1': rng[1],
                                        'y1': 1,
                                        'opacity': 0.3,
                                        'line': {
                                                'width': 4,
                                                'color':brand_colors_t[brand]
                                                }
                                        }
                            layout_d['shapes'].append(shape_c)
                        
            layout_d['xaxis'] = xaxis_d
            fig = dict(data=data, layout=layout_d)
            return fig
        
        elif (day_mo == 'day' or day_mo == None) and flag == 0:
        
            sql_cmd = '''
            SELECT COUNT(DISTINCT ID), Brand,
            Tweet_Date FROM EWS_Tweet_Stream_Ph2
            WHERE Safety > %s
            AND Brand in %s
            GROUP BY Brand, Tweet_Date;''' %(str(tol), b_list) 
            cursor.execute(sql_cmd)
            rows_c = [x for x in cursor.fetchall()]
    
            counts_c = {}
    
            for brand in selected_brands:
        
                counts_c[brand] = np.zeros(len(dates_t))
                
            for row in rows_c:
        
                counts_c[row[1]][dates_t.index(row[2])] = row[0]
    
            data = []

            for brand in selected_brands:
        
                if count_type == 'std':
                    sig = counts_c[brand].std()
                    if sig != 0:
                        cnts = (counts_c[brand]-np.median(counts_c[brand]))/sig
                    else:
                        cnts = counts_c[brand]
                else:
                    cnts = counts_c[brand]
                
                datum = dict(
                        type='scatter',
                        mode='lines+markers',
                        x=dates_t,
                        y=cnts,
                        name = brand,
                        line = dict(color= brand_colors_t[brand],
                            shape= 'linear'
                           ),
                            opacity = 1,
                            marker=dict(symbol='circle-open-dot'))
                data.append(datum)
            layout_d = layout
            layout_d['shapes'] = []
            layout_d['xaxis'] = xaxis_d
            fig = dict(data=data, layout=layout_d)
            return fig
        
        elif day_mo == 'month':
            sql_cmd = '''
                 SELECT COUNT(DISTINCT ID), Brand, YEAR(Tweet_Date),
                 MONTH(Tweet_Date)  FROM EWS_Tweet_Stream_Ph2
                 WHERE Safety > %s
                 AND Brand in %s
                 GROUP BY Brand, YEAR(Tweet_Date),
                 MONTH(Tweet_Date);''' %(str(tol), b_list)
            cursor.execute(sql_cmd)
            rows_c = [x for x in cursor.fetchall()]
        
            counts_c = {}
        
            for brand in selected_brands:
            
                counts_c[brand] = np.zeros(len(months_t))
        
            for row in rows_c:
            
                cur_m = '-'.join([date_str(x) for x in row[-2:]])
                counts_c[row[1]][months_t.index(cur_m)] = row[0]
            
            data = []
        
            for brand in selected_brands:
        
                if count_type == 'std':
                    sig = counts_c[brand].std()
                    if sig != 0:
                        cnts = (counts_c[brand]-np.median(counts_c[brand]))/sig
                    else:
                        cnts = counts_c[brand]
                else:
                    cnts = counts_c[brand]
                
                datum = dict(
                        type='scatter',
                        mode='lines+markers',
                        x=months_t,
                        y=cnts,
                        name = brand,
                        line = dict(color= brand_colors_t[brand],
                            shape= 'linear'
                           ),
                                    opacity = 1,
                                    marker=dict(symbol='circle-open-dot'))
                data.append(datum)
            layout_m = layout
            layout_m['shapes'] = []
            layout_m['xaxis'] = xaxis_m
            fig = dict(data=data, layout=layout_m)
            return fig
        
        else:
            
            sql_cmd = '''
            SELECT COUNT(DISTINCT ID),
            BRAND,
            DATEADD(week, DATEDIFF(week, 0, Tweet_Date), 0)
            FROM EWS_Tweet_Stream_Ph2
            WHERE Safety > %s
            AND Brand in %s
            GROUP BY BRAND, DATEADD(week, DATEDIFF(week, 0, Tweet_Date), 0)
            ;''' %(str(tol), b_list)
            
            cursor.execute(sql_cmd)
            rows_c = [(x[0],x[1],x[-1].strftime('%Y-%m-%d')) 
                      for x in cursor.fetchall()]
            counts_c = {}
            
            for brand in selected_brands:
                
                counts_c[brand] = np.zeros(len(weeks_t))
                
            for row in rows_c:
                
                counts_c[row[1]][weeks_t.index(row[-1])] = row[0]
                
            data = []
            for brand in selected_brands:
        
                if count_type == 'std':
                    sig = counts_c[brand].std()
                    if sig != 0:
                        cnts = (counts_c[brand]-np.median(counts_c[brand]))/sig
                    else:
                        cnts = counts_c[brand]
                else:
                    cnts = counts_c[brand]
                datum = dict(
                        type='scatter',
                        mode='lines+markers',
                        x=weeks_t,
                        y=cnts,
                        name = brand,
                        line = dict(color= brand_colors_t[brand],
                            shape= 'linear'
                           ),
                            opacity = 1,
                            marker=dict(symbol='circle-open-dot'))
                data.append(datum)
            layout_w = layout
            layout_w['shapes'] = []
            layout_w['xaxis'] = xaxis_d
            fig = dict(data=data, layout=layout)
            return fig       
            
    else:

        sql_cmd = '''
            SELECT COUNT(DISTINCT [Report No.]), [Product Type],
            [Report Date]  FROM CPSC_Safer_Prod
            WHERE [Safety Relevance] > %s
            AND [Product Type] in %s
            GROUP BY [Product Type] , [Report Date];''' %(str(tol), b_list) 
        cursor.execute(sql_cmd)
        rows_c = [x for x in cursor.fetchall()]
    
        counts_c = {}
    
        for brand in selected_brands:
        
            counts_c[brand] = np.zeros(len(months_sp))
    
        for row in rows_c:
        
            counts_c[row[1]][months_sp.index(row[2][:-3])] += row[0]
    
        data = []

        for brand in selected_brands:
        
            if count_type == 'std':
                sig = counts_c[brand].std()
                if sig != 0:
                    cnts = (counts_c[brand]-np.median(counts_c[brand]))/sig
                else:
                    cnts = counts_c[brand]
            else:
                cnts = counts_c[brand]
            datum = dict(
                type='scatter',
                mode='lines+markers',
                x=months_sp,
                y=cnts,
                name = brand,
                line = dict(color= brand_colors_sp[brand],
                            shape= 'linear'
                           ),
                opacity = 1,
                marker=dict(symbol='circle-open-dot'))
            data.append(datum)
        layout_m = layout
        layout_m['shapes'] = []
        layout_m['xaxis'] = xaxis_m
        fig = dict(data=data, layout=layout)
        return fig       
        
@app.callback(
    dash.dependencies.Output('tweet-table', 'children'),
    [dash.dependencies.Input('ts-graph', 'clickData'),
     dash.dependencies.Input('ts-graph', 'figure'),
     dash.dependencies.Input('tol-slider', 'value'),
     dash.dependencies.Input('table-choice', 'value'),
     dash.dependencies.Input('day-month', 'value'),
     dash.dependencies.Input('main-selec', 'value')])  
def generate_table(clickData, fig, tol, tbl, day_mo, ds):
    brand_c = (fig['data'][(clickData['points'][0])['curveNumber']])['name']
    date_c = (clickData['points'][0])['x']
    
    if ds == 'twitter' and day_mo == 'day':

        sql_cmd = '''SELECT Text, Tweet_Date FROM EWS_Tweet_Stream_Ph2
             WHERE Safety > %s
             AND Brand = '%s'
             AND Tweet_Date = CAST('%s' AS DATE)
             ;''' %(str(tol), brand_c.replace("'", "''"), date_c)
    
        cursor.execute(sql_cmd)
        
    elif ds == 'twitter' and day_mo == 'month':
        
        split_date = date_c.split('-')
        year = split_date[0]
        month = str(int(split_date[1]))
       
        sql_cmd = '''SELECT Text, Tweet_Date FROM EWS_Tweet_Stream_Ph2
             WHERE Safety > %s
             AND Brand = '%s'
             AND YEAR(Tweet_Date) = %s
             AND MONTH(Tweet_Date) = %s
             ;''' %(str(tol), brand_c.replace("'", "''"), year, month)
             
        cursor.execute(sql_cmd)  
    
    elif ds == 'twitter' and day_mo == 'week':
        
        sql_cmd ='''
          SELECT Text, Tweet_Date FROM EWS_Tweet_Stream_Ph2
          WHERE Safety > %s
          AND Brand = '%s'
          AND DATEADD(week,DATEDIFF(week,0,Tweet_Date),0) = CAST('%s' AS DATE)
          ;''' %(str(tol), brand_c.replace("'", "''"), date_c)
             
        cursor.execute(sql_cmd)  
        
    else:
        
        split_date = date_c.split('-')
        year = split_date[0]
        month = str(int(split_date[1]))
        sql_cmd = '''
            SELECT [Incident Description], [Report Date] FROM CPSC_Safer_Prod
            WHERE [Safety Relevance] > %s
            AND [Product Type] = '%s'
            AND YEAR([Report Date]) = %s
            AND MONTH([Report Date]) = %s
            ;''' %(str(tol), brand_c.replace("'", "''"), year, month)
             
        cursor.execute(sql_cmd) 
    if tbl == 'tweets':
        
        tweets_c = [x for x in cursor.fetchall()]
        
        child = [html.Tr([html.Th('Date:'), html.Th('Brand:'),
                          html.Th('Tweet Text:')])] 
        child += [html.Tr([html.Td(tweets_c[i][1]),
                           html.Td(brand_c),html.Td(tweets_c[i][0])])
                           for i in range(len(tweets_c))]
    
        return child
    
    else:
        
        tweets_c = [process_tweet(x[0]) for x in cursor.fetchall()]
        df = summary_extract(tweets_c)
        
        child = [html.Tr([html.Th(col) for col in df.columns])]

        child += [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
                           for i in range(len(df))]
    
        return child


if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
