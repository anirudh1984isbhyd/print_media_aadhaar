## This is the core function file used across 
## The stopplaces function is used to eliminate state / city names at the start of the news article sentence
## Current active dir is "/Users/anirudhsyal/Desktop/Hindu_news" where all files are stored and uploaded from. Please modify 
## The web extracted corpus is read a data frame
## install the following ( pip or conda )
## pip install --upgrade gensim
## conda install scikit-learn for TSNE / Kmeans / LDA 
## refer : http://scikit-learn.org/stable/install.html
## pip install lda
## pip install pyldavis
## pip install tqdm


import re                                                        
import os 
import pandas as pd
import numpy as np
from pandas import ExcelWriter


import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
#%matplotlib inline

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cluster import MiniBatchKMeans

from sklearn.manifold import TSNE

active_dir = "/Users/anirudhsyal/Desktop/Hindu_news"
os.chdir (active_dir )


global count_proc
count_proc = 0
#################################################
def stopplaces():
    stop_places_f_name = "location_stop_words.txt"
    stop_places = list(open( stop_places_f_name, 'r'))
    lis_iterator = 0
    while lis_iterator <len(stop_places):
        stop_places[lis_iterator] = re.sub(r'\n', '', stop_places[lis_iterator] )
        stop_places[lis_iterator] = re.sub(r'[^a-zA-Z\s]', '', stop_places[lis_iterator] )
        stop_places[lis_iterator] = re.sub(r'\s\s+', ' ', stop_places[lis_iterator] )

        lis_iterator = lis_iterator + 1

    return stop_places

def clean_corpus_1(text): # read text and return a list of clean paragraphs
   
    import re                                                        
    import os 
    import pandas as pd
    
    #fname= input('Input corpus filename - e.g :/Users/anirudhsyal/Desktop/reviews.txt')
    #fname = "/Users/anirudhsyal/Desktop/nklrev_2.txt"
    #text = open(fname, 'r').readlines()        
    #text = [' I am happy & AT&T and AT &T working', ' I $ $40 is great']
    #print('no of pararagraphs in texts your file=', len(text), 'stored as ', type(text))
                                  
    
    if(len(text) == 0):
        text = 0
        return text
    
    replacement_patterns = [
     (r'won\'t', 'will not'),
     (r'can\'t', 'cannot'),
     (r'i\'m', 'i am'),
     (r'ain\'t', 'is not'),
     (r'(\w+)\'ll', '\g<1> will'),
     (r'(\w+)n\'t', '\g<1> not'),
     (r'(\w+)\'ve', '\g<1> have'),
     (r'(\w+)\'s', '\g<1> is'),
     (r'(\w+)\'re', '\g<1> are'),
     (r'(\w+)\'d', '\g<1> would')]


    # Subdivide text into paragraphs
    # convert to lower case
    # eliminate HTML type elements within '< > '
    # eliminate all ? , ! , ?? , #  type tokens from the paragrpah
    # input a space between a digit and a word boundary
    # do not tokenzie M.R or U.S.A  type tokens from the paragrpah - Keep them as is
    # eliminate words recurring after one another
    # remove dangling spaces, collapse one or more spaces into one space #
    # substitute known short forms with actual words from the defined list replacement patterns
    # remove remaining '  or " characters
    
    text = text.lower() 

    if 'aadhaar' in text or 'aadhar' in text or 'aaadhar' in text or 'aaadhaar' in text:
        text = re.sub(r'(<.*?>)', ' ', text)  
        text = re.sub(r'(\d+)(\-|:)(\d+)', ' ' , text)
        text = re.sub(r'(\&|\%|\$|\.|\?|!|#)(\s|\&|\.|\%|\$|\?|!|#)*(\1+)', r' \1 ' , text)  

        text = re.sub(r'(\%)', ' percent ' , text) 
        text = re.sub(r'(\$)', ' dollars ' , text)  

        text = re.sub(r'[^a-z0-9\s\.\'\"\-\_\&]', ' ',text)  
        
        text  = re.sub(r'(\d)\s([a-z])([\s\.])(?![a-z])',r'\1\2\3' , text )  
        text  = re.sub(r'(\w\w+\s*)(\.)(\w\w+|i|\d+)',r'\1 \2 \3' , text )  
        text = re.sub(r'\s+[\-\&\_]', ' ', text)  

        text  = re.sub(r'(\d)([a-z]{2,})',r'\1 \2' , text )
        text  = re.sub(r'(\d)\s([a-z])([\s\.])(?![a-z])',r'\1\2\3' , text )  

        text = re.sub(r'\.\.+', '. ', text)
        text = re.sub(r'xa0',' ', text)

        
        text = re.sub(r'aadhar', 'aadhaar' , text)
        text = re.sub(r'aaadhar', 'aadhaar' , text)
        text = re.sub(r'aaadhaar', 'aadhaar' , text)
        text = re.sub(r'\s+pd(s)*\s+', 'public_distribution ' , text)
        text = re.sub(r'\s+',' ', text)  
        
        for pattern in replacement_patterns: 
            text = re.sub (pattern[0],  pattern[1] , text)
            
        text = re.sub(r'[^a-z0-9\s\.\&\_\-]', ' ', text)  
        text = re.sub(r'(\be)(\s|\-|\_)*(mail|watch|commerce|bay)', r'\1\3' , text)
        text = re.sub(r'(\bi)(\s|\-|\_)*(phone|watch|tunes)', r'\1\3' , text)
        text = re.sub(r'dbt', ' direct benefit transfer ' , text)
        text = re.sub(r'm(g)*nrega', ' nrega ' , text)
        text = re.sub(r'per cent', ' percent ' , text)
        text = re.sub(r'\.the', '. the' , text)
        text = re.sub(r'(database\.|\s*)(googletag)(\.cmd\.push|\.display)', ' ' , text)
        text = re.sub(r'adslotnativevideo', ' ' , text)
        text = re.sub(r'food\s+grain(s)*', 'foodgrains' , text)

    else:
        text = 0

    return text 

def clean_corpus_2 (text, stop_places): 
    text = text.strip()
    iter = True
    while iter:
        for junk in stop_places:
            if text.startswith(junk):
                text = text[len(junk):]
                iter = False
        iter = False
        
    return text

def clean_corpus_3(text,stop_words): 
    global count_proc
    count_proc = count_proc +1
    import re
    new_lis =text.split()
    word_list = []
    for word in new_lis:
        word = re.sub(r'\s+', '', word) 
        word = re.sub(r'^[^a-z0-9]+', '', word)
        word = re.sub(r'[^a-z0-9]+$', '', word)
        word = re.sub(r'\.com|\.in|\.net', '', word)
        word = re.sub(r'\.',"" , word)
        appen_d = True
    
        if word in stop_words or word.isdigit() or len(word)< 2:
            appen_d = False  
        elif re.findall(r'[\&\-\_]', word) or len(word) == 2:
            appen_d = True

        else: 
            word = lemmatize(word)

        if appen_d:
            word_list.append(word)
    
    text = ' '.join(word_list) 
    count_proc_all = [100, 400, 500, 1000, 1500, 2000, 2500, 3000]
    if count_proc in count_proc_all:
        print(count_proc)

    return text

def clean_corpus_4(text):
    repl_words_lis = ['accept','affect','generate','challenge','concern','consolidate',
                        'prescribe','absolute','alleged','apparent','careful','complete',
                         'digital','direct','elder','eventual','exact','final','formal',
                         'full','heavy','immediate','initial','direct','particular','physical',
                         'public','quick','recent','reported','simple','subsequent',
                         'successful','virtual','wide','unauthorise', 'unfortunate']
 
    new_lis =text.split()
    
    for i in range(0,len(new_lis)):
        for elem in repl_words_lis:
            if elem in new_lis[i]:
                new_lis[i] = elem

    text = ' '.join(new_lis)
    return text

def clean_corpus_5_replace_unigrams_with_bi_grams(text, bi_gram_lis,tri_gram_lis ):
    
    bi_gram_lis = bi_gram_lis
    tri_gram_lis  = tri_gram_lis
    new_lis =text.split()
    ret_list = []
    i=0
    while i < len(new_lis):
        if i == len(new_lis)-1:
            ret_list.append(new_lis[i])
            i = i+1
        else:
            query_bi = new_lis[i] + '_' + new_lis [i+1]
            if i <= len(new_lis) -3:
                query_tri = query_bi + '_' + new_lis [i+2]

                if query_tri in tri_gram_lis:
                    ret_list.append(query_tri)
                    i = i+3 
                elif query_bi in bi_gram_lis: 
                    ret_list.append(query_bi)
                    i = i+2
                else:
                    ret_list.append(new_lis[i])
                    i=i+1
            elif query_bi in bi_gram_lis:
                ret_list.append(query_bi)
                i = i+2
            else:
                ret_list.append(new_lis[i])
                i=i+1


    text = ' '.join(ret_list)  

    return text

def stopwords():  
# import NLTK stopwords and user defined stopwords if available and return a list 

    from nltk.corpus import stopwords
    #import pandas as pd
    #from pandas import ExcelWriter
    #import os 
    #import re
    # genereic stopwords found in NLTK
    stop_words = stopwords.words('english')
    # user defined stop words in text format handles both Y and N - add stopwords as text file only
    repeat_input = True
    while repeat_input: 
        #Response = input('Do you have a specefic stopwords text file - input  Y for yes or N for no ')
        Response = 'y'
        if Response.lower() == 'y':
            #my_stop_words_fname = input('Input user defined stop words in text file format')
            my_stop_words_fname = "more_stop_words.txt"
            my_stop_words = list(open( my_stop_words_fname, 'r'))
            for word in my_stop_words:
                stop_words.append(word)
            repeat_input = False
        if Response.lower() == 'n':
            repeat_input = False                

    lis_iterator = 0
    while lis_iterator <len(stop_words):
        stop_words[lis_iterator] = re.sub(r'\n', '', stop_words[lis_iterator] )
        stop_words[lis_iterator] = re.sub(r'[^a-zA-Z]', '', stop_words[lis_iterator] )
        lis_iterator = lis_iterator + 1
    
    new_append = ['said', 'although','though', 'actually', 'hi', 'hello',  'it', 'get' ,'k.k','function',
                  'commentsthe', 'said', 't.co', 'official', 'advocate',  'shyam', 'gupta', 'roy', 'kv', 'raja'
                  'commentshe', 'commentsmr', 'chidambaram', 'besides','ariyalur', 'devi', 'kurnool','minister dikshit'
                 'commentsa', 'dr', 'chandrachud ashok', 'today', 'chandra', 'kiran reddy','sp','anand','anil', 
                 'po', 'krishna rao', 'mv','ii', 'one', 'two', 'day', 'us']
    
    
    
    for new in new_append:
        stop_words.append(new)

    stop_words = list(set(stop_words))
    return stop_words

def remove_more_stopwords(text):  
    new_append = ['use','give', 'name', 'naidu', 'percent', 'reddy', 'need', 'may', 
                  'move', 'another', 'place', 'call', 'look', 'past', 'others', 'ask', 'http', 
                  'chief_minister_dikshit', 'station', 'tiruchi', 'science', 'exhibition', 'stadium', 'training',
                  'tamil_nadu', ' join', 'thomas', 'say', 'see', 'pathanamthitta' , 'along', 'patel', 'bhola', 
                  'none', 'sadasivan', 'sit' , 'four', 'pm','govindaraopet', 'april', 
                  'koyas', 'matunga', 'murali', 'dilshad', 'islamu' , 'singh',  'pawar', 'malakpur',
                  'ponda', 'nirancal', 'visakhapatnam','vangal', 'vijaya', 'shivampet', 
                  'tiruvallur', 'ranjani', 'bhavani','joshi', 'sagar', 'tendulkar', 
                  'mekhliganj', 'nagadevatha', 'mangu' , 'ranjana' , 
                  'ankita' , 'mahesh' , 'kle' , 'ara' , 
                  'budigajangalu' , 'oommen' , 'usilampatti' ,
                  'ammini' , 'venkat', 'manachanallur' , 'moovarayanpalayam', 
                  'radhanpur' , 'khushbu' ,'shendul', 'sangeeta' , 'cavinkare', 'biju', 
                  'amar','kunool', 'put', 'pintu', 'hai', 'bhatkhande', 
                  'rythu', 'vazhakkai', 'madam' ,'sir', 'go','vivek', 'arulnidhi' ,'kannan', 
                  'ha', 'tharoor', 'michael' ,'turn' , 'kaka','ramu',
                  'first','ranganath', 'privatisationthere', 'sanashree' ]
    import re
    new_lis =text.split()
    word_list = []
    for word in new_lis: 
        if word not in new_append :
            word_list.append(word)
            
    text = ' '.join(word_list) 
    
    return text

def part_of_speec_tag (treebank_tag):
    import nltk
    from nltk.corpus import wordnet
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None 
        

def lemmatize(word):
    
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    lemmatizer = WordNetLemmatizer()
    word = word_tokenize(word)
    tagged = nltk.pos_tag(word)
    
    for word, tag in tagged:
        wntag = part_of_speec_tag(tag)
        if wntag is None:
            word = lemmatizer.lemmatize(word) 
        else:
            word = lemmatizer.lemmatize(word, pos=wntag) 
    return word

def tokenizer_tf_idf(text):
    word_vector = text.split()
    return word_vector


def text_to_ngrams(paragraph_list,stop_words, n):  
    
    from nltk.util import ngrams
    import pandas as pd
    import numpy as np
    import os 
    
    stop_words = stop_words
    paragraph_list = paragraph_list
    no_of_paras = len(paragraph_list)
    n_grams = []
    para_iterator = 0
    for paragraphs in paragraph_list:
        tokens = list(ngrams(paragraphs.split(), n))
        para_iterator = para_iterator +1
        token_iterator = 0
        while token_iterator < len(tokens):
            ngram = ''
            new_lis = list(tokens[token_iterator])
            i = 0
            while i < n:
                new_lis[i] = clean_word(new_lis[i])  
                if len(new_lis[i]) == 0 or new_lis[i] in stop_words:
                    token_iterator=token_iterator+i+1
                    i = n
                    ngram = ''
                else:
                    if i==0:
                        ngram =  new_lis[i]
                    else:
                        ngram =  ngram + "-" + new_lis[i]
                    i=i+1
                    
            if len(ngram) >0:
                n_grams.append(dict( doct_no =para_iterator, ngram = ngram ))                
                token_iterator=token_iterator+1
    n_grams = pd.DataFrame(n_grams)
    
    n_grams['freq_in_para'] = n_grams.groupby(['doct_no', 'ngram' ])['ngram'].transform('count')
    n_grams_temp = n_grams.copy(deep=True)
    n_grams_temp.drop_duplicates(inplace = True)
    n_grams_temp['presence_in_docs'] = n_grams_temp.groupby(['ngram' ])['doct_no'].transform('count')
    n_grams_temp.drop(['freq_in_para', 'doct_no' ], axis = 1, inplace = True)
    n_grams_temp.drop_duplicates(inplace = True)
    n_grams = pd.merge(n_grams,n_grams_temp, on = 'ngram')
    
    if n ==2:
        n_grams_temp = n_grams[n_grams['presence_in_docs']>=4]
        n_grams_temp = n_grams_temp[n_grams_temp['freq_in_para']>=2]
   
    if n >2:
        n_grams_temp = n_grams[n_grams['presence_in_docs']>=4]
        n_grams_temp = n_grams_temp[n_grams_temp['freq_in_para']>1]

    g = n_grams_temp.groupby(['doct_no'])
    n_grams = []
    col_name =str(n) + '_'+ 'grams'
    for name, group in g:
        s = group.ngram.tolist()
        n_grams.append(dict(doct_no =name,lis =  s))
    n_grams = pd.DataFrame(n_grams)
    n_grams.rename(columns={'lis': col_name}, inplace=True)

    return n_grams

def final_tokenzied_list(df, colname, ngrams):
    stop_words = stopwords()
    paragraph_list = df[colname].tolist()
    df['tokens'] = df[colname].map(tokenizer_tf_idf)
    
    n_grams=2
    while n_grams < ngrams+1:
        col_name =str(n_grams) + '_'+ 'grams'
        n_gram_df = text_to_ngrams(paragraph_list,stop_words,n_grams)
        df = pd.merge(df,n_gram_df, on = 'doct_no',how='left')
        df['tokens'] = np.where(df[col_name].isnull(), df['tokens'], df['tokens'] + df[col_name])
        df.drop([col_name], axis = 1, inplace = True)
        n_grams = n_grams +1
    
    return df

def TF_IDF(df,colname, min_df, ngram_range_tuple):
    from sklearn.feature_extraction.text import TfidfVectorizer
    a = list(df[colname])
    vectorizer = TfidfVectorizer(min_df=min_df, max_features=10000, tokenizer=tokenizer_tf_idf,stop_words =  stopwords(), ngram_range = ngram_range_tuple)
    vz = vectorizer.fit_transform(a)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']

    return tfidf

def clean_non_news_body_cols(text):
    import re                                                                                     
    if(len(text) == 0):
        text = 0
        return text
    
    replacement_patterns = [
     (r'won\'t', 'will not'),
     (r'can\'t', 'cannot'),
     (r'i\'m', 'i am'),
     (r'ain\'t', 'is not'),
     (r'(\w+)\'ll', '\g<1> will'),
     (r'(\w+)n\'t', '\g<1> not'),
     (r'(\w+)\'ve', '\g<1> have'),
     (r'(\w+)\'s', '\g<1> is'),
     (r'(\w+)\'re', '\g<1> are'),
     (r'(\w+)\'d', '\g<1> would')]

    text = text.lower() 

 
    text = re.sub(r'(<.*?>)', ' ', text)  
    text = re.sub(r'(\&|\%|\$|\.|\?|!|#)(\s|\&|\.|\%|\$|\?|!|#)*(\1+)', r' \1 ' , text)  
    text = re.sub(r'(\%)', ' percent ' , text) 
    text = re.sub(r'(\$)', ' dollars ' , text)  

    text = re.sub(r'[^a-z0-9\s\.\'\"\-\_\&]', ' ',text)  
    text = re.sub(r'\s+[\-\&\_]', ' ', text)  

    text  = re.sub(r'(\d)([a-z]{2,})',r'\1 \2' , text )
    text  = re.sub(r'(\d)\s([a-z])([\s\.])(?![a-z])',r'\1\2\3' , text )  

    text = re.sub(r'\.\.+', '. ', text)
    text = re.sub(r'xa0',' ', text)

    text = re.sub(r'aadhar', 'aadhaar' , text)
    text = re.sub(r'\s+',' ', text)  

    for pattern in replacement_patterns: 
        text = re.sub (pattern[0],  pattern[1] , text)

    text = re.sub(r'[^a-z0-9\s\.\&\_\-]', ' ', text)  
    text = re.sub(r'(\be)(\s|\-|\_)*(mail|watch|commerce|bay)', r'\1\3' , text)
    text = re.sub(r'(\bi)(\s|\-|\_)*(phone|watch|tunes)', r'\1\3' , text)
    text = re.sub(r'dbt', ' direct benefit transfer ' , text)
    text = re.sub(r'mgnrega', ' nrega ' , text)
    text = re.sub(r'per cent', ' percent ' , text)
    text = re.sub(r'\.the', '. the' , text)
    text = re.sub(r'(database\.|\s*)(googletag)(\.cmd\.push|\.display)', ' ' , text)
    text = re.sub(r'adslotnativevideo', ' ' , text)


    return text 

def count_token(tok_list, token):
    from collections import Counter
    counter = Counter(tok_list)
    counter_out = counter[token]
    return counter_out

def clean_check (text):
    text = re.sub(r'[^a-z]', ' ',text)  
    text = re.sub(r'\s\s+', ' ', text)  
    return text
  
def clear_more_duplicates(df,on_col):

    df.reset_index(inplace=True,drop=True)
    df['index1'] = df.index
    sq = pd.DataFrame(df[on_col].copy(deep=True))

    sq.drop_duplicates(inplace=True)
    sq['index1'] = sq.index

    merged = pd.merge(sq,df, on=['index1', on_col])
    merged.drop(['index1'], axis = 1, inplace = True)
    merged.reset_index(inplace=True,drop=True)
    merged['doct_no'] = range(len(merged))
    merged['doct_no'] = merged['doct_no'] +1
    return merged

def date_time_hindu(df, col_name):
    
    def my_year(text):
        s= text.split()
        return s[2]

    def my_months(text):
        s= text.split()
        m = {'jan': 1,'feb': 2,'mar': 3,
                'apr':4,'may':5,'jun':6,
                'jul':7,'aug':8,'sep':9,
                'oct':10, 'nov':11,'dec':12 }
        month = s[0]
        month = month[:3]
        try:
            out = m[month]
            return out
        except:
            raise ValueError('Not a month')
    
    def my_day(text):
        s= text.split()
        return s[1]
    
    def my_num_dates(text):
        s= text.split()
        return s[1]

    
    df['year']=  df[col_name].apply(my_year)
    df.year = df.year.astype(np.int64)
    df['month']= df[col_name].map(my_months)
    df.month = df.month.astype(np.int64)
    df['day'] =  df[col_name].map(my_day)
    df.day = df.day.astype(np.int64)
    df['num_date']  = 10000* df['year']+ 100*df['month'] + df['day']
    df.drop(['year', 'month' , 'day' ], axis = 1, inplace = True)

    return df

def remove_more_junk_hindu_posts (text, compare_list):
    if text in compare_list:
        return 0
    else:
        return 1

def count_most_common(df,tok_col_name, top_n_counts):
    from collections import Counter
    def count_a(token_list):
        from collections import Counter
        counter = Counter(token_list)
        check = len(list(set(token_list)))
        if len(list(set(token_list)))<=top_n_counts:
            counter_tuple = counter.most_common(check)
            append_count = check
        else:
            counter_tuple = counter.most_common(top_n_counts)
            append_count = top_n_counts
        return_list = []
        for i in range (0,append_count):
            return_list.append(counter_tuple[i][0])
        return return_list
    n_count_col_name = 'top'+ '_' + str(top_n_counts)+'_'+'counts'
    df[n_count_col_name]= df[tok_col_name].map(count_a)
    return df

def co_occurence_matrix(df, text_col_name,top_n_counts):
    df['tokens'] = df[text_col_name].map (tokenizer_tf_idf)
    df = count_most_common(df, 'tokens',top_n_counts)
    n_count_col_name = 'top'+ '_' + str(top_n_counts)+'_'+'counts'
    keywords_array=[]
    for index, row in df.iterrows():
        keywords=row[n_count_col_name]
        for kw in keywords:
            keywords_array.append((kw, row[n_count_col_name]))
    kw_df = pd.DataFrame(keywords_array).rename(columns={0:'keyword', 1:'keywords'})
    
    from collections import OrderedDict
    document = kw_df.keywords.tolist()
    names = kw_df.keyword.tolist()
    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)
    for l in document:
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1

    co_occurence_matrix = pd.DataFrame.from_dict(occurrences )
    co_occurence_matrix.to_csv('co-occurancy_matrix.csv')
    return co_occurence_matrix

 
def K_m_cluster (num_clusters, df, text_col_name, top_n_terms ):
    
    import numpy as np
    import pandas as pd
    import bokeh.plotting as bp
    from bokeh.models import HoverTool, BoxSelectTool
    from bokeh.plotting import figure, show, output_notebook

    from sklearn.feature_extraction.text import TfidfVectorizer
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.cluster import MiniBatchKMeans
    
    vectorizer = TfidfVectorizer(min_df=0.01, max_features=10000, tokenizer=tokenizer_tf_idf,stop_words =  stopwords(), ngram_range = (1,1))
    
    vz = vectorizer.fit_transform(list(df[text_col_name]))
    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
    kmeans = kmeans_model.fit(vz)
    kmeans_clusters = kmeans.predict(vz)
    kmeans_distances = kmeans.transform(vz)

    high_impact_cluster_terms = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=1)
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['text'] = df[text_col_name]

    
    cumpercent = 0
    
    Km_file = []
    for i in range(num_clusters):
        clust_num  = i
        percent_in_doc = 100*len(kmeans_df[kmeans_df['cluster']==i])/ len(kmeans_df)
        hot_terms = ''
        for j in high_impact_cluster_terms[i, :top_n_terms]:
            hot_terms += terms[j] + ' | '
        Km_file.append(dict(clust_num = clust_num, 
                            clust_terms = hot_terms, percent_in_doc = round(percent_in_doc, 2)))
    Km_file = pd.DataFrame(Km_file)
    name_csv = 'Km_clusters' + '_' +  str(num_clusters) + '.csv'
    Km_file.to_csv(name_csv)
        
    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
    "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
    "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
    "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

    kmeans_df['colors'] = colormap[kmeans_clusters]

    plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)
    

    plot_kmeans.scatter(x='x', y='y', color='colors', source=kmeans_df)
    
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips={"text": "@text", "cluster":"@cluster"}
    show(plot_kmeans)

    
    return  kmeans_df