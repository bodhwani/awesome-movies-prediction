import urllib
from urllib2 import Request, urlopen, URLError
import json
import pandas as pd
import os
import requests
from lxml import html
import sys
import xlsxwriter
import csv
import xlrd
array = []
name_movie= []
rating_movie = []
genre_movie =  []
plot_movie = []
final = [[1,2,3,4,5,6,7,8,9],['MOVIES NAME','RATINGS','GENRE','PLOT', 'release_year','POSTER_URL','DIRECTOR', 'STARS','MOVIES_ID']]
movie_names = []
genre = []
plot = []
ratings = []
year = []
poster_url = []
director = []
stars = []
d = {
    'data' : {
    'genre' : 0,
    'plot' : 0,
    'rating' : 0,
    'stars' : 0
    }
}

def get_imdb_id(input):
    query = urllib.quote_plus(input)
    url = "http://www.imdb.com/find?ref_=nv_sr_fn&q="+query+"&s=all"
    page = requests.get(url)
    tree = html.fromstring(page.content)
    if"No results" in (tree.xpath('//h1[@class="findHeader"]/text()')[0]):
        imdb_id = "tt00000"
    else:
        imdb_id=(tree.xpath('//td[@class="result_text"]//a')[0].get('href'))
        imdb_id = imdb_id.replace('/title/','')
        imdb_id = imdb_id.replace('/?ref_=fn_al_tt_1','')
    return (imdb_id)

def get_info(id):  
    omdb_request = Request("http://theapache64.xyz:8080/movie_db/search?keyword="+id)
    response = urlopen(omdb_request)
    data = response.read()
    d=json.loads(data)

    print "Data as json : ",d
    
    print "\n\n"
    

    if (d["error"] ==  True) :
        message = "No results found"
        genre.append(message)
        plot.append(message)
        ratings.append(message)
        year.append(message)
        poster_url.append(message)
        director.append(message)
        stars.append(message)
           
    else:
        if(len(d["data"]) == 9):
            genre.append(d["data"]["genre"])
            plot.append(d["data"]["plot"])
            ratings.append(d["data"]["rating"])
            year.append(d["data"]["year"])
            poster_url.append(d["data"]["poster_url"])
            director.append(d["data"]["director"])
            stars.append(d["data"]["stars"])           
        else:
            message = "No results found"
            genre.append(message)
            plot.append(message)
            ratings.append(message)
            year.append(message)
            poster_url.append(message)
            director.append(message)
            stars.append(message)
        pass
      
def main():
    filepath = "/Users/Vinit/Documents/PROJECTS_MY/IMDB_MYPROJECT/extras/testing"
    print("Processing...")
    for file in os.listdir(filepath):
        get_info(get_imdb_id(file))
        movie_names.append(file)
    array.append(movie_names)
    array.append(ratings)
    array.append(genre)
    array.append(plot)
    array.append(year)
    array.append(poster_url)
    array.append(director)
    array.append(stars)
    
    for i in range(len(array[0])):
        newarray = []    
        newarray.append(array[0][i])
        newarray.append(array[1][i])
        newarray.append(array[2][i])
        newarray.append(array[3][i])
        newarray.append(array[4][i])
        newarray.append(array[5][i])
        newarray.append(array[6][i])
        newarray.append(array[7][i])
        newarray.append(i)
        
        
        if(newarray[0] == ".DS_Store"):
            continue
        final.append(newarray)
    print "this is final",final

    workbook = xlsxwriter.Workbook('dataset.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range(0,len(final)):
        worksheet.write('A'+str(i), final[i][0])
        worksheet.write('B'+str(i), final[i][1])
        worksheet.write('C'+str(i), str(final[i][2]))
        worksheet.write('D'+str(i), str(final[i][3]))
        worksheet.write('E'+str(i), final[i][4])
        worksheet.write('F'+str(i), str(final[i][5]))
        worksheet.write('G'+str(i), str(final[i][6]))
        worksheet.write('H'+str(i), str(final[i][7]))
        worksheet.write('I'+str(i),final[i][8])
    workbook.close()

    print("Successfully Created Excel file in the same directory in which your python script is present")

if __name__ == "__main__":
    main()




import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import urllib
from urllib2 import Request, urlopen, URLError
import json
import pandas as pd
import os
import requests
from lxml import html
import sys
import xlsxwriter


import json
import six.moves.urllib
import json
import requests
from requests.auth import HTTPBasicAuth


#876877979868767987798687tugnjukm

#/Users/Vinit/Documents/PROJECTS_MY/IMDB_MYPROJECT/extras




data = pd.read_csv('testing.csv')

# print data.head()
# print data.describe()
# ids =  data['MOVIES_ID']
# print "\n"
# print ids    
    
# data.plot(kind='bar')
# plt.ylabel('Ratings')
# plt.xlabel('Movies index')
# plt.title('Movies graph')

fig = plt.figure(figsize=[10, 10])

ax1 = fig.add_subplot(111)

data.plot(x="release_year", y=["MOVIES_ID","RATINGS"], kind="bar")


df=data

# ax = df.plot(x="Movies ratings", y="Ratings", kind="bar")
# df.plot(x="Movies ratings", y="Ratings", kind="bar", ax=ax, color="C2")
# df.plot(x="Movies ratings", y="Genre", kind="bar", ax=ax, color="C3")

ax = df.plot(x="release_year", y="MOVIES_ID", kind="bar")
df.plot(x="release_year", y="MOVIES_ID", kind="bar", ax=ax, color="C2")
df.plot(x="release_year", y="RATINGS", kind="bar", ax=ax, color="C3")


plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("whitegrid")
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0
mpl.rcParams['lines.linewidth'] = 20

fig = plt.figure(figsize=[10, 10])

tmdb_df = pd.read_csv("/Users/Vinit/Documents/PROJECTS_MY/IMDB_MYPROJECT/extras/testing.csv")
# Checking the data types and total number of data points before starting the analysis. 
tmdb_df.info()
#print tmdb_df.head()


# Obtaining a list of genres
genre_details = list(map(str,(tmdb_df['GENRE'])))
genre = []
for i in genre_details:
    split_genre = list(map(str, i.split(',')))
    for j in split_genre:
        if j not in genre:
            genre.append(j)
# printing list of seperated genres.
# print(genre)


# minimum range value
min_year = tmdb_df['release_year'].min()
# maximum range value
max_year = tmdb_df['release_year'].max()
# print the range
# print(min_year, max_year)




# Creating a dataframe with genre as index and years as columns
genre_df = pd.DataFrame(index = genre, columns = range(int(min_year), int(max_year) + 1))
# to fill not assigned values to zero
genre_df = genre_df.fillna(value = 0)
# print (genre_df.head())



# list of years of each movie
year = np.array(tmdb_df['release_year'])
# print "year is ",year
# index to access year value

z = 0
for i in genre_details:
    split_genre = list(map(str,i.split(',')))
    
#     print "split_genre is",split_genre
    for j in split_genre:
        if(j != "nan"):
            genre_df.loc[j, year[z]] = genre_df.loc[j, year[z]] + 1
    z+=1
genre_df



# number of movies in each genre so far.
genre_count = {}
genre = []
for i in genre_details:
    split_genre = list(map(str,i.split('|')))
    for j in split_genre:
        if j in genre:
            genre_count[j] = genre_count[j] + 1
        else:
            genre.append(j)
            genre_count[j] = 1
gen_series = pd.Series(genre_count)
# pi chart
gen_series = gen_series.sort_values(ascending = False)
label = list(map(str,gen_series[0:10].keys()))
label.append('Others')
gen = gen_series[0:10]
sum = 0
for i in gen_series[10:]:
    sum += i
gen['sum'] = sum

ax1 = fig.add_subplot(111)


#fig1, ax1 = plt.subplots()



ax1.pie(gen, labels = label, autopct = '%1.1f%%', startangle = 140)
ax1.axis('equal')
plt.title("Percentage of movies in each genre between 1960 and 2015")
plt.show()




#from urllib.request import Request, urlopen
import urllib

from urllib2 import Request, urlopen, URLError

from bs4 import BeautifulSoup
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


from nltk import tokenize

from nltk import sent_tokenize


import matplotlib.pyplot as plt;
import numpy as np


#training
def word_feats(words):
#    print "Words are ",words
    return dict([(word, True) for word in words])


positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice',
                  'great', ':)','love','loved','insightful','clever','first-rate',
                  'charming','comical','charismatic','enjoyable','uproarious','original'
                  ,'tender', 'absorbing','sensitive', 'riveting','intriguing',
                  'powerful','fascinating','pleasant','surprising','dazzling', 'thought provoking'
                  ,'imaginative','legendary', 'unpretentious','perfect','aced','rocked','amazing','perfect','rock']

# positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice',
#                    'great', ':)','love','insightful','first-rate',
#                     'charming','enjoyable','original'
#                    ,'tender','hilarious','sensitive', 'riveting','intriguing'
#                    ,'imaginative','legendary','rocked','amazing','perfect','comical','fascinating'
#                    ,'pleasant']


    
    
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':('
                   ,'second-rate','violent','moronic','third-rate','flawed','juvenile'
                   ,'boring','distasteful','ordinary','disgusting','senseless','static'
                   ,'brutal','confused','disappointing','bloody','silly','tired'
                   ,'predictable','stupid','uninteresting','weak','incredibly tiresome']
    
    
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not'
                  ,'suspenseful','low-budget','dramatic','highly-charged','sentimental',
                  'fantasy','slow','romantic','satirical','fast-moving','oddball','picaresque',
                  'big-budget','wacky','an','a','ok','fine',]

print "Length is \n" 
print len(positive_vocab)
print len(negative_vocab)

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 
#print "Positice features",positive_features
train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set) 

inp= raw_input("Input the Name of the Movie :  ")
for_rotten_inp=inp.replace(" ","_")
for_meta_inp=inp.replace(" ","-")
for_flex_inp=inp.replace(" ","-")
for_common_inp=inp.replace(" ","-")

sentences=[]

#print "this is the input",inp

try:
        
    wiki1='https://www.rottentomatoes.com/m/'+for_rotten_inp+'/reviews/?type=user'

    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = Request(wiki1,headers=hdr)
#    print "this is req",req

    ss=wiki1.split('?')
    page = urlopen(req)
    soup = BeautifulSoup(page,'html.parser')
    
    
    #print "this is soup from rottentomatoes",soup
    p_list=soup.find_all('div',{'class':'user_review'})
    for i in range(0,5):
        a=str(p_list[i].text).strip()
#        print "\n\n a is ",a
        paragraph=a
        
        
        #print "this is tokenize.sent_tokenize(paragraph)",sent_tokenize(paragraph)
        lines_list = nltk.tokenize.sent_tokenize(paragraph)
        #print "\n\n lines_list is ",lines_list
        sentences.extend(lines_list)
        #print "\n\n\ this is the sentences from rotten tomatoes",sentences

except:
    print(" ")
    

try:
    count=0
    wiki2= "http://www.metacritic.com/movie/"+for_meta_inp+"/user-reviews"
    req = Request(wiki2,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page,'html.parser')
    
    
    p_list=soup.find_all('div',{'class':'review_body'})
    l=len(p_list)
    for i in range(0,l):
        count+=1
        a=str(p_list[i].text).strip()
        a=a.replace('Expand',' ')
        paragraph=a
        lines_list = tokenize.sent_tokenize(paragraph)
        sentences.extend(lines_list)
except:
    print(" ")

#print "\n\n\ this is the sentences",sentences


try:
    count=0
    wiki3= "https://www.flixster.com/movie/"+for_flex_inp+"/"
    req = Request(wiki3,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page,'html.parser')

    p_list=soup.find_all('span',{'class':'hidden as-is'})
    l=len(p_list)
    for i in range(0,l):
        count+=1
        a=str(p_list[i].text).strip()
        a=a.replace('Expand',' ')
        paragraph=a
        lines_list = tokenize.sent_tokenize(paragraph)
        sentences.extend(lines_list)
except:
    print(" ")

#print "\n\n\ this is the sentences",sentences


try:
    count=0
    site1= "https://www.commonsensemedia.org/movie-reviews/"+for_common_inp+"/user-reviews/child"
    req = Request(site1,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page,'html.parser')

    p_list=soup.find_all('div',{'class':'views-field views-field-field-user-review-review user-review-text'})
    #print (len(p_list1))

    l=len(p_list)
    for i in range(0,l):
        count+=1
        a=str(p_list[i].text).strip()
        paragraph=a
        lines_list = tokenize.sent_tokenize(paragraph)
        sentences.extend(lines_list)
except:
    print(" ")

print "ths is the sentences",sentences

#CONVERTING INTO TEXT FILE :


text_file = open("Output.txt", "w")
text_file.write("Purchase Amount: %s" % sentences)
text_file.close()
with open("Output.txt", "w") as text_file:
    text_file.write("{}".format(sentences))

with open("Output2.txt", "w") as text_file:
    text_file.write("Purchase Amount: {}".format(sentences))

########################

count_pos=0
count_neg=0
final_val=0
temp = []
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    neg=02
    pos=0
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        classResult = classifier.classify(word_feats(word))
        #print "word is \n",word
        #print "This is the result for classresult\n", classResult
        
        
        if classResult == 'neg':
            neg = neg + 1
            
        if classResult == 'pos':
            pos = pos + 1
            
    pos_review=str(float(pos)/len(words))
    neg_review=str(float(neg)/len(words))
    
    if (float(pos_review)>float(neg_review)):
        final_val+=1
        count_pos+=1
    else:
        final_val-=1
        count_neg+=1

objects = ('Positive Reviews', 'Negative Reviews')
y_pos = np.arange(len(objects))
#print "this is y_pos\n", y_pos
performance = [count_pos,count_neg]


plt.subplot(1, 2, 1)
plt.bar(y_pos, performance, align='center', alpha=0.5,width=0.3)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Review Meter 1')
#plt.show()

plt.subplot(1, 2, 2) 
labels = 'Positive Reviews', 'Negative Reviews'
sizes = [count_pos,count_neg]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Review Meter 2')
plt.show()


if(final_val>0):
    print("\n\n MOVIE IS WORTH WATCHING\n\n")
else:
    print("\n\n MOVIE IS NOT WORTH WATCHING\n\n")
