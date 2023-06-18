import requests
from flask import Flask, render_template, request,redirect,url_for,session
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask_session import Session
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import re

app=Flask(__name__,template_folder='.\Template')


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


cols=['ProductName','ProductCost','hyperlinks','Product']
ClothingData=pd.DataFrame(columns=cols)
credentials=pd.read_csv(r'D:\myproject\scrap\envcode\Data\creds.csv')
def make_clickable(url, name):
    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,name)


@app.route("/",methods=['POST','GET'])
def login():
    user = request.form.get('UserName')
    user = str(user)
    password = request.form.get('password')
    password = str(password)
    output = ''
    if len(credentials[(credentials['User'] == user) & (credentials['Password'] == password)]) == 0:
        return render_template('Login.html', output='Please enter valid credentials!!')
    else:
        session["name"] = request.form.get("UserName")
        return redirect(url_for('func'))


@app.route("/logout",methods=['POST','GET'])
def logout():
    session["name"] = None
    return redirect(url_for('login'))

@app.route("/func1",methods=['POST','GET'])

def func1():
    if not session.get("name"):
        # if not there in the session then redirect to the login page
        return redirect("/")


    input1 = request.form.get('searchfor')
    name=session.get("name")
    input = str(input1)

    url1 = "https://www.flipkart.com/search?q="+input
    content = requests.get(url1)
    HTMLCON = content.content
    soup = BeautifulSoup(HTMLCON, 'html.parser')

    # Hyperlinksm
    list1 = soup.findAll('div', {'class': '_13oc-S'})
    list2 = []
    for i in range(0, len(list1)):
        list2.append(list1[i].find('a').get('href'))
    hyperlinks = []
    for i in list2:
        hyperlinks.append('https://www.flipkart.com'+i)

    # Scraping all hyperlinks

    title = []
    data = []
    for i in hyperlinks:
        content = requests.get(i)
        HTMLCON = content.content
        soup = BeautifulSoup(HTMLCON, 'html.parser')
        title.append(soup)


        imgdiv = []
        imgtag = []
        for j in range(0, len(title)):
            imgdiv.append(title[j].find('div', {'class': '_1AtVbE col-12-12'}))



        data.append({"Name": soup.find('span', {'class': 'B_NuCI'}).text,
                 "Cost": soup.find('div', {'class': '_30jeq3 _16Jk6d'}).text,
                 "Link": 'https://www.flipkart.com'+list1[j].find('a').get('href'),
                 "Imgurl":imgdiv[j].find('img').get('src')}
                    )




    cat = soup.findAll('a', {'class': '_2whKao'})
    cat = cat[1].get_text()
    cat1 = soup.findAll('a', {'class': '_2whKao'})
    cat1 = cat1[2].get_text()
    rec = pd.DataFrame(columns=['Search', 'Time', 'Category1', 'Category2'])
    search = []
    search.append(input)

    prodname = request.form.get('input_value')
    name1 = session.get("name")
#    prodname=re.sub('[^A-Za-z0-9]+', '', prodname)
    prodname = str(prodname)
    print(prodname)
    productname=[]
    name=[name1]
    productname.append(prodname)

    rec['Search'] = search
    rec['Time'] = datetime.now().date()
    rec['Category1'] = cat
    rec['Category2'] = cat1
    rec['User']=name
    rec['Product']=productname


    # rec.set_index=len(rec)+1
    if search[0]!='' or search[0]!=None:

        rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv', mode='a', index=False, header=False)

    else:
        pass
    #



    rec = pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')
    rec['Search'].dropna(axis=0, inplace=True)
    rec['Product'].fillna('',inplace=True)

    rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)

    label=LabelEncoder()
    X = rec.drop(columns=['Group', 'Time'])
    X = X.fillna('')
    X['Search'] = label.fit_transform(X['Search'])
    X['Category1'] = label.fit_transform(X['Category1'])
    X['Category2'] = label.fit_transform(X['Category2'])
    X['User'] = label.fit_transform(X['User'])
    X['Product'] = label.fit_transform(X['Product'])

    scores = []

    # Loop through a range of possible k values
    silhouette = []
    for i in range(1, 11):
        k = KMeans(n_clusters=i)
        k.fit_predict(X)
        silhouette.append(k.inertia_)

    # Find the index of the highest silhouette score
    best_k = silhouette.index(max(silhouette)) + 2

    kmeans = KMeans(n_clusters=best_k)
    kmeans.fit(X)
    class_col = list(kmeans.labels_)

    y = rec['Search'].values
    x = X.drop(columns='Search')
    x = x.values
    # x = x.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    Ypred = model.predict(x_test)
#    accuracy_score(y_test, Ypred)
    prediction = model.predict([[x[-1][0], x[-1][1], x[-1][2], x[-1][3]]])[0]
    print(prediction)



#    y = rec.iloc[:, -1:].values
   #  y=rec['Category2']
#    x = rec.drop(columns=[ 'Time','Category2','Search'])
#    x = label.fit_transform(x)
#    x = x.reshape(-1, 1)
#    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
#    model = RandomForestClassifier()
#    model.fit(x_train, y_train)
#    Ypred = model.predict(x_test)
#    accuracy_score(y_test, Ypred)
#    prediction = model.predict([[x[-1][0]]])[0]
    print(prediction)
    urlfinal = "https://www.flipkart.com/search?q="+prediction
   # rec['Product'] =

    if "Buy" in request.form:



        return render_template('home.html', name=name, tables=data, titles=[''], predict=prediction,urlf=urlfinal)

    else:

         return render_template('home.html', tables=data, titles=[''],predict=prediction,urlf=urlfinal)
#return render_template('home.html', tables=[HTML(ClothingData.to_html(render_links=True))], titles=[''])




'''@app.route("/buy",methods=['POST','GET'])

def buy():
    if not session.get("name"):
        # if not there in the session then redirect to the login page
        return redirect("/")
    quantity = request.form.get('quantity')
    name = session.get("name")
    quantity = int(quantity)
    print(quantity)
    print(quantity)
    return render_template('home.html', quantity=quantity,tables=data, titles=[''],predict=prediction,urlf=urlfinal)



'''











@app.route("/func",methods=['POST','GET'])

def func():
    try:
        if not session.get("name"):
            # if not there in the session then redirect to the login page
            return redirect("/")

        prediction=''
        data=pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')
        if data['Search'].tail(1).isna().values[0]==True:
            data.iloc[-2]['Product'] = data.iloc[-1]['Product']
            data['Search']=data['Search'].fillna(method='ffill')
            data=data.head(len(data))
            data.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)
        else:
            data=data
            data.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv', index=False)


        input1 = request.form.get('searchfor')
        name = session.get("name")
        input = str(input1)

        url1 = "https://www.flipkart.com/search?q=" + input
        content = requests.get(url1)
        HTMLCON = content.content
        soup = BeautifulSoup(HTMLCON, 'html.parser')

        # Hyperlinksm
        list1 = soup.findAll('div', {'class': '_13oc-S'})
        list2 = []
        for i in range(0, len(list1)):
            list2.append(list1[i].find('a').get('href'))
        hyperlinks = []
        for i in list2:
            hyperlinks.append('https://www.flipkart.com' + i)

        # Scraping all hyperlinks

        title = []
        data = []
        for i in hyperlinks:
            content = requests.get(i)
            HTMLCON = content.content
            soup = BeautifulSoup(HTMLCON, 'html.parser')
            title.append(soup)

            imgdiv = []
            imgtag = []
            for j in range(0, len(title)):
                imgdiv.append(title[j].find('div', {'class': '_1AtVbE col-12-12'}))

            data.append({"Name": soup.find('span', {'class': 'B_NuCI'}).text,
                         "Cost": soup.find('div', {'class': '_30jeq3 _16Jk6d'}).text,
                         "Link": 'https://www.flipkart.com' + list1[j].find('a').get('href'),
                         "Imgurl": imgdiv[j].find('img').get('src')}
                        )

        cat = soup.findAll('a', {'class': '_2whKao'})
        cat = cat[1].get_text()
        cat1 = soup.findAll('a', {'class': '_2whKao'})
        cat1 = cat1[2].get_text()
        rec = pd.DataFrame(columns=['Search', 'Time', 'Category1', 'Category2'])
        search = []
        search.append(input)

        prodname = request.form.get('input_value')
        name1 = session.get("name")
        #    prodname=re.sub('[^A-Za-z0-9]+', '', prodname)
        prodname = str(prodname)
        print(prodname)
        productname = []
        name = [name1]
        productname.append(prodname)

        rec['Search'] = search
        rec['Time'] = datetime.now().date()
        rec['Category1'] = cat
        rec['Category2'] = cat1
        rec['User'] = name
        rec['Product'] = productname
        rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv', mode='a', index=False, header=False)
        rec = pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')



        rec = pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')
        #rec.dropna(subset=['Search'],axis=0, inplace=True)
        rec['Product'].fillna('',inplace=True)

        rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)

        label=LabelEncoder()
        X = rec.drop(columns=['Group', 'Time'])
        X = X.fillna('')
        X['Search'] = label.fit_transform(X['Search'])
        X['Category1'] = label.fit_transform(X['Category1'])
        X['Category2'] = label.fit_transform(X['Category2'])
        X['User'] = label.fit_transform(X['User'])
        X['Product'] = label.fit_transform(X['Product'])

        scores = []

        # Loop through a range of possible k values
        silhouette = []
        for i in range(1, 11):
            k = KMeans(n_clusters=i)
            k.fit_predict(X)
            silhouette.append(k.inertia_)

        # Find the index of the highest silhouette score
        best_k = silhouette.index(max(silhouette)) + 2

        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(X)
        class_col = list(kmeans.labels_)

        data1 = pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')
        data1['Group']=class_col
        data1 = data1.dropna(subset=['Search'], axis=0)
        data1.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data1['Search'])
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        relevant_data = data1.loc[
            (data1['User'] == data1['User'].tail(1).values[0]) & (data1['Product'] == data1['Product'].tail(1).values[0]) & (data1['Category1'] == data1['Category1'].tail(1).values[0]) & (
                        data1['Category2'] == data1['Category2'].tail(1).values[0])]

        # Get the search query that was used in each of those rows
        relevant_queries = relevant_data['Search']

        # Compute the average cosine similarity between those queries and all the other queries in the dataset
        query_indices = [data1[data1['Search'] == data1['Search'].tail(1).values[0]].index[0] for query in relevant_queries]
        similarities = cosine_similarities[query_indices].mean(axis=0)

        # Find the index of the most similar search query
        most_similar_index = similarities.argmax()
        prediction=data1.iloc[most_similar_index]['Search']




       # prediction= recommend_search(data['Search'].tail(1).values[0], data['Product'].tail(1).values[0], data['Category1'].tail(1).values[0], data['Category2'].tail(1).values[0])
        print(prediction)
        urlfinal = "https://www.flipkart.com/search?q=" + prediction
        # rec['Product'] =
        rec = pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')

        if rec['Search'].tail(1).isna().values[0]==True:
            rec.iloc[-2]['Product'] = rec.iloc[-1]['Product']
            rec['Search']=rec['Search'].fillna(method='ffill')
            rec=rec.head(len(rec))
            rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)
        else:
            rec=rec
            rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv', index=False)


    except Exception as e:
        print(f"An error occurred : {e}")

    if "Buy" in request.form:






        return render_template('home.html', name=name, tables=data, titles=[''], predict=prediction, urlf=urlfinal)

    else:

        return render_template('home.html', tables=data, titles=[''], predict=prediction, urlf=urlfinal)






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)

    '''    rec=pd.read_csv(r'D:\myproject\scrap\envcode\Data\record1.csv')
        if rec['Product'].tail(1).isna().values[0]==True:
            rec=rec.tail(len(rec)-1)
        else:
            rec.loc[len(rec)-2, 'Product'] = rec.loc[len(rec)-1, 'Product']
           # rec=rec.tail(len(rec-1))
            rec.to_csv(r'D:\myproject\scrap\envcode\Data\record1.csv',index=False)
    '''