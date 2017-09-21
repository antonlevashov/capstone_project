# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
September 13st, 2017


http://www.sciencedirect.com/science/article/pii/S2090447914000550
https://www.cse.iitb.ac.in/~pb/cs626-449-2009/prev-years-other-things-nlp/sentiment-analysis-opinion-mining-pang-lee-omsa-published.pdf
https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/


## Proposal

The purpose of this project is to create a small module for a future trading bot, this bot will take different positions in crypthocurency market. 
My module will performe sentiment analysis on twitter and other news data comming from a list of important news sources and influencial people and give a prediction about direction of the market. I'll also give weiths for each data source/twitter account so we can see how influencial they are.
During the exploration part the module will give us some statistics about tweets, hot topics and trends in market to see what are the things  people are talking the most. The goal is to understand the general "feeling" of the markets and trends to give us some insights.
The second part is about finding correlation betwen tweets (and they sources) at the time T and prices change in T+1 in the given market.

I'll investigate whether measurements of collective mood states derived from Twitter and other sourcess are correlated to the value of given cryptocurency over time. To analyze content (mood) of Twitter feeds I'll try this tools (because I don't have twitts labeled as pos/neg): 
  - OpinionFinder that measures positive /negative mood (http://mpqa.cs.pitt.edu/opinionfinder/)
  - Google-Profile of Mood States (Calm, Alert, Sure, Vital, Kind, and Happy)
  - CoreNLP (https://nlp.stanford.edu/sentiment/)
  - Natural Language Classifier by IBM Watson (https://github.com/watson-developer-cloud/python-sdk/tree/master/examples)
  - and others...
  

After performing sentiment analysis I'll feed the mood of the tweets/news and source as predictors to a neural network and price of coin as a predicted value. And see if NN can capture some hidden and interesting patterns. Ex: if someone says every time you should buy it but the price drop naybe NN can capture it and make the "inverse" prediction. In a few month I'll take Artificial Intelligence NANODEGREE, so I hope I'll use more sophisticated/appropriate NN in next itiration of this project.

Such things were tried before in financial markets but wasn't very conclusive because there is many big players who can move markets, in cryptocurencies (a way smaller capitalisation) it's a bit different - there are many small players and a few big instituions so social medias have more influence than in equity market for example. 
There is a lot of room for exploration: 
  - Test the system on different cryptocurrencies.
  - Test the different combination of coins and tweets.
  - Find if tweets about one coin have an influence on others (at the moment different crypto coins are extremly positivly correlated).


### Domain Background
I have a lot of intereste in fiancial markets, especially in the new cryptocurency market. I think it's an interresting play ground as he markets are growing and they are more "social" because one of foundation of blockchain - no autorities.
I found many papers on line that explains well the impact of social media on financial markets and I would like to see how it performe in cryptocurency market. Here is some good introdcution:

Behavioral economics tells us that emotions can profoundly affect individual behavior and decision-making. Does this also apply to societies at large, i.e. can societies experience mood states that affect their collective decision making? By extension is the public mood correlated or even predictive of economic indicators? Here we investigate whether measurements of collective mood states derived from large-scale Twitter feeds are correlated to the value of the Dow Jones Industrial Average (DJIA) over time.
http://www.sciencedirect.com/science/article/pii/S187775031100007X


Behavioral finance researchers have shown that the stock market can be driven by emotions of market participants. In a number of recent studies mood levels have been extracted from Social Media applications in order to predict stock returns. The paper tries to replicate these findings by measuring the mood states on Twitter. The sample consists of roughly 100 million tweets that were published in Germany between January, 2011 and November, 2013. In a first analysis, a significant relationship between aggregate Twitter mood states and the stock market is not found. However, further analyses also consider mood contagion by integrating the number of Twitter followers into the analysis. The results show that it is necessary to take into account the spread of mood states among Internet users. Based on the results in the training period, a trading strategy for the German stock market is created. The portfolio increases by up to 36 % within a six-month period after the consideration of transaction costs.
https://rd.springer.com/article/10.1007/s12599-015-0390-4

For example those guys could obtain 75.56% accuracy (it's a research paper, in real life it can be less glorious):
"In order to test our results, we propose a new cross validation method for financial data and obtain 75.56% accuracy using Self Organizing Fuzzy Neural Networks (SOFNN) on the Twitter feeds and DJIA values from the period June 2009 to December 2009."
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.375.4517&rep=rep1&type=pdf

SOURCES:
Twitter can help predict stock market performance
https://hub.jhu.edu/2015/04/23/twitter-predicts-ipos/
Can Tweets And Facebook Posts Predict Stock Behavior
http://www.investopedia.com/articles/markets/031814/can-tweets-and-facebook-posts-predict-stock-behavior-and-rt-if-you-think-so.asp
Neural Networks and Bitcoin
https://medium.com/@binsumi/neural-networks-and-bitcoin-d452bfd7757e

### Problem Statement
Our future trading bot (I'm not making a bot but just a part of it) need to make a bet buy, sell or keep and our module will try to help this bot make a guess about direction of the market and act on it. I think by analysing social media (twitter in this example) we can "feel" if market is bullish or bearish.  The final output of the algorithm will be a prediction about direction of the market for the next few hours (I think it will be beetwen 1 hour and 1 day). 

### Datasets and Inputs
I will use Twitter Streaming API to download tweets related to the choosen keywords (ex: "bitcoin", "ethereum", "ripple", etc).
In order to access Twitter Streaming API (https://dev.twitter.com/streaming/overview), I'll get 4 pieces of information from Twitter: API key, API secret, Access token and Access token secret (using Tweepy https://github.com/tweepy/tweepy or something similiar from this list https://dev.twitter.com/resources/twitter-libraries). I will also pre-processing the tweets by removing URLs, special symbols, convert to lower case and using 4 Python libraries: json for parsing the data, pandas for data manipulation,  for creating charts, and RE for regular expressions. 
I was also thinking about adding other data sources from https://www.quandl.com like price of gold, different exchange rate, etc... but for this project I'll use only twitter data. I hope in AI nanodegre I'll extend this projet more by including top 100 Bitcoin Blogs and Websites on Bitcoin Crypto-Currency and Blockchain Technology (http://blog.feedspot.com/bitcoin_blogs/).

SORUCES:
Sentiment Analysis of Twitter Data Using Machine Learning Approaches and Semantic Analysis 
http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6897213
An Introduction to Text Mining using Twitter Streaming API and Python
http://adilmoujahid.com/posts/2014/07/twitter-analytics/
Neural Networks and Bitcoin
https://medium.com/@binsumi/neural-networks-and-bitcoin-d452bfd7757e


### Solution Statement
To find the sentiment of the tweets and other data seources, and they correaltion to the market I want to try as many classification methods as possible. Because different algorithms can deliver different results. Generally it is expected that state of the art classification techniques such as SVM would outperform more simple techniques such as Naïve Bayes, and tools using NN can outperforme all other solutions. But the finall solution must be viable for implimentaion into production enviroment and predict more accurate the direction of market movement on he given day. What will be interesting is to find: how the degree of importance of different data sources (ex: famous crypto twitter that everyone follows) affect ou correlate with the price of coin. Too optimized the implementation, I'll use NLTK. It is a python library for text processing, and can be used to help in implementation of certain algorithms (ex : support vector machine).


### Benchmark Model
-First I will use classification using the Naïve Bayes Classifier to have a minimum accuracy. Feauther I will try to improved accuracy by using the feature selection method. Also the classification accuracy of Maximum Entropy Classifier is better than the Naïve Bayes Classifier but still less than the Naïve Bayes Classifier with Feature Selection based on Entropy Method whose accuracy is further less than the Maximum Entropy Classifier with Feature Selection based on Entropy Method.
After some research I found that maximum Entropy Classifier with Feature Selection based on Entropy Method is best for classification of a given input. So my benchmark model will be NB Classifier (compared against a random predicition). 


### Evaluation Metrics
In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

My evaluation metric will be ...


### Project Design

EXTRACTION OF TWEETS AND NEWS



PREPROCESSING and Word Embeddings -----------------------------------------


I'll use In learning based techniques, before training the classifier, we need to select the words/features that we will use in your model. We  can’t just use all the words that the tokenization algorithm returned (ex: NLTK) because there are several irrelevant words within them. so we need to clean the data from irrelevant information like links, certain stop words, icons, etc...

Also I need to investigate hastags, should I use them or not.

EXPLORATION AND STATS -----------------------------------------


MOOD EXTRACTION ------------------------------------------
For Mood extraction I'll try this existing tools (certain tools are written in Java, I'll see if I can handle it):
  - OpinionFinder that measures positive /negative mood (http://mpqa.cs.pitt.edu/opinionfinder/)
  - Google-Profile of Mood States (Calm, Alert, Sure, Vital, Kind, and Happy)
  - CoreNLP (https://nlp.stanford.edu/sentiment/), How to Build an Email Sentiment Analysis Bot: An NLP Tutorial https://www.toptal.com/java/email-sentiment-analysis-bot
  - Natural Language Classifier by IBM Watson, In the Bluemix Catalog (https://github.com/watson-developer-cloud/python-sdk/tree/master/examples)
  - Microsoft LUIS (https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
  ...
  


SIMPLE CLASSIFIERS FOR SENTIMENT -----------------------------------------

Max Entropy classifier can benefit from the neutral class
Just remember that in case that you use the n-grams framework, the number of n should not be too big. Particularly in Sentiment Analysis you will see that using 2-grams or 3-grams is more than enough and that increasing the number of keyword combinations can hurt the results. Moreover keep in mind that in Sentiment Analysis the number of occurrences of the word in the text does not make much of a difference. Usually Binarized versions (occurrences clipped to 1) of the algorithms perform better than the ones that use multiple occurrences.

For example you might find that Max Entropy with Chi-square as feature selection (https://cmm.cit.nih.gov/maxent/l... ) is the best combination for restaurant reviews, while for twitter the Binarized Naïve Bayes with Mutual Information feature (Machine Learning Tutorial: The Naive Bayes Text Classifier ) selection outperforms even the SVMs. Be prepared to see lots of weird results. Particularly in case of twitter, avoid using lexicon based techniques because users are known to use idioms, jargons and twitter slangs what heavily affect the polarity of the tweet.

https://www.quora.com/What-kind-of-algorithms-do-we-use-for-sentiment-analysis-Is-there-any-list-for-the-algorithms-and-about-their-structure



There is two very interesting Deep Learning inspired tools for text analysis. The first is the Word2Vec (https://en.wikipedia.org/wiki/Word2vec) algorithm and the second tool is GloVe (https://nlp.stanford.edu/projects/glove/). Both of these representation learning algorithms seemed really useful for analyzing text. I'll use Word2Vec because word2vec is a "predictive" model and GloVe is a "count-based" model.(https://www.quora.com/How-is-GloVe-different-from-word2vec)

It takes a large text corpus as input and outputs a numeric vector representation for each word. The vectors are supposed to represent the semantic similarity between the words. I chose to use Word2Vec due to its outstanding Python implementation written by Radim Řehůřek, and because of an excellent tutorial that was written by Angela Chapman during her internship at Kaggle.

Classify and Regress with GraphLab Create
Now that I settled on a large dataset and a deep learning inspired tool, the next step was to glue all the parts together. I wanted to create an end-to-end machine learning solution that takes a blogger’s posts as input and predicts the blogger’s gender and age with high accuracy. To make all the parts work together, I used GraphLab Create in the following manner.

-First, I processed the blog posts using Beautiful Soup, a handy Python package for text scraping. I then loaded the text into an SFrame, a scalable dataframe object in GraphLab Create.
- Noise Removal
Cleaning the data from irrelevant news as well as advertisements/bio(if you have collected data by web crawling)
-Then, I trained a Word2Vec model on the blog posts. It worked! The trained model “knew” that vector representation of “hehe” is similar to the vector representation of “LOL.” (I found this to be mind-blowing.)
-Next, for each blogger, I used Word2Vec to calculate the average vector representation of all words that appeared in the bloggers posts.
-Lastly, I used GraphLab Create’s classification toolkit to construct classifiers that takes the average vector of each blogger as input features and predicts the blogger’s gender and age with high accuracy.

The exciting part, I believe, is that the results obtained in this way are better than the known state-of-the-art algorithm for this problem. But I still need to perform additional tests to verify this conclusion.

If you want to try Deep Learning on text, take a look at my IPython notebook that describes in detail how to create a text classifier using Word2Vec and GraphLab Create. You can use it to create your own blogger gender and age classifier, or construct your own deep text classifiers for other NLP tasks.

-------------------

-------------------
Deep Learning for NLP Best Practices
http://ruder.io/deep-learning-nlp-best-practices/index.html
-------------------
Implementing a CNN for Text Classification in TensorFlow
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
-------------------
https://www.quora.com/What-kind-of-algorithms-do-we-use-for-sentiment-analysis-Is-there-any-list-for-the-algorithms-and-about-their-structure
2 Noise Removal
Cleaning the data from irrelevant news as well as advertisements/bio(if you have collected data by web crawling)
3 Feature Selection
In learning based techniques, before training the classifier, you must select the words/features that you will use on your model. You can’t just use all the words  because there are several irrelevant words within them.
The features can be unigrams and/or bigrams or higher ngrams with/without punctuation and with/without stopwords.
Some of the current features used are :
Terms presence and frequency: These features are individual words or word n-grams and their frequency counts. It either gives the words binary weighting (zero if the word appears, or one if otherwise) or uses term frequency weights to indicate the relative importance of features.
Parts of speech (POS): Finding adjectives, as they are important indicators of opinions.
Opinion words and phrases: These are words commonly used to express opinions including good or bad, like or hate. On the other hand, some phrases express opinions without using opinion words. For example: cost me an arm and a leg.
Negations: The appearance of negative words may change the opinion orientation like not good is equivalent to bad.
4. Classification Algorithm - The text classification methods using ML approach can be roughly divided into supervised and unsupervised learning methods. The supervised methods make use of a large number of labeled training documents. The unsupervised methods are used when it is difficult to find these labeled training documents.

Some of the classifiers along with the libraries (as building classifiers from scratch is painstakingly difficult :P ) that you can try are these:

Naive bayes - BernoulliNB, GaussianNB, MultinomialNB. Naive Bayes 
Support Vector Classifiers - LinearSVC, PolynomialSVC, RbfSVC, NuSVC. Support Vector Machines
Maximum Entropy Model - GIS, IIS, MEGAM, TADM nltk.classify package

NOTE: Different Classifiers deliver different results
Make sure you try as many classification methods as possible. Have in mind that different algorithms deliver different results. Also note that some classifiers might work better with specific feature selection configuration.
Generally it is expected that state of the art classification techniques such as SVM would outperform more simple techniques such as Naïve Bayes. Nevertheless be prepared to see the opposite. Sometimes Naïve Bayes is able to provide the same or even better results than more advanced methods. Don’t eliminate a classification model only due to its reputation.

I'll point you to a few good resources(both for extracting features/learning techniques):-

1. Google Word2Vec(https://code.google.com/p/word2vec/) : Provides methods to convert text to features automatically. Here's a link to a tutorial notebook(Bag of Words Meets Bags of Popcorn)

2. Deep Learning(Deeply Moving: Deep Learning for Sentiment Analysis): This is currently the model used by Stanford. https://nlp.stanford.edu/sentiment/

Some of the papers on sentiment analysis may help you -
One of the earlier works by Bo Pang, Lillian Lee http://acl.ldc.upenn.edu/acl2002...
A comprehensive survey of sentiment analysis techniques http://www.cse.iitb.ac.in/~pb/cs...
Study by Hang Cui, V Mittal, M Datar using 6-grams http://citeseerx.ist.psu.edu/vie...
-------------------
https://www.quora.com/What-are-the-best-supervised-learning-algorithms-for-sentiment-analysis-in-text-I-want-to-do-text-analysis-to-determine-whether-a-news-item-is-positive-or-negative-for-a-given-subject
NLP Tools for News Analytics

Sentiment Analysis will require the following pre-processing:

1. Noise Removal - Cleaning the data from irrelevant news as well as advertisements/bio(if you have collected data by web crawling)

2. Classification - Categorizing the news data to different domains - "Markets",   "Economy", "Industry", "Technology" and so on. It is as necessary as the algorithm because you will have different set of features for different domains and thus, each domain should have different classifier. For example, A positive news in Technology sector for Microsoft may be a negative news for Apple stocks.

3. Named Entity Recognition - This is the most important part of sentiment analysis as the objective of sentiment analysis is (In words of Bing Liu):

"Given an opinion document, discover all the opinion quintuples - entity,aspect,sentiment on aspect of the entity, opinion holder and the time/context of opinion."

For example, Sentiment analysis on political news to predict elections will obviously have to extract political entities - Narendra Modi/Rahul Gandhi and the aspects of their campaign - secularism/minority upliftment from the news and then, tag them as positive or negative.

4. Subjectivity Classification - Classifying sentences as subjective or objective since subjective sentences hold sentiments while objective sentences are facts and figures.

5. Feature Selection - The features can be unigrams and/or bigrams or higher ngrams with/without punctuation and with/without stopwords with presence(boolean)/count(int)/tfidf(float) as accompanying feature scorer for each sentence/paragraph/file. Filtering Stopwords reduces accuracy. Adverbs and determiners that start with "wh" can be valuable features, and removing them as English Stopwords causes dip in accuracy. Similarly, punctuation helps in detecting sarcasm and exclamation.
I used unigrams and bigrams without removing stopwords but removing proper nouns as my features. Presence, Count or TfIdf score were used depending on classifier as Presence gave better results with Naive Bayes while Tf Idf gave better results with Linear SVM. I then, did an association measure test based on chi square/poisson stirling ratio/likelihood ratio to find the most informative features and used them to train the model.  

I used unigrams and bigrams without removing stopwords but removing proper nouns as my features. Presence, Count or TfIdf score were used depending on classifier as Presence gave better results with Naive Bayes while Tf Idf gave better results with Linear SVM. I then, did an association measure test based on chi square/poisson stirling ratio/likelihood ratio to find the most informative features and used them to train the model.  

6. Sentiment Extraction - It can be done using unsupervised learning, supervised learning, sentiment lexicon based approach or a mix of these. Coming to your question, there is no such thing as best algorithm. I have tried and tested the following algorithms :
 
ExtraTreesClassifier
GradientBoostingClassifier
RandomForestClassifier
LogisticRegression
BernoulliNB
GaussianNB
MultinomialNB
KNeighborsClassifier
LinearSVC
NuSVC
SVC
DecisionTreeClassifier
Naive Bayes
http://blog.datumbox.com/machine-learning-tutorial-the-naive-bayes-text-classifier/
Maximum Entropy Model

Sentiment analysis algorithms and applications: A survey
http://www.sciencedirect.com/science/article/pii/S2090447914000550

For Naive Bayes, I had used unigrams (68% accuracy) and bigrams (79% accuracy) as features. I don't exactly remember the PRF value for each classifier on my data set but I got maximum accuracy for Bernoulli Naive Bayes, Maximum Entropy Model (IIS), Linear SVM using five fold validation. I would suggest you not to go for Tree Classifiers as they are not space optimal for training data creating over-complex trees leading to over-fitting and also, learning an optimal decision tree is an NP complete problem. I will thus suggest you to try these :

Naive bayes - BernoulliNB, GaussianNB, MultinomialNB
Support Vector Classifiers - LinearSVC, PolynomialSVC, RbfSVC, NuSVC
Maximum Entropy Model - GIS, IIS, MEGAM, TADM
I am currently using a voted model between Naive Bayes and Lexicon based approach.
-------------------
https://www.quora.com/What-are-the-best-algorithms-for-sentiment-analysis
What are the best algorithms for sentiment analysis?
In a broad sense, you can say that the best algorithms as of now, dpending on the size/type of dataset that you have will be one the the three : RNN derivatives , DeepForests or NBSVM.

If you have a small dataset and its very far from day-to-day English (hence you cannot use pretrained word2vecs etc), NBSVM is a simple and effective algorithm. http://www.aclweb.org/anthology/...

DeepForests Towards An Alternative to Deep Neural Networks claim to be better than even Deep Neural Networks just using tfidf vectors. If combined with Embeddings, these models can work really well.

RNN derivatives (like LSTM or GRUs) are currently the best models which get high accuracies as well as generalize well.

At ParallelDots for earliest sentiment models were Convnets on text (implemented in pure theano in 2014 looking somewhat like Yoon Kim’s paper, the code for which you can now find in keras demos), which we then shifted to LSTMs (which you can see currently on our website) and now we are combining some new LSTM techniques (self attention and Multi Task learning (MTL)) to make them even better if you check them out in a next few weeks of time. Most current NLP benchmarks are slowly being beaten by using one of a combination of these techniques. (attention for sequence prediction, self attention or MTL). For more details you can check out Salesforce Research https://einstein.ai/research .

Google’s [1706.03762] https://arxiv.org/abs/1706.03762 Attention Is All You Need is a very exciting paper, which might replace all RNN derivatives once in for all and looks like a very strong contender for the future.

Interestingly I am working to compile a detailed blog post on this for sometime. I will post the post here when it is completed.

-----------
SOURCES:
Sentiment Analysis on Movie Reviews
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
Personality Prediction Based on Twitter Stream
https://www.kaggle.com/c/twitter-personality-prediction#description
Tutorial on how to use Google's Word2Vec
https://www.kaggle.com/c/word2vec-nlp-tutorial
ANALYZING CRYPTOCURRENCY MARKETS USING PYTHON
https://blog.patricktriest.com/analyzing-cryptocurrencies-python/

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
