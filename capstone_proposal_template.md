# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
September 13st, 2017


http://www.sciencedirect.com/science/article/pii/S2090447914000550
https://www.cse.iitb.ac.in/~pb/cs626-449-2009/prev-years-other-things-nlp/sentiment-analysis-opinion-mining-pang-lee-omsa-published.pdf
https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

## Proposal
- Trading bot based on sentiment analysis of different medias in crypthocurency market.
- Or find correlation betwen tweets and trends in markets.
- Test on different cryptocurrencies.
- Try to find correlation betwen tweets on the journey T and prices change in T+1
- Find the hot topics and they trends

- First stage:
  - Analysis of smal corpus for day trading 
  - Only twitter and comment section in facebook/forums
  - Use Word2Vec and other ML algo to compare the metrics and choose the most suitable.

- Second stage:  
  - Make a execution module in trading bot.
  - Find wich source has more influence in market. 
  
- Third stage:
  - Add more sources of input (blogs, news portals, etc...)
  
  
Max Entropy classifier can benefit from the neutral class
Just remember that in case that you use the n-grams framework, the number of n should not be too big. Particularly in Sentiment Analysis you will see that using 2-grams or 3-grams is more than enough and that increasing the number of keyword combinations can hurt the results. Moreover keep in mind that in Sentiment Analysis the number of occurrences of the word in the text does not make much of a difference. Usually Binarized versions (occurrences clipped to 1) of the algorithms perform better than the ones that use multiple occurrences.
In learning based techniques, before training the classifier, you must select the words/features that you will use on your model. You can’t just use all the words that the tokenization algorithm returned simply because there are several irrelevant words within them.

Two commonly used feature selection algorithms in Text Classification are the Mutual Information and the Chi-square test. Each algorithm evaluates the keywords in a different way and thus leads to different selections. Also each algorithm requires different configuration such as the level of statistical significance, the number of selected features etc. Again you must use Trial and error to find the configuration that works better in your project.


What kind of algorithms do we use for sentiment analysis? 
Is there any list for the algorithms and about their structure?
There are basically 2 broad types of algorithms for sentiment analysis: lexicon based and learning based techniques
- Particularly in case of twitter, avoid using lexicon based techniques because users are known to use idioms, jargons and twitter slangs what heavily affect the polarity of the tweet.
https://www.quora.com/What-kind-of-algorithms-do-we-use-for-sentiment-analysis-Is-there-any-list-for-the-algorithms-and-about-their-structure

In word2vec, this is cast as a feed-forward neural network and optimized as such using SGD
-  for short texts Naive Bayes may perform better than Support Vector Machines

I used unigrams and bigrams without removing stopwords but removing proper nouns as my features. Presence, Count or TfIdf score were used depending on classifier as Presence gave better results with Naive Bayes while Tf Idf gave better results with Linear SVM. I then, did an association measure test based on chi square/poisson stirling ratio/likelihood ratio to find the most informative features and used them to train the model.  

_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

Crypto currency markets.
Twitter.

Fundamental analysis




Neural Networks and Bitcoin
https://medium.com/@binsumi/neural-networks-and-bitcoin-d452bfd7757e

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

-------------------
https://medium.com/@binsumi/neural-networks-and-bitcoin-d452bfd7757e
When you want to work with Neural Networks the first thing you need is data. Lots of data. That’s why data is also called the new oil. Neural Networks are only effective if you have massive amounts of data. If you are looking for data I can recommend https://www.quandl.com. They have assembled a great collection of databases to get you started.

We started of with a simple LSTM network with 4 input nodes and 100 hidden nodes and started of with the usual input variables such as:
Difficulty
Volume
Price of Gold
Exchange rate of USD/CNY

Even with those variables alone the Neural Network had already a decent fit. We cycled through more than 500 input variables and finally settled on 20. We will keep these 20 a secret for now.
-------------------
Top 100 Bitcoin Blogs and Websites on Bitcoin Crypto-Currency and Blockchain Technology
http://blog.feedspot.com/bitcoin_blogs/
-------------------



### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

Different Classifiers deliver different results
Make sure you try as many classification methods as possible. Have in mind that different algorithms deliver different results. Also note that some classifiers might work better with specific feature selection configuration.
Generally it is expected that state of the art classification techniques such as SVM would outperform more simple techniques such as Naïve Bayes. Nevertheless be prepared to see the opposite. Sometimes Naïve Bayes is able to provide the same or even better results than more advanced methods. Don’t eliminate a classification model only due to its reputation.

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

-Classification using the Naïve Bayes Classifier has minimum accuracy. Its accuracy is improved by using the feature selection method.
-The classification accuracy of Maximum Entropy Classifier is better than the Naïve Bayes Classifier but still less than the Naïve Bayes Classifier with Feature Selection based on Entropy Method whose accuracy is further less than the Maximum Entropy Classifier with Feature Selection based on Entropy Method.
-So it is found out that maximum Entropy Classifier with Feature Selection based on Entropy Method is best for classification of a given input.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-------------------
https://github.com/turi-code/tutorials/blob/master/notebooks/deep_text_learning.ipynb

I needed a proper Deep Learning algorithm that can analyze text. In my readings, I came across two very interesting Deep Learning inspired tools for text analysis. The first tool was the Word2Vec algorithm invented by Thomas Mikolov et al. from Google, and the second tool was GloVe invented by Jeffrey Pennington et al. from Stanford University. Both of these representation learning algorithms seemed really useful for analyzing text. In the end, 
I chose to use Word2Vec because word2vec is a "predictive" model, whereas GloVe is a "count-based" model.(More on the differences https://www.quora.com/How-is-GloVe-different-from-word2vec)

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

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
