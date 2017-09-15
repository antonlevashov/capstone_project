# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
September 13st, 2017

## Proposal
- Trading bot based on sentiment analysis of different medias in crypthocurency market. 
- Learn media influence on the crypto market.
- Try to find correlation betwen tweets on the journey T and prices change in T+1

- First stage:
  - Analysis of smal corpus for day trading 
  - Only twitter and comment section in facebook/forums
  - Use Word2Vec and other ML algo to compare the metrics and choose the most suitable.

- Second stage:  
  - Make a execution module in trading bot.
  - Find wich source has more influence in market. 
  
- Third stage:
  - Add more sources of input (blogs, news portals, etc...)



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

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-------------------

I needed a proper Deep Learning algorithm that can analyze text. In my readings, I came across two very interesting Deep Learning inspired tools for text analysis. The first tool was the Word2Vec algorithm invented by Thomas Mikolov et al. from Google, and the second tool was GloVe invented by Jeffrey Pennington et al. from Stanford University. Both of these representation learning algorithms seemed really useful for analyzing text. In the end, I chose to use Word2Vec. It takes a large text corpus as input and outputs a numeric vector representation for each word. The vectors are supposed to represent the semantic similarity between the words. I chose to use Word2Vec due to its outstanding Python implementation written by Radim Řehůřek, and because of an excellent tutorial that was written by Angela Chapman during her internship at Kaggle.

Classify and Regress with GraphLab Create
Now that I settled on a large dataset and a deep learning inspired tool, the next step was to glue all the parts together. I wanted to create an end-to-end machine learning solution that takes a blogger’s posts as input and predicts the blogger’s gender and age with high accuracy. To make all the parts work together, I used GraphLab Create in the following manner.

-First, I processed the blog posts using Beautiful Soup, a handy Python package for text scraping. I then loaded the text into an SFrame, a scalable dataframe object in GraphLab Create.

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

-------------------

-------------------

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
