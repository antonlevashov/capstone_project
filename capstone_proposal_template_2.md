# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
27/09/2017

## Proposal

- Logo recognition on Instagram photos.

- Sentiment analisys of the comment section in Instagram.
- use of convolutional neural networks
- data augmentation and create new dataset


### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
I'll use FlickrLogos-47 logo dataset (it is publicly available).
The FlickrLogos dataset consists of real-world images collected from Flickr depicting company logos in various circumstances. The dataset comes in two versions: The original FlickrLogos-32 dataset and the FlickrLogos-47 dataset. FlickrLogos-32 was designed for logo retrieval and multi-class logo detection and object recognition. However, the annotations for object detection were often incomplete,since only the most prominent logo instances were labelled. FlickrLogos-47 uses the same image corpus as FlickrLogos-32 but has been re-annotated specifically for the task of object detection and recognition. New classes were introduced (i.e. the company logo and text are now treated as separate classes where applicable) and missing object instances have been annotated.
(Source: http://www.multimedia-computing.de/flickrlogos/)

FlickrLogos-47 is a very small dataset for Deep Learning problem but using Data Augmentation techniques it can be transformed to a bigger dataset suitable for training.

I'll cropped every logo (or take them from official web site) and performed some transformations like horizontal flip, vertical flip, adding noise, rotation, blurring etc. Then pasting each of transformed images to images with no logos at some random location on it, as well as recording location co-ordinates that will work as annotations for training later. After this, I'll have around 100,000 examples wich I'll divied in training, validation and test sets. For the test set I'll use half of the FlickrLogos-47 (or maybe even all images) to see how CNN trained with "artificial" images perform on real data.

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

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
