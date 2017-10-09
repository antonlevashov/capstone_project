# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
27/09/2017

## Proposal
In this project I'll use Convolutional Neural Networks to find brand logos in photographs uploaded to
different social media websites, I'll start with Instagram. CNNs are the best available method for image classification/treatement in both performance and "simplicity" (we can use them in production enviroment). Training neural networks needs a large
number of training examples, and our dataset contains rather few training examples, so I'll use Data Augmentation to increase the  size of our training/validation/testing sets by taking each training image and creating multiple random transformations around the bounding box of the logo using ImageDataGenerator from Keras. I'll also "synthesize" some aditional images by clipping logos on images from other sources. 

I'll also use image segmentation to generate region proposals where the logo might reside, which we can classify separately before
assigning to each image the label of the region whose predicted classification has the highest confidence. Both of these methods result in dramatic improvements in classification accuracy.

Also to reduce traning time I'll use Transfer Learning, as it's an important concept for Deep Learning. In transfer Learning we retrain the already trained models by giving the input as own dataset. We can reduce the size of data required for training and also the training time. The training utilizes the weights from the already trained model (bottleneck features) and starts learning new weights in the last fully conected layers. I'll try and comapre a few models:  Xception, ResNet-50, InceptionV4 and other model I come a crosse.



Sources:
https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba
http://cs231n.stanford.edu/reports/2015/pdfs/jiahan_final_report.pdf
http://matthewearl.github.io/2016/05/06/cnn-anpr/


### Domain Background
I think today there is a lot of potential in extracting information from images in social medias. If we take the example of Instagram we can only extract a few peaces of info from bio, short description, hashtags and few comments everything else is in the photos that people put on Instagram. I was thinking about it a few years qgo but in order to succed you needed a lot of resources and results wasnt so fantastic. I think today is different we have working CNNs, datasets, cheap infrastructure (ex: EC2). There is already some players who tries to extract information from images and use it to better target advertisement campains. 
https://www.brandwatch.com/blog/top-5-image-recognition-tools/

### Problem Statement
Almost all brands today need image detection to work out where their customers, prospects, critics, and fans. To better target they campaign (know the age, sex, activites and location of peopel on Instagram), to mesure the effectivenes, to "sample" the mood, to follow the live of the products and etc...   Without extracting these information from images, a brand is simply "blind"to all opportunities directed at them each day. Brand can't relly on the classical markenting as almost everyone spend the most of his free time on social medias. And social medias today are more about images and videos than about text. So we need tools to exctract the information from images. The good start will be a logo recognition system, combined with some NLP and available (very few) information available on Instagram, this system can deliver basic statistics like Sentiment Analysis about brend, presents of given brend (and it competitors), people who use it, the context of images, etc... Once we have these classified images we can undertake differents analysis, better understand brend customers and also follow/improve the marketing strategy.

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
