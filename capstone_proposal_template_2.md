# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
27/09/2017

## Proposal
I'll use the following chalenge for the capstone project : Cdiscount’s Image Classification Challenge on Kaggle (https://www.kaggle.com/c/cdiscount-image-classification-challenge). I hope this project will help me understand better all details about different CNNs and learn from the wonderful community on Kaggle, how to use and tune the neural networks. What works and what dosen't.

Here is the brief description:
In this challenge you will be building a model that automatically classifies the products based on their images. As a quick tour of Cdiscount.com's website can confirm, one product can have one or several images. The data set Cdiscount.com is making available is unique and characterized by superlative numbers in several ways:

- Almost 9 million products: half of the current catalogue
- More than 15 million images at 180x180 resolution
- More than 5000 categories: yes this is quite an extreme multi-class classification!

In this project I'll use Convolutional Neural Networks to classifie images uploaded to Cdiscount website. CNNs are the best available method for image classification/treatement in both performance and "simplicity" (we can use them in production enviroment). Training neural networks needs a large number of training examples (in this competition we given 7000000 iamges for test and 1700000 training sets), I'll also try to use Data Augmentation to increase the  size of our training/validation/testing sets by taking each training image and creating multiple random transformations around the bounding box of the object using ImageDataGenerator from Keras. 

I'll also try image segmentation to generate region proposals where the object might reside, which we can classify separately before
assigning to each image the label of the region whose predicted classification has the highest confidence. Both of these methods result in dramatic improvements in classification accuracy.

To reduce traning time I'll use Transfer Learning, as it's an important concept for Deep Learning. In transfer Learning we retrain the already trained models by giving the input as own dataset. We can reduce the size of data required for training and also the training time. The training utilizes the weights from the already trained model (bottleneck features) and starts learning new weights in the last fully conected layers. I'll try and comapre a few models:  Xception, ResNet-50, InceptionV4 and other model I come acrosse.

I don't have a powerful machine so I'll continue to use EC2 from Amazon for this project.

Sources:
https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba
http://cs231n.stanford.edu/reports/2015/pdfs/jiahan_final_report.pdf
https://www.kaggle.com/c/cdiscount-image-classification-challenge/

### Domain Background
Cdiscount.com generated nearly 3 billion euros last year, making it France’s largest non-food e-commerce company. While the company already sells everything from TVs to trampolines, the list of products is still rapidly growing. By the end of this year, Cdiscount.com will have over 30 million products up for sale. This is up from 10 million products only 2 years ago. Ensuring that so many products are well classified is a challenging task.

Currently, Cdiscount.com applies machine learning algorithms to the text description of the products in order to automatically predict their category. As these methods now seem close to their maximum potential, Cdiscount.com believes that the next quantitative improvement will be driven by the application of data science techniques to images.


https://www.brandwatch.com/blog/top-5-image-recognition-tools/

### Problem Statement

The goal of this competition is to predict the category of a product based on its image(s)m it's a challenging data set of over 15 million images representing 9 million products in over 5,000 categories.. A product can have one or several images associated. For every product _id in the test set, you should predict the correct category_id. 




### Datasets and Inputs

This competion uses the real-world data sets, from current catalogue of products on Cdiscount. Files are in BSON format. 
BSON, short for Binary JSON, is a binary-encoded serialization of JSON-like docments, used with MongoDB. 
There is more than 15 million images in total at 180x180 resolution.

Apparently the number of training examples (images) per class (category) varies a lot in the training set, from over 80000 in the classes with most training examples to just 12 in the class with the least. So we can assume that the testing set has a different distribution and a class with just a few images in the training set can have thousands of images in the test set.

File Descriptions:

- train.bson - (Size: 58.2 GB) Contains a list of 7,069,896 dictionaries, one per product. Each dictionary contains a product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'...binary string...'}. The binary string corresponds to a binary representation of the image in JPEG format. This kernel provides an example of how to process the data.

- test.bson - (Size: 14.5 GB) Contains a list of 1,768,182 products in the same format as train.bson, except there is no category_id included. The objective of the competition is to predict the correct category_id from the picture(s) of each product id (_id). The category_ids that are present in Private Test split are also all present in the Public Test split.

- category_names.csv - Shows the hierarchy of product classification. Each category_id has a corresponding level1, level2, and level3 name, in French. The category_id corresponds to the category tree down to its lowest level. This hierarchical data may be useful, but it is not necessary for building models and making predictions. All the absolutely necessary information is found in train.bson.



https://www.kaggle.com/c/cdiscount-image-classification-challenge/data
Fast Image Data Annotation Tool (FIAT)
https://github.com/christopher5106/FastAnnotationTool

### Solution Statement
I'll approach the problem as both a classification problem (classifying images that contain one or more instances
of a single logo) as well as a detection problem (determining whether or not an image contains a logo). In this project I'll use  Convolutional Neural Networks, a model that was pretrained on the ImageNet Large Scale Visual Recognition Challenge. Training neural networks needs a large number of training examples, and our dataset contains rather few training examples, so I'll use Data Augmentation to increase the  size of our training/validation/testing sets by taking each training image and creating multiple random transformations around the bounding box of the logo using ImageDataGenerator from Keras.
I'll also study the use of a sliding window method and an image-segmentation based approach for generating region proposals where the logo might be. To "synthesize" some aditional images I'll paste logos on images from other sources. In the end the solution should classify images as logo/no logo present and if logo is present give an istimation wich logo amog 47 is present in the given image.

Source:
http://matthewearl.github.io/2016/05/06/cnn-anpr/


### Benchmark Model
Model currently used by Cdiscount is KNN over tf-idf on the title and description of the items (with a lemmatisation to prepare the data).
In production, they set up a threshold value for the level of confidence (i.e. the probability) of the predictions, in order to guarantee an error rate lower that some imposed value. They are able to classify two thirds of the products with the imposed level of accuracy (classify products only on categories where they can reach more than 90% of accuracy).

Cdiscount have been working on classifying products based on their images - best results so far obtained using Inception v3.
In the competition the participants already tired different (single) models with the following benchamrks:

Resnet 101, no bagging, no TTA: 
 - val_loss = 1.18
 - val_acc = 0.744
 - LB = 0.7437
 
SE-ResNet-50, SGD with Nesterov momentum (0.9) and batch_size=256:
 - finetuning about 7 epochs (about 16 hours per epoch on 2x1080)
 - train augmentation: 161x161 random crops + horizontal flips
 - val single crop accuracy: 70.96
 - val 10 crops accuracy: 71.79
 - LB 10 ceps accuracy: 71.67
 
 
 
 
 



### Evaluation Metrics
Setting aside the images whiout logo, I'll first treat the problem as a classification problem with 47 categories for each image.
Then each example in the test set can be assigned exactly one label, and I'll measure the accuracy of our classifiers in predicting the label of each test image. I'll also try to include the images whiout logo in the test procedure and treat this problem as a detection one - configure the models to label as ’whiout logo’ those images whose predictions are made below a certain level of certainty, and then see how the classification rate and false positive rate change as L vary this confidence threshold parameter.

This competition is evaluated on the categorization accuracy of the predictions of 5,000 categories (the percentage of products we get correctly). Currently the top scores are around 70%.


### Project Design
_(approx. 1 page)_




Data augmentation part:

The model that got stuck was high capacity but colour/saturation augmentation was enabled. The one that's still chugging along and doing much better only had random crop/scale and rotation. Tthe prevalence of almost constant black or white backgrounds in most images means that adding variability to that requires much more learning than leaving it as it is.
Next angle is to see if disabling the random crop and just leaving h-flip and a bit of rotation on does better... basically assuming that the preprocessing of the competition dataset leaves things relatively well centred and similarly scaled across train and test datasets. 


Sources:
https://arxiv.org/abs/1610.02357
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://arxiv.org/pdf/1605.07678.pdf
https://distill.pub/2016/augmented-rnns/
http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
