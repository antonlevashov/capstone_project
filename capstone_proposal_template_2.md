# Machine Learning Engineer Nanodegree
## Capstone Proposal
Anton Levashov 
27/09/2017


## Proposal
I'll use the following challenge for the capstone project: [Cdiscount’s Image Classification Challenge on Kaggle](https://www.kaggle.com/c/cdiscount-image-classification-challenge). I hope this project will help me understand better all details about different CNNs and learn from the wonderful community on Kaggle, how to use and tune the neural networks. In this challenge data set is huge, to train a model I'll need to precisely tune it because it will take many hours just to run one epoch on 30-40GB GPU cluster, one epoch is about 10 to 11 hrs on one 1080TI.

Here is the brief description:
In this challenge you will be building a model that automatically classifies the products based on their images. As a quick tour of Cdiscount.com's website can confirm, one product can have one or several images. The data set Cdiscount.com is making available is unique and characterized by superlative numbers in several ways:

- Almost 9 million products: half of the current catalogue
- More than 15 million images at 180x180 resolution
- More than 5000 categories: yes this is quite an extreme multi-class classification!

In this project I'll use Convolutional Neural Networks to classify images uploaded to Cdiscount website. CNNs are the best available method for image classification/treatment in both performance and "simplicity"  of use. Training neural networks needs a large number of training examples (in this competition we given 7000000 iamges for test and 1700000 for training sets), I'll also try to use Data Augmentation to increase the  size of our training/validation sets by taking each training image and creating multiple random transformations around the bounding box of the object using ImageDataGenerator from Keras.

To cut training time I'll use Transfer Learning, as it's an important concept for Deep Learning. In Transfer Learning we retrain the already trained models by giving the input as own dataset. We can cut the size of data required for training and the training time. The training uses the weights from the already trained model and starts learning new weights in the last fully connected layers. I'll try to comapre a few models:  Xception, ResNet-50, InceptionV4 and other model I come across.

I don't have a powerful machine so I'll continue to use EC2 from Amazon for this project.

Sources:

[Neural Network Architectures](https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba)

https://www.kaggle.com/c/cdiscount-image-classification-challenge/


## Domain Background
Cdiscount.com generated nearly 3 billion euros last year, making it France’s largest non-food e-commerce company. While the company already sells everything from TVs to trampolines, the list of products is still rapidly growing. By the end of this year, Cdiscount.com will have over 30 million products up for sale. This is up from 10 million products only 2 years ago. Ensuring that so many products are well classified is a challenging task.

Currently, Cdiscount.com applies machine learning algorithms to the text description of the products in order to automatically predict their category. As these methods now seem close to their maximum potential, Cdiscount.com believes that the next quantitative improvement will be driven by the application of data science techniques to images.


### Problem Statement
The goal of this competition is to predict the class of a product based on its image(s). It's a challenging data set of over 15 million images representing 9 million products in over 5,000 categories.. A product can have one or several images associated. For every product _id in the test set, you should predict the correct category_id. The distribution of product group is highly skewed - there is a lot of products in one class and very few in another.


## Datasets and Inputs
This competition uses the real-world data sets, from current catalogue of products on Cdiscount. Files are in BSON format. BSON, short for Binary JSON, is a binary-encoded serialization of JSON-like documents, used with MongoDB. There is more than 15 million images in total at 180x180 resolution.

Apparently the number of training examples (images) per class (category) varies a lot in the training set, from over 80000 in the classes with most training examples to just 12 in the class with the least. So we can assume that the testing set has a different distribution and a class with just a few images in the training set can have thousands of images in the test set - imbalance problem is present. It looks like distribution of product category is highly skewed along product id. Most frequent product in all products is just 19th frequent among the first 100000 products.

Also someone noticed that product id values have strong relation to category_id. There's [the histogram of top 10 frequent items by product id (X-axis label 0.5 means product id 5,000,000, 2.0 is 20,000,000, etc)](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41650)


[File Descriptions](https://www.kaggle.com/c/cdiscount-image-classification-challenge/data):

- train.bson - (Size: 58.2 GB) Contains a list of 7,069,896 dictionaries, one per product. Each dictionary contains a product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'...binary string...'}. The binary string corresponds to a binary representation of the image in JPEG format. This kernel provides an example of how to process the data.

- test.bson - (Size: 14.5 GB) Contains a list of 1,768,182 products in the same format as train.bson, except there is no category_id included. The objective of the competition is to predict the correct category_id from the picture(s) of each product id (_id). The category_ids that are present in Private Test split are also all present in the Public Test split.

- category_names.csv - Shows the hierarchy of product classification. Each category_id has a corresponding level1, level2, and level3 name, in French. The category_id corresponds to the category tree down to its lowest level. This hierarchical data may be useful, but it is not necessary for building models and making predictions. All the absolutely necessary information is found in train.bson.


## Solution Statement
I'll approach this task as a classification problem. In this project I'll use Convolutional Neural Networks, a model that was pretrained on the ImageNet Large Scale Visual Recognition Challenge. Training neural networks needs a large number of training examples, and our dataset has a lot of training examples, but for some products there is only few examples so I'll use Data Augmentation to increase the  size of our training/validation sets by taking each training image and creating multiple random transformations around the bounding box of the object using ImageDataGenerator from Keras. I'll use this technic only on the categories with a few images.

The original images are 180x180 so training on random crops of 160x160 is a quick way to do data augmentation (ImageNet models are often trained on crops of 224x224). I will start with 1-2 epochs. 

I'll use Xception, Inception V3, SE-ResNet-50 with a 5270-neuron Softamx classification layer on Keras (TensorFlow), with the batch size 256 at first, training/validation split is 80%-20%, input image size 160x160 (downsampling). It will takes around 4-6 hours to do a single epoch with 8 workers in model.fit_generator() on EC2 p2.xlarge. This uses the BSON generator posted in competition kernels.

For a moment I'll stick to the single model but it's worth investigate the use of an ensemble of models.

[Distilling the Knowledge in a Neural Network.](https://www.cs.toronto.edu/~hinton/absps/distillation.pdf) "... We introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel."
 

## Benchmark Model
Model now used by Cdiscount is KNN over TF-IDF on the title and description of the items (with a lemmatisation to prepare the data). In production, they set up a threshold value for the level of confidence (i.e. the probability) of the predictions, in order to guarantee an error rate lower that some imposed value. They are able to classify two-thirds of the products with the imposed level of accuracy (classify products only on categories where they can reach more than 90% of accuracy). Cdiscount have been working on classifying products based on their images - best results so far obtained using Inception v3 and the accuracy around 70.

In the competition the participants already tired [different (single) models]( https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41652) with the following benchmarks:

Resnet 101, no bagging, no TTA (test time augmentation), pre-trained, 4 x GTX 1080 Ti, 18 epochs x 4.5 hours, I have 0.69 after 7th epoch: 
 - val_loss = 1.18
 - val_acc = 0.744
 - LB = 0.7437
 
SE-ResNet-50, SGD with Nesterov momentum (0.9) and batch_size=256:
 - finetuning about 7 epochs (about 16 hours per epoch on 2x1080)
 - train augmentation: 161x161 random crops + horizontal flips
 - val single crop accuracy: 70.96
 - val 10 crops accuracy: 71.79
 - LB 10 ceps accuracy: 71.67
 
Resnet-18, simple SGD, lr 0.1 -> 0.01 -> 0.001, change lr after 2 epochs, just change FC layer to get the 5270 classes and train:
 - trained for 6 epochs
 - val_acc = 0.63
 - 180x180, no data augmentation no crop
 
Inception_v3, 160 crop:
 - val_loss = 1.78~
 - val_acc = 0.66~
 - LB = 0.673~


## Evaluation Metrics
This competition is evaluated on the categorization accuracy of the predictions of 5,000 categories (the percentage of products we get correctly). Currently the top scores are around 70%.

## Project Design


### PRE-PROCESSING AND HARDWARE


#### Hardware
I'll use primary EC2 p2.xlarge 61 GiB RAM and 12 GiB of GPU with SSD drive. To cut pricess I'll try to take the advantage of [EC2 Spot pricing.](https://aws.amazon.com/fr/ec2/spot/pricing/)

I'll also try deepsense.ai, signing up at https://neptune.ml will [get me $100 to train my models.](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41478). The full functional deepsense.ai experiment example can be found [here.](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41506)


#### Downloading Data:
The training data is huge 60G for training and 15G for testing set. To unsure the quality of download I'll validate download with sha256 hash following this [tutorial](https://www.kaggle.com/blazeka/validate-download-with-sha256-hash). 


#### Loading Data:
The size of data is big - it's impossible to load all data in memory. It can cause a lot of trouble - the training time is already huge and chosing wrong loading model can add too many hours to the training process and iterations. Fortunately there is a few kernels proposed by other participants in the competition:

There is a [kernel](https://www.kaggle.com/inversion/processing-bson-files) created by Cdiscount about how to load BSON files in python notebook. If it will be difficult to load all data at once I'll use MongoDB. Here is a great [tutorial](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41193) on how to do it.

Antoher kernel: [Keras generator for reading directly from BSON.](https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson) This one indexes the image file and feed the images in batches with random access directly from original BSON files.

[Fast Thread Safe Keras Generator from bin files.](https://www.kaggle.com/aloisiodn/fast-thread-safe-keras-generator-from-bin-files) This approach eliminates the need of random access, thus improving access time, with the drawback of using more disk space.

[Naive way to convert BSON to files](https://www.kaggle.com/bguberfain/not-so-naive-way-to-convert-bson-to-files)
This script will create one folder per category and save all respective images on it.


#### Preprocess input mode for pretrain Keras model:
When using pretrain models with Keras there is two preprocessing mode (caffe (0-center each color channel) al and tensorflow(scale pixels between -1 and 1)). Sklearn can complain about the stratified split with 10000 values because some classes end up with just 1 product. I'll do a simple split instead of stratified by removing the 'stratify=category_ids' parameter.


#### Data augmentation and trasformation:
There is two ways in how to use data augmentaion upsampling and downsampling. 
I'll try to do an upsampling from 180x180 to 224x224 and adding some data augmentation. But apparently it makes more sense to use dilated convolutions instead, since then the input is not interpolated, but it still maintains high resolution feature maps. The low-level layers can still stay the same and use the imagenet weights as initialization since these are pretty much only simple filters (which for example react to edges, like Sobel). The mid- to high-level filters will probably have to change a lot though, it may make sense to use dilated convolutions there. I'll also try downsampling by cropping from 180x180 to 160x160 as suggested by many posts in the forum.
https://github.com/Cadene/pretrained-models.pytorch/issues/8
https://www.reddit.com/r/MachineLearning/comments/52drsq/what_is_dilated_convolution/
http://colah.github.io/posts/2014-12-Groups-Convolution/

By looking at the images it can be tempting to desaturate photos (RGB -> NB). But apparently the models got stuck with only colour/saturation augmentation enabled. The ones are doing much better only had random crop/scale and rotation. The prevalence of almost constant black or white backgrounds in most images means that adding variability to that requires much more learning than leaving it as it is. Another angle to explore is to see if disabling the random crop and just leaving h-flip and a bit of rotation does better, basically assuming that the preprocessing of the competition dataset leaves things relatively well centred and similarly scaled across train and test datasets. There are many classes with only a handful of images. For these classes data augmentation may be more important than for the classes with 80,000 images.

There is a lot of different advices on the forum, I'll try to explore e.g.:
" To get 0.68 I took 10 crops for each test image (4 corners + center, and their horizontal flips) and did a prediction for each, then took the average of those predictions. So for products with 4 images I actually did separate 40 predictions. This technique scores higher than just doing one prediction per image (which scored 0.6435 on the LB, using the exact same model)."

Important note: augmentation by CPU can causes big bottleneck and extend training time.
[Keras Data Augmentation](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

### MODEL TRAINING


#### Training/Validation split:
I'll try to use 80/20 or 90/10 by either using a random split or a stratified split. The probleme with the last is the data distribution - some clases have a lot of examples and other just a few. 


#### Keras Importance Sampling:
Imbalance probleme is present in this cahllenge - the distribution of the samples vary a lot. The use of importance sampling can help. Importance sampling focuses the computation to informative/important samples (by sampling mini-batches from a distribution other than uniform) thus accelerating the convergence. Also importance sampling has been successfully used to accelerate stochastic optimization in many convex problems (this method results in 30% faster training of a CNN for CIFAR10 than when using uniform sampling).
http://idiap.ch/~katharas/importance-sampling/
https://github.com/idiap/importance-sampling

["Sample Importance in Training Deep Neural Networks".](https://openreview.net/forum?id=r1IRctqxg)
["A systematic study of the class imbalance problem in convolutional neural networks".](https://arxiv.org/pdf/1710.05381.pdf)
["Biased Importance Sampling for Deep Neural Network Training".](https://arxiv.org/abs/1706.00043)


#### The use of hard samples:
Based on the prediction scores, it can identify which training (and testing) samples are difficult. Training and testing datasets contain an overwhelming number of easy examples and a small number of hard examples (just a few examples per class). Automatic selection of these hard examples can make training more effective and efficient. We can speedup training with less samples.

[Original disscusion on Kaggle.](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41523)
["Curriculum Learning with Deep Convolutional Neural Networks".](http://kth.diva-portal.org/smash/get/diva2:878140/FULLTEXT01.pdf) Apparently we can speedup training with less samples. 
["Training Region-based Object Detectors with Online Hard Example Mining".](https://arxiv.org/abs/1604.03540) OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use.


#### Layer Configuration and Architecture:
I'll use pre-trained networks like Xception, Inception_v3, SE-ResNet-50, etc... by only adding a 5270-element dense layer with softmax activation function at the end and also set the last "block" of the network to be trainable (so it trains the classification layer and fine-tunes the last N layers of the network). This is gonna be my base model. I'll also try to freez/unfreez certain layers and adding more layers in the last block.

[Neural Network Architectures](https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba)
[Using Bottleneck Features for Multi-Class Classification in Keras and TensorFlow](http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html)
[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
[VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)
[AN ANALYSIS OF DEEP NEURAL NETWORK MODELS FOR PRACTICAL APPLICATIONS](https://arxiv.org/pdf/1605.07678.pdf)


#### Accelerating training of Deep Convolutional Networks:
I'll try to train for a few epoch first. Best parameters gives accuracy of + 0.0xx per epoch. So I don't have to train too many epoch at first to see if I have a good model.
[Accelerating Very Deep Convolutional Networks for Classification and Detection.](https://arxiv.org/abs/1505.06798)


#### Learning Rate:
I'll start by using 0.01 for epoch 1 and 2, 0.001 for epoch 3, 0.0001 till epoch 4.
["Reinforcement Learning for Learning Rate Control"](https://arxiv.org/abs/1705.11159), an algorithm to automatically learn learning rates using neural network based actor-critic methods from deep reinforcement learning (RL).


#### Batch Size:
I'll try different batch sizes from 64 to 1024. There is a lot of discussion on forum about the right number of batch size in different models and how it can affect training time and accuracy. I need to investigate further about the different effects of the batch size.
["Online Batch Selection for Faster Training of Neural Networks".](https://arxiv.org/abs/1706.00043)


#### Optimizer and Momentum:
I'll start by using SDG + momentum. It's the most used optimiser in this competition, but I'll try to investigate other  
Apparently reducing momentum can improve results, reducing convergence time by half.



### PREDICTION AND SUBMITION
For each _id in the test set, I'll predict a category_id. After I'll submit the file should containing a header and have the following format:

_id,category_id
2,1000000055
5,1000016018
6,1000016055
etc.

This competition is evaluated on the categorization accuracy of predictions (the percentage of products you get correct).


### DIFFERENT TRICKS AND TIPS
https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/40937
https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/40934
[What's your approach on this competition?](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/40033)

-----------

