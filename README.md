# Unsupervised Learning:Clustering

## Introduction to Clustering

### Introduction
Welcome to the module on 'Unsupervised Learning'. In the previous modules, you learnt about several supervised learning techniques such as regression and classification. These techniques use a training set to make the algorithm learn, and then apply what it learnt to new, unseen data points.

In this module, you will be introduced to unsupervised learning.

#### In this session
You will start by learning what “clustering” is. It is an unsupervised learning technique, where you try to find patterns based on similarities in the data. Then, you will be introduced to a case study that shows the applicability of clustering in the industry.

You will learn the two most commonly used types of clustering algorithms - **K-Means Clustering** and **Hierarchical Clustering**, as well as their application in Python. Then, you will also look at what segmentation is and how it is different from clustering.

## Understanding Clustering
In the previous modules, you saw various supervised machine learning algorithms. Supervised machine learning algorithms make use of labelled data to make predictions.

For example, an email will be classified as spam or ham, or a bank’s customer will be predicted as ‘good’ or ‘bad’. You have a target variable Y which needs to be predicted.
 
On the other hand, in unsupervised learning, you are not interested in prediction because you do not have a target or outcome variable. The objective is to discover interesting patterns in the data, e.g. are there any subgroups or ‘clusters’ among the bank’s customers?

Let’s learn clustering in detail.

### PRACTICAL APPLICATIONS OF CLUSTERING
1. **Customer Insight**: Say, a retail chain with so many stores across locations wants to manage stores at best and increase the sales and performance. Cluster analysis can help the retail chain to get desired insights on customer demographics, purchase behaviour and demand patterns across locations. This will help the retail chain for assortment planning, planning promotional activities and store benchmarking for better performance and higher returns.
2. **Marketing**: Cluster Analysis can help with In the field of marketing, Cluster Analysis can help in market segmentation and positioning, and to identify test markets for new product development.
3. **Social Media**: In the areas of social networking and social media, Cluster Analysis is used to identify similar communities within larger groups.
4. **Medical**: Cluster Analysis has also been widely used in the field of biology and medical science like human genetic clustering, sequencing into gene families, building groups of genes, and clustering of organisms at species. 

In the next segment, you will be introduced to a real-life application of clustering — grouping customers of an online store into different clusters and making a separate targeted marketing strategy for each group. We will be using this example throughout the module.

## Practical Example of Clustering - Customer Segmentation
In the last segment, you got a basic idea of what clustering is. So let’s consider a real-life application of the unsupervised clustering algorithm.

**You can download the data set for the case study from below. We will be using the same data for Python Lab.** <br/>
[Online Retail Data Set](dataset/Online+Retail.csv)

Lets say you run an online retail store and you want to group your exsisting customers into different segments so as to make specific marketing strategies for them. This is called customer segmentation. Now we don't have any predefined labels to classify the customers into nor we know what will be the charachteristic of the group that eventually will be form. Its only after the grouping have been form we can analyse the charachteristic of the group to derieve actionable insights. Thus such a problem is classified as unsupervised clustering problem.

![title](img/customer-segmentation.JPG)

Customer segmentation for targeted marketing is one of the most vital applications of the clustering algorithm. Here, as a manager of the online store, you would want to group the customers into different clusters, so that you can make a customised marketing campaign for each of the group. You do not have any label in mind, such as good customer or bad customer. You want to just look at patterns in customer data and then try and find segments. This is where clustering techniques can help you with segmenting the customers. Clustering techniques use the raw data to form clusters based on common factors among various data points. This is exactly what will also be done in segmentation, where various people or products will be grouped together on the basis of similarities and differences between them.

As a manager, you would have to decide what the important business criteria are on which you would want to segregate the customers. So, you would need a method or an algorithm that itself decides which customers to group together based on this criteria.

Sounds interesting? Well, that is the beauty of unsupervised learning, especially clustering. But before we conclude this introductory session, it would be best to get an industry perspective on the application of clustering in the world of analytics.

**Difference between clustering and segmentation**
1. Clustering is an analytics technique.
2. Segmentation is a business problem/case.

To be able to do segmentation we use clustering technique. Clustering technique can be used in multiple places. We can segment/cluster multiple things:

![title](img/segmentation.JPG)

![title](img/segmentation1.png)


You saw that, for successful segmentation, the segments formed must be stable. This means that the same person should not fall under different segments upon segmenting the data on the same criteria. You also saw that segments should have **intra-segment homogeneity** and **inter-segment heterogeneity**. You will see in later sessions how this can be defined mathematically.

![title](img/inter-intra.png)

![title](img/inter-intra1.png)

Now you will see what types of market segmentations are commonly used.

The 3 types of segmentation are used for customer segmentation:

1. **Behavioural segmentation**: Segmentation is based on the actual patterns displayed by the consumer
2. **Attitudinal segmentation**: Segmentation is based on the beliefs or the intents of people, which may not translate into similar action
3. **Demographic segmentation**: Segmentation is based on the person’s profile and uses information such as age, gender, residence locality, income, etc.

#### Additional reading
You can read more about business cases where clustering is used here (https://www.jigsawacademy.com/cluster-analysis-for-business/)4

#### Summary
In this session, we covered the basics of unsupervised learning and also got a little idea about how clustering works. In the next sessions, you will go deeper into the details of clustering and learn about the 2 common clustering algorithms — the K-Means algorithm and the Hierarchical clustering algorithm.