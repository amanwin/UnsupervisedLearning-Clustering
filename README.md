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

### K Means Clustering

#### Introduction

Welcome to the session on 'K-Means Clustering'. In the previous session, you got a basic idea of what unsupervised learning is. You also learnt about one such unsupervised technique called clustering. Now let's dive deeper into the concept and learn about the first common algorithm to achieve this unsupervised clustering — the K-Means algorithm.

### Steps of the Algorithm
Let’s go through the K-Means algorithm using a very simple example. Let’s consider a set of 10 points on a plane and try to group these points into, say, 2 clusters. So let’s see how the K-Means algorithm achieves this goal.

Before moving ahead, think about the following problem. Let’s say you have the data of 10 students and their marks in Biology and Math (as shown in the plot below). You want to divide them into two clusters so that you can see what kind of students are there in the class.

The y-axis shows the marks in Biology, and the x-axis shows the marks in Math.

Imagine two clusters dividing this data — one red and the other yellow. How many points would each cluster have?

![title](img/clusters.JPG)

### Centroid
The K-Means algorithm uses the concept of the centroid to create K clusters. Before you move ahead, it will be useful to recall the concept of the centroid (https://en.wikipedia.org/wiki/Centroid).

In simple terms, a centroid of n points on an x-y plane is another point having its own x and y coordinates and is often referred to as the geometric centre of the n points.

For example, consider three points having coordinates (x1, y1), (x2, y2) and (x3, y3). The centroid of these three points is the average of the x and y coordinates of the three points, i.e.

(x1 + x2 + x3 / 3, y1 + y2 + y3 / 3).

Similarly, if you have n points, the formula (coordinates) of the centroid will be:

(x1+x2…..+xn / n, y1+y2…..+yn / n). 

So let’s see how the K-Means algorithm achieves this goal. In K-means k stands for number of clusters.

![title](img/k-means.png)

Note that the initial choice of cluster centers(yellow and red one) is completely random. Then in step one we will allocate each point in the dataset to the nearest cluster center. To do this we calculate the distance of each datapoint to the two cluster centers i.e. the centroid and allocate the point to the centroid with the least distance. We will use euclidean distance as a common measure for calculating this.

![title](img/k-means1.png)

![title](img/k-means2.jpg)

Now we can see we have set of points allocated to the yellow centroid and a set of points allocated to the red centroid. So in effect we have two clusters now. This set is a assigment step where each point is assigned to a cluster.

![title](img/k-means3.jpg)

The next step is to recompute the center of each of these clusters which will simply be the mean of individual points in each of the clusters. Then we will get our new cluster centers. This is the optimasation step.

![title](img/k-means-optimsation.jpg)

Now again we will go to assignment step and will assign each datapoint to the nearest cluster center using the same method as discussed earlier.

We again update the position of the cluster center for both the clusters. We keep iterating through this process of assignment and optimisation till the centroids no longer update. At this point the algorithm has reached optimal grouping and we have got our two clusters. <br/>
So essentially K-means is a algorithm that take n datapoints and group them into k-clusters. Grouping is done in the way to maximize the tightness/closeness of the points among a cluster while maximizing the distance between the clusters.

[Clustering Activity File](dataset/Clustering_activity_K_Means.xlsx)

### K Means Algorithm
In the previous segment, we learned about K-means clustering and how the algorithm works using a simple example. We learned about how assignment and the optimisation work in K Means clustering, We will look K-means more algorithmically. We will be learning how the K Means algorithm proceeds with the assignment step and then with the optimisation step and will also be looking at the cost of function for the K-means algorithm.

Let's understand the K-means algorithm in more detail.
We understood that the algorithm’s inner-loop iterates over two steps:

![title](img/k-means-steps.JPG)

In the next segment, we will learn about the Kmeans cost function and will also see how to compute the cost function for each iteration in the K-means algorithm.

![title](img/cost-function.JPG)

So the cost function for the K-Means algorithm is given as: 

![title](img/cost-function1.JPG)

We will learn what exactly happens in the assignment step? and we will also look at how to assign each data point to a cluster using the K-Means algorithm assignment step.

In the assignment step, we assign every data point to K clusters. The algorithm goes through each of the data points and depending on which cluster is closer, in our case, whether the green cluster centroid or the blue cluster centroid; It assigns the data points to one of the 2 cluster centroids.

The equation for the assignment step is as follows:

![title](img/assignment-step.JPG)

Now having assigned each data point to a cluster, now we need to recompute the cluster centroids. In the next section, we will explain how to recompute the cluster centroids or the mean of each cluster.

In the optimisation step, the algorithm calculates the average of all the points in a cluster and moves the centroid to that average location.

The equation for optimisation is as follows:

![title](img/optimisation-step.JPG)

The process of assignment and optimisation is repeated until there is no change in the clusters or possibly until the algorithm converges.

In the next segment, we will learn how to look K-Means algorithm as a coordinate descent problem. We will also learn about the constraint of the K-Means cost function and how to achieve global minima.

**Additional Reading**
You may go through this document(http://thespread.us/clustering.html).

## K Means as Coordinate Descent
In the previous segment, we learned that the K-means algorithm iterate between two steps.
1. In the first step, we assign each observation to the nearest cluster centre.
2. In the second step, we update the cluster center.

If we carefully look into the first step in which we are assigning each step to the nearest cluster center, we are minimising the objective function. In general, we want the cluster assignment in such a way that the corresponding cost can be reduced.

![title](img/k-means-optimsation-step.png)

![title](img/optimisation-step1.JPG)

In the next segment, we will look that the K-Means cost function is a non-convex function, which means the coordinate descent is not guaranteed to converge to the global minimum and the cost function can converge to local minima. Choosing the initial value of K centroids can affect the K-Means algorithm and its final results.

![title](img/non-complex.png)

**Additional Reading**
1. In the previous lecture, we got an idea that the K-Means algorithm is not a convex clustering algorithm. To find a convex solution for K-means, there is a technique called Support Vector Clustering.(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.9806&rep=rep1&type=pdf) In Support Vector Clustering,  we describe a smooth boundary around the data points for which we need to state the length scale for the Gaussian Kernel to define how smooth we want the boundary to be. 
2. To understand more about Convex and Non-Convex cost function, you may go through this YouTube link (https://www.youtube.com/watch?v=LxjL_yLaFS8&feature=youtu.be).

## K Means++ Algorithm
We looked in the previous segment that for K-Means optimisation problem, the algorithm it iterate between two steps and tries to minimise the objective function given as,

![title](img/objective-function.JPG)

We also looked that the K-Means cost function is a non-convex function, which means the coordinate descent is not guaranteed to converge to the global minimum. The problem with K-Means is to initialise the cluster centers to achieve the global minima smartly. 

To choose the cluster centers smartly, we will learn about K-Mean++ algorithm. K-means++ is just an initialisation procedure for K-means. In K-means++ you pick the initial centroids using an algorithm that tries to initialise centroids that are far apart from each other.

![title](img/k-means++.JPG)

### Visualising the K Means Algorithm
Let’s see the K-Means algorithm in action using a visualisation tool. This tool can be found on naftaliharris.com (https://www.naftaliharris.com/blog/visualizing-k-means-clustering/). You can go to this link and play around with the different options available to get an intuitive feel of the K-Means algorithm.

Upon trying the different options, you may have noticed that the final clusters that you obtain vary depending on many factors, such as choice of the initial cluster centres and the value of K, i.e. the number of clusters that you want. You will understand these factors and other practical considerations while using the K-means algorithm in more detail in the next segment.

### Practical Consideration in K Means Algorithm
Let’s understand some of the factors that can impact the final clusters that you obtain from the K-means algorithm. This would also give you an idea about the issues that you must keep in mind before you start to make clusters to solve your business problem.

The major practical considerations involved in K-Means clustering are:
1. The number of clusters that you want to divide your data points into, i.e. the value of K has to be pre-determined.
2. The choice of the initial cluster centres can have an impact on the final cluster formation. The K-means algorithm is **non-deterministic**. This means that the final outcome of clustering can be different each time the algorithm is run even on the same data set. This is because, as you saw, the final cluster that you get can vary by the choice of the initial cluster centres.
3. The clustering process is very sensitive to the presence of outliers in the data.
4. Since the distance metric used in the clustering process is the Euclidean distance, you need to bring all your attributes on the same scale. This can be achieved through standardisation.
5. The K-Means algorithm does not work with categorical data.
6. The process may not converge in the given number of iterations. You should always check for convergence.

You will understand some of these issues in detail and also see the ways to deal with them when you implement the K-means algorithm in Python.

Now let's look in detail how to choose K for K-Means algorithm.

Having understood about the approach of choosing K for K-Means algorithm, we will now look at silhouette analysis or silhouette coefficient. Silhouette coefficient is a measure of how similar a data point is to its own cluster (cohesion) compared to other clusters (separation).

![title](img/silhouette.JPG)

![title](img/silhouette1.png)

**Additional reading**
You can read more about K-Mode clustering here(https://shapeofdata.wordpress.com/2014/03/04/k-modes/), We will be covering it in detail in the next section.

### Cluster Tendency
Before we apply any clustering algorithm to the given data, it's important to check whether the given data has some meaningful clusters or not? which in general means the given data is not random. The process to evaluate the data to check if the data is feasible for clustering or not is know as the clustering tendency.

As we have already discussed in the previous segment that the clustering algorithm will return K clusters even if that data does not have any clusters or have any meaningful clusters. So before proceeding for clustering, we should not blindly apply the clustering method and we should check the clustering tendency.

![title](img/hopkins-statistics.JPG)

To check cluster tendency, we use Hopkins test. Hopkins test examines whether data points differ significantly from uniformly distributed data in the multidimensional space.

**Additional Resources**

To read about Hopkins test in detail, please follow this link1 (https://www.datanovia.com/en/lessons/assessing-clustering-tendency/#methods-for-assessing-clustering-tendency), link2(https://stats.stackexchange.com/questions/332651/validating-cluster-tendency-using-hopkins-statistic), remember that the document is described using R programming, please ignore it.



