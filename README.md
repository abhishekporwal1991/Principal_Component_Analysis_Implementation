# Principal_Component_Analysis_Implementation
Understanding of Principal Component Analysis (PCA) and Implementation from scratch.


## What is PCA and need of PCA?
PCA stands for Principal Component Analysis. In a day to day machine learning tasks you might have encountered a problem of high dimensionality of the dataset (telecom/banking data). The problem with such a data is that it leads to overfitting problem in the model. The available features are generally highly correlated with each other and are redundant to the model. Adding these variables to the model increase the complexity of the model as well as inflate the model coefficients (Multicollinearity).

The solution to the problem is to reduce the dimensionality of the dataset and PCA is one such technique.
PCA is a method of summarizing data with fewer characteristics. It constructs some new characteristics. These new characteristics are constructed using the old ones (Linear combinations).


## Knowledge Map
Understanding of PCA required understanding of several small small concepts. We will learn them one after the other in the following manner - 
![Imgur](https://i.imgur.com/9JL4VLf.jpg)


## Prerequisites
There are some prerequisites to fully understand the PCA like -
* Linear Algebra
* Projection of vector onto another vector
* Covariance matrix
* Eigen-decomposition (Spectral Theorem)

### Linear Algebra
Understanding of [Linear transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=4&t=0s), [determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=7&t=0s), [change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA) and [eigenvectors & eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) are essential to understand PCA. 

In simplistic words, Eigenvectors are those vectors which won't change direction in case of linear transformation. They only scaled up/down as per corresponding eigenvalue of the eigenvector. The linear transformations are very helpful sometime to reduce the complexity of the problem. This makes eigenvectors even more important to understand linear algebra.

### Vector Projection
![Imgur](https://i.imgur.com/GGLsuVg.jpg)


The definition of vector projection for the indicated red vector is the called ![Imgur](https://i.imgur.com/9CtZlBZ.jpg). When you read ![Imgur](https://i.imgur.com/9CtZlBZ.jpg), you should say "the vector projection of v onto u." This implies that the new vector is going in the direction of u. The vector projection is the vector produced when one vector is resolved into two component vectors, one that is parallel to the 2nd vector and one that is perpendicular to the 2nd vector. The parallel vector is the vector projection.

Formula for vector projection is -   
![Imgur](https://i.imgur.com/0b2pO3f.jpg) 

Projection of vector `v` onto another vector `u` in matrix form is -  
![Imgur](https://i.imgur.com/0WLVbSw.jpg)

### Covariance Matrix
Variance can be used to explain the spread of the data in the directions parallel to the axes of the feature space. Consider the 2D feature space shown by figure:  
![Imgur](https://i.imgur.com/gij2KhB.jpg)  
Figure - The diagonal spread of the data is captured by the covariance

For this data, we could calculate the variance Var(x,x) in the x-direction and the variance Var(y,y) in the y-direction. However, the horizontal spread and the vertical spread of the data does not explain the clear diagonal correlation. Figure above clearly shows that on average, if the x-value of a data point increases, then also the y-value increases, resulting in a positive correlation. This correlation can be captured by extending the notion of variance to what is called the ‘covariance’ of the data.  

These four values can be summarized in a matrix, called the covariance matrix:  
![Imgur](https://i.imgur.com/cbZZz43.jpg)

If x is positively correlated with y, y is also positively correlated with x. In other words, we can state that Var(x,y) = Var(y,x). Therefore, the covariance matrix is always a `symmetric matrix` with the variances on its diagonal and the covariances off-diagonal. Two-dimensional normally distributed data is explained completely by its mean and its 2 x 2 covariance matrix. Similarly, a 3 x 3 covariance matrix is used to capture the spread of three-dimensional data, and a N x N covariance matrix captures the spread of N-dimensional data.  
![Imgur](https://i.imgur.com/uaDHT9S.jpg)  
Figure - The covariance matrix defines the shape of the data

### Eigendecomposition of covariance matrix
The covariance matrix defines both the spread (variance), and the orientation (covariance) of our data. So, if we would like to represent the covariance matrix with a vector and its magnitude, we should simply try to find the vector that points into the direction of the largest spread of the data, and whose magnitude equals the spread (variance) in this direction.

If we define this vector as ![Imgur](https://i.imgur.com/kfgPtLX.jpg), then the projection of our data ![Imgur](https://i.imgur.com/2YdqrdT.jpg) onto this vector is obtained as ![Imgur](https://i.imgur.com/Bm2PIvh.jpg), and the variance of the projected data is ![Imgur](https://i.imgur.com/6yeKmUd.jpg)[[source]](https://stats.stackexchange.com/questions/32174/pca-objective-function-what-is-the-connection-between-maximizing-variance-and-m/136072#136072). Since we are looking for the vector ![Imgur](https://i.imgur.com/kfgPtLX.jpg) that points into the direction of the largest variance, we should choose its components such that the covariance matrix ![Imgur](https://i.imgur.com/6yeKmUd.jpg) of the projected data is as large as possible. Maximizing any function of the form ![Imgur](https://i.imgur.com/6yeKmUd.jpg) with respect to ![Imgur](https://i.imgur.com/kfgPtLX.jpg), where ![Imgur](https://i.imgur.com/kfgPtLX.jpg) is a normalized unit vector, can be formulated as a so called [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient). The maximum of such a Rayleigh Quotient is obtained by setting ![Imgur](https://i.imgur.com/kfgPtLX.jpg) equal to the largest eigenvector of matrix ![Imgur](https://i.imgur.com/WKIeUQy.jpg).

In other words, the largest eigenvector of the covariance matrix always points into the direction of the largest variance of the data, and the magnitude of this vector equals the corresponding eigenvalue.

We can convert our covariance matrix to the form ![Imgur](https://i.imgur.com/QBM5YjX.jpg) using the [Spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem) where P is the matrix of eigenvectors and D is the diagonal matrix with eigenvalues on the diagonal and values of zero everywhere else. The eigenvalues on the diagonal of D will be associated with the corresponding column in P.
![Imgur](https://i.imgur.com/QBM5YjX.jpg) form is called the eigendecomposition of the covariance matrix.

## Importing data for implementing PCA
As we have understood the prerequisites of PCA, now we need a dataset to implement our understanding. We are going to use python jupyter notebook for the [implementation of PCA](https://github.com/abhishekporwal1991/Principal_Component_Analysis_Implementation/blob/master/PCA_Boston_Pricing_data.ipynb).

The dataset we are using is `Boston Pricing Data`. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The dataset contains 506 observations, 13 features and 1 target variable. The variable description is given below -  

| Column Name | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| CRIM        | per capita crime rate by town                                |
| ZN          | proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS       | proportion of non-retail business acres per town             |
| CHAS        | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
| NOX         | nitric oxides concentration (parts per 10 million)           |
| RM          | average number of rooms per dwelling                         |
| AGE         | proportion of owner-occupied units built prior to 1940       |
| DIS         | weighted distances to five Boston employment centres         |
| RAD         | index of accessibility to radial highways                    |
| TAX         | full-value property-tax rate per $10,000                     |
| PTRATIO     | pupil-teacher ratio by town                                  |
| B           | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
| LSTAT       | % lower status of the population                             |
| MEDV        | Median value of owner-occupied homes in dollor 1000          |


Following line of code is used to import the data into jupyter notebook:    

![Imgur](https://i.imgur.com/4rIgAKq.jpg)

## Understanding the dataset
A data scientist should have a thorough understanding of the dataset before using it from any machine learning task. Performance of any ML algorithm largely dependent on the input data.

### Exploratory Data Analysis (EDA)
* `Missing values`: Data do not have any missing value:  

![Imgur](https://i.imgur.com/Jn2JC1X.jpg)

* `Histograms`: Histograms can be used to visualize the distribution of the individual variables. As you can see, most of features are not following the normal distribution and are skewed. Histogram for different features: 

![Imgur](https://i.imgur.com/yHu7QB6.jpg)

* `Boxplots`: Boxplots are also useful in visualizing distribution of data. Information of existing outliers in the data can also be visualized through the boxplots. Boxplots of different features are as follows:  

![Imgur](https://i.imgur.com/K61B9HK.jpg)

* `Target distribution`: Distribution of target variable is close to the normal distribution. Figure below -  

![Imgur](https://i.imgur.com/2Z52rzK.jpg)

* `Correlation Matrix`: This matrix is used to identify the correlation between the features. It can be seen that some of the variables are highly correlated with each other and will cause the problem of multicollinearity if any two highly correlated variables are added to the model. Correlation matrix is shown below -   

![Imgur](https://i.imgur.com/fdkDoUU.jpg)

## Getting Principal Components
To get the principal components from the given data there are certain steps to follow - 

**1. Compute the d-dimensional mean vector where d in the number of features in the datadet -**

![Imgur](https://i.imgur.com/fM7lw5V.jpg)

**2. Compute the covariance matrix of the dataset -**  

![Imgur](https://i.imgur.com/Rd1e5RR.jpg)

The shape of the matrix is 13 x 13 as the data have 13 features.

**Graphical representation of correlation matrix -**

![Imgur](https://i.imgur.com/R5EsMrS.jpg)

**3. Compute eigenvectors and corresponding eigenvalues -**

![Imgur](https://i.imgur.com/Jh1vfsF.jpg)

**4a. Sort the eigenvectors by decreasing eigenvalues -**

![Imgur](https://i.imgur.com/dKf2exe.jpg)

**4b. Choose k eigenvectors with the largest eigenvalue to form a d x k dimensional matrix W -**

![Imgur](https://i.imgur.com/URlZQHw.jpg)

**5. Use this d x k eigenvector matrix to transform the samples onto the new subspace -**

![Imgur](https://i.imgur.com/CKLtv0G.jpg)

### PCA using sklearn library
PCA is implemented using the sklearn.decomposition library for comparison purpose. Similar results are achieved using this in built implementation - 

![Imgur](https://i.imgur.com/9ZBFObb.jpg) 

## Visualizing PCA
For visualization purpose target variable is been binned into 3 price range categories i.e. low (0-1st quartile), medium (1-3rd quartile) and high (3rd quartile & more) -

![Imgur](https://i.imgur.com/k7mNpgh.jpg)


### Scree Plot
Scree plot is used to visualize the cumulative percentage variance explained by the principal components. Here two principal components cumulatively explaining `approximately 60% variance` of the target variable.

![Imgur](https://i.imgur.com/WMm0w1j.jpg) 

## Conclusion
PCA can be used as an effective tool for dimensionality reduction. As the results are showing that we can able to explain upto 90% of variance just using 8 features out of 13 features.

One of the drawback of the technique is that it makes independent variable less interpretable. 

## References
[Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g)  
[Vector Projection](https://www.ck12.org/book/CK-12-College-Precalculus/section/9.6/)  
[Covariance matrix](https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/#id3483335494)  
[Spectral Theorem](https://www.youtube.com/watch?v=13r9QY6cmjc&list=PL49CF3715CB9EF31D&index=22)  
[Understaning PCA 1](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)  
[Understanding PCA 2](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues)  
[Implementing PCA](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)