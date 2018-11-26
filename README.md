# Opinion-mining-on-movie-reviews
CHAPTER 1 : INTRODUCTION

1.1 AIM / OBJECTIVE

The problem statement involves performing sentiment analysis and classification as well as uncovering the attitude of the author on a particular topic from the written text alternatively known as “opinion mining” and “subjectivity detection”. Also to use natural language processing and machine learning techniques to find statistical and/or linguistic patterns in the text that reveal attitudes.

1.2 SCOPE OF THE PROJECT

This project works for datasets containing positive and negative reviews for a particular movie. All the reviews are literary and pre classified so that we can train the model and test our accuracies easily. 
For each class of data - positive, negative and neutral each of size 12500 reviews, it takes more than 6 hours to process and analyse. So we reduce the size of the dataset by about 50% to satisfy the boundaries of the course project.
We use a Tab-separated-values file to provide the input to the program. The first column provides gives the result of the review ie., 0 if it is a negative review and 1 if it is a positive review followed a literary review of the movie.

1.3 APPLICATIONS AND BENEFITS

Sentiment analysis has many applications and benefits to your business and organization. It can be used to give your business valuable insights into how people feel about your product brand or service.
When applied to social media channels, it can be used to identify spikes in sentiment, thereby allowing you to identify potential product advocates or social media influencers. It can be used to identify when potential negative threads are emerging online regarding your business, thereby allowing you to be proactive in dealing with it more quickly.Sentiment analysis could also be applied to your corporate network, for example, by applying it to your email server, emails could be monitored for their general “tone”. 

CHAPTER 2 : DISCUSSION

2.1 OVERVIEW

People have always had an interest in what people think, or what their opinion is.  Since the inception of the internet, increasing numbers of people are using websites and services to express their opinion. With social media channels such as Facebook, LinkedIn, and Twitter, it is becoming feasible to automate and gauge what public opinion is on a given topic, news story, product, or brand.
Opinions that are mined from such services can be valuable. Datasets that are gathered can be analyzed and presented in such a way that it becomes easy to identify if the online mood is positive, negative or even indifferent. This allows individuals or business to be proactive as opposed to reactive when a negative conversational thread is emerging.  Alternatively, positive sentiment can be identified thereby allowing the identification of product advocates or to see which parts of a business strategy are working.

Sentiment analysis has become very popular over the years. Every manufacturer, service provider wants to know how much a customer likes their product or service. With the increasing number of review blogs and forums, a vast resource of data has been created which can be mined for sentiment on a particular entity. For example, the entire cast and crew of a movie would want to know the public opinion of their movie. State of the art opinion miner has been developed using supervised classification methods. The fundamental working principle of such techniques is feature extraction. In any machine learning approach feature extraction and selection has to be done manually, thus posing a challenge.
The Sentiment found within comments, feedback or critiques provide useful indicators for many different purposes and can be categorized by polarity. By polarity we tend to find out if a review is overall a positive one or a negative one. For example:
Positive Sentiment in subjective sentence: “I loved the movie Mary Kom”—This sentence is expressed positive sentiment about the movie Mary Kom and we can decide that from the sentiment threshold value of word “loved”. So, threshold value of word “loved” has positive numerical threshold value.
Negative sentiment in subjective sentences: “Phata poster nikla hero is a flop movie” defined sentence is expressed negative sentiment about the movie named “Phata poster nikla hero” and we can decide that from the sentiment threshold value of word “flop”. So, threshold value of word “flop” has negative numerical threshold value.

 
2.2 NAIVE BAYES CLASSIFIER 

Bayesian network classifiers are a popular supervised classification paradigm. A well-known Bayesian work classifier is the Naïve Bayes’ classifier is a probabilistic classifier based on the Bayes’ theorem, considering Naïve (Strong) independence assumption.
It was introduced under a different name into the text retrieval community and remains a popular(baseline) method for text categorization. The problem of judging documents as belonging to one category or the other with word frequencies as the feature. An advantage of Naïve Bayes’ is that it only requires a small amount of training data to estimate the parameters necessary for classification. 
Abstractly, Naïve Bayes’ is a conditional probability model. Despite its simplicity and strong assumptions, the naïve Bayes’ classifier has been proven to work satisfactorily in many domains.
Bayesian classification provides practical learning algorithms and prior knowledge and observed data can be combined. In Naïve Bayes’ technique, the basic idea is to find the probabilities of categories given a text document by using the joint probabilities of words and categories. It is based on the assumption of word independence. The starting point is the Bayes’ theorem for conditional probability, stating that, for a given data point x and class C:

 P (C / X) = P(X / C) * P(C)P(X)

Where,
P(C | X) is the probability of data X to fall in the given class C. This is called the posterior probability.
P(X | C) is the probability of data d to lie in the class C is true.
P(C) is the probability of the class (regardless of the data). This is called the prior probability of C.
P(X) is the probability of the data (regardless of the hypothesis).

We are interested in calculating the posterior probability of P(C | X) from the prior probability p(C) with P(X) and P(X | C).
After calculating the posterior probability for a number of different hypotheses, you can select the hypothesis with the highest probability. This is the maximum probable hypothesis and may formally be called the maximum a posteriori (MAP) hypothesis.

This can be written as:
MAP( C ) = max(P( C | X))
or
MAP(C) = max((P(X | C) * P(C)) / P(X))
or
MAP(C) = max(P(X|C) * P(C))
The P(X) is a normalizing term which allows us to calculate the probability. We can drop it when we are interested in the most probable hypothesis as it is constant and only used to normalize.

Back to classification, if we have an even number of instances in each class in our training data, then the probability of each class (e.g. P(C)) will be equal. Again, this would be a constant term in our equation and we could drop it so that we end up with:
MAP(C) = max(P(X|C))

Furthermore, by making the assumption that for a data point x = {x1,x2,...xj}, the probability of each of its attributes occurring in a given class is independent, we can estimate the probability of x as follows:

 P(C/x)=P(C) * ∏P(xi/C)


2.3 k- Nearest Neighbour CLASSIFIER

This classifier is a Lazy Classifier, i.e, these classifiers delay the process of modeling the training data until it is needed to classify the test examples. A nearest-neighbor classifier represents each example as a data point in a d-dimensional space, where d is the number of attributes. Given a test example, we compute its proximity to the rest of the data points in the training set. The k-nearest neighbors of a given example z refer to the k points that are closest to z. It is non parametric method used for classification or regression. In case of classification the output is class membership (the most prevalent cluster may be returned) , the object is classified by a majority vote of its neighbours, with the object being assigned to the class most common among its k nearest neighbours. This rule simply retains the entire training set during learning and assigns to each query a class represented by the majority label of its k-nearest neighbours in the training set.

Choosing the value for k is an important task. If k is too small, then the nearest-neighbor classifier may be susceptible to overfitting because of noise in the training data. On the other hand, if k is too large, the nearest-neighbor classifier may misclassify the test instance because its list of nearest neighbors may include data points that are located far away from its neighborhood.
The Nearest Neighbour rule (NN) is the simplest form of K-NN when K = 1. Given an unknown sample and a training set, all the distances between the unknown sample and all the samples in the training set can be computed. The distance with the smallest value corresponds to the sample in the training set closest to the unknown sample. Therefore, the unknown sample may be classified based on the classification of this nearest neighbour.

Nearest-neighbor classification is part of a more general technique known as instance-based learning, which uses specific training instances to make predictions without having to maintain an abstraction (or model) derived from data. Instance-based learning algorithms require a proximity measure to determine the similarity or distance between instances and a classification function that returns the predicted class of a test instance based on its proximity to other instances.
Lazy learners such as nearest-neighbor classifiers do not require model building. However, classifying a test example can be quite expensive because we need to compute the proximity values individually between the test and training examples. In contrast, eager learners often spend the bulk of their computing resources for model building. Once a model has been built, classifying a test example is extremely fast.
Nearest-neighbor classifiers make their predictions based on local information, whereas decision tree and rule-based classifiers attempt to find a global model that fits the entire input space. Because the classification decisions are made locally, nearest-neighbor classifiers (with small values of k ) are quite susceptible to noise. Nearest-neighbor classifiers can produce arbitrarily shaped decision boundaries. Such boundaries provide a more flexible model representation compared to decision tree and rule-based classifiers that are often constrained to rectilinear decision boundaries. The decision boundaries of nearest-neighbor classifiers also have high variability because they depend on the composition of training examples. Increasing the number of nearest neighbors may reduce such variability.

The K-NN is an easy algorithm to understand and implement , and a powerful tool we have at our disposal for sentiment analysis. KNN is powerful because it does not assume anything about the data, other than a distance measure can be calculated consistently between two instances. As such, it is called non-parametric or non-linear as it does not assume a functional form.

2.4 LOGISTIC REGRESSION

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

In logistic regression, the dependent variable is binary or dichotomous, i.e. it only contains data coded as 1 (TRUE, success, etc.) or 0 (FALSE, failure, etc.).

The goal of logistic regression is to find the best fitting (yet biologically reasonable) model to describe the relationship between the dichotomous characteristic of interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables. Logistic regression generates the coefficients (and its standard errors and significance levels) of a formula to predict a logit transformation of the probability of presence of the characteristic of interest.

Logistic regression is named for the function used at the core of the method, the logistic
function.

The logistic function, also called the sigmoid function was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

2.5 SUPPORT VECTOR MACHINES

Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However, it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyperplane that differentiate the two classes very well.
This classification technique has received considerable attention. This technique has its roots in statistical learning theory and has shown promising empirical results in many practical applications, from handwritten digit recognition to text categorization. SVM also works very well with high-dimensional data and avoids the curse of dimensionality problem. Another unique aspect of this approach is that it represents the decision boundary using a subset of the training examples, known as the support vectors.
There are infinitely many hyperplanes that are used for classification for a problem . Although their training errors are zero there is no guarantee that the hyperplanes will perform equally well on previously unseen examples. The classifier must choose one of these hyperplanes to represent its decision boundary, based on how well they are expected to perform on test examples.
The distance between the classification hyperplane and the nearest training point is called a
margin.The goal of SVM is to maximize the margin, so that test data is classified correctly.

2.6 DATASET DESCRIPTION

The dataset we are using is provided by Kaggle. It is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. It has a set of 8000 highly polar movie reviews for training and testing. 

The dataset is in the form of a TSV file (Tab separated values) in which the first column represents the class of the movie review (1 for positive and 0 for negative)

Total number of Instances : 
Positive : 4000
Negative : 4000

