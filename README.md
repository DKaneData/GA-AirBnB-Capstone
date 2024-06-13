# GA-AirBnB-Capstone
by David Kanevsky
david@davidkanevsky.com
General Assembly, Data Science Immersive
June 2024

### Problem Statement
Using natural language processing, the goal is to identify poorly rated AirBnB listings in the Washington, D.C. area and the specific terms that people use to describe poorly rated listings, with the hope that identifying the words people use to describe a poorly reviewed listing can help hosts understand how to improve the guest experience and improve their rating.

### Hypothesis and Key Objectives

Since most AirBnB listings are rated positively, my hypothesis is that reviewers of poorly rated listings will be offering comments about why their listing is not rated highly, especially compared to other listings. Even if a reviewer does not compare it directly to other listings, the words and phrases they use to describe the listing should be dramatically different than the words/phrases used to describe positively rated listings that an NLP model using a TFIDF Vectorizer to train a model on words/phrases that appear more often in poorly rated listings than in well-rated listings. Once such a model is built (looking at maximizing balanced accuracy and recall), the features that are used to build that model can help identify why these listings are poorly rated, which would provide guidance to AirBnB and the hosts on how to improve their listing's rating and reviews, and increase their revenue.

#### Detailed Project Overview

Data was acquired from [Insider AirBnB](https://insideairbnb.com/get-the-data/) for the Washington, D.C. metro on June 7, 2024.

The workbooks should be read in the following order:

1. 01_EDA.ipynb

This workbook read in the data on the listings.csv data set, and did some initial exploration and cleaning, like removing null data on the review scores, and looking at the distribution and relationship of the various review scores. The various review categories (rated on a five point scale from 1 to 5, with 1 being poor and 5 being great) around overall rating, accuracy, cleanliness, check-in, communication, location and value were combined into a single review category based on the mean average of the other categories for each listing. Because the mean (4.79) and median (4.86) average rating was so high, to identify poorly rated listings, 2 categories were created: one where the listing has an average score below 4 (which has just 56 listings) and one where the listing has an average score below the 5th percentile (172 listings where the average rating is 4.43 or lower). The listings with the average rating and the categories the listing fell into was then exported as review_scores.csv.

2. 02_NL_Cleaning_Init_Model.ipynb

This workbook read in the reviews.csv data set. Since the reviews just have the comments each person left, but not the rating they gave the listing, the average rating and the categories it belongs (above or below a 4 rating, and above/below the 5th percentile) was appended onto the review. Reviews that did not have any comments were removed. Some additional analysis showed that the class imbalance on the number of listings above or below 4 (56 listings with an average rating below 4 and 3,747 listings with an average rating above 4) ballooned when the number of reviews were added on (178 reviews for listings with an average rating below 4 and 322,629 reviews for listings with an average rating above 4). Even when we changed the categories to being above or below the 5th percentile, there remained a heavy class imbalance, as there were 172 listings with 2,033 reviews for those below the 5th percentile compared to 3,631 listings and 320,774 reviews for those above the 5th percentile.

The comments were cleaned by removing punctuation and making everything lower case to prepare to run through a TFIDF Vectorizer, resulting in 78,280 unique words/features in the cleaned comments.

An initial Logistic Regression model was built based on those with a review score above or below 4 as the dependent variable. The model grid searched over a pipeline to determine the best parameters to determine the best score. However, while this model had good accuracy because it predicted the majority class, since the classes were imbalanced it had a recall score and f1 score of 0 as the model failed to predict the minority class.

The cleaned reviews and comments were saved as reviews_with_ratings_cleaned.csv so that I would not need to re-clean the data when running additional models in other notebooks.

3. 03_Model_Balanced_Class

This workbook read in the reviews_with_ratings_cleaned.csv. This also grid searched over multiple parameters with a TFIDF Vectorizer and a logistic regression looking at whether the review score was above or below 4, but this time set the class weight to be balanced. This improved the recall score to 0.111 and the f1 score to 0.009, but the model was still predicting too many false negatives towards the majority class.

4. 04_Logr_Model_Below_5th_Percentile

This model is similar to the 03_Model_Balanced_Class workbook, but this time instead of predicting whether the review was for a listing with a rating above or below an average rating of 4, it looked to predict whether the rating was above or below the 5th percentile (4.43). This model also grid searched over multiple parameters using a TFIDF Vectorizer and Logistic Regression with class weights being balanced. The best model had a recall score of 0.343 and an f1 score of 0.589, as even changing the dependent variable to expand the definition of the minority class still resulted in too many false positives.

5. 05_Bayes_Model_Below_5th_Percentile

This model is similar to the 04_Logr_Model_Below_5th_Percentile, but instead of using a Logistic Regression, it uses Multinomial Naive Bayes. However, this model produces a recall and f1 score of 0 as it never predicts the minority class.

6. 06_LogR_CVEC_Model_Below_5th_Percentile

This model is similar to 04 _Logr_Model_Below_5th_Percentile, but instead of using a TFIDF Vectorizer, it used a Count Vectorizer to see if that could improve model performance. This model had a recall score of 0.241 and an f1 score of 0.047.

7. 07_Oversample_Model

This model is similar to the 04_Logr_Model_Below_5th_Percentile, but this time used Random Oversampling in the hopes that by balancing the classes, it could improve the model's recall and f1 score. While the recall score improved to 0.467, it still had a poor f1 score of 0.048.

8. 08_Add_Stopwords

Looking at the predicted outputs from previous models, one of the features that kept repeatedly showing up was the word "jasmine." Especially since the Random Over Sampler could multiply a document 100 times in order to make the minority class equal to the majority class, the concern was the model was learning too much about the names of a few hosts that were having the comments duplicated multiple times. This time I added the term 'jasmine' into the stop words, along with other negative terms that were showing up as features in some models like 'poor', 'bad', 'terrible', 'worst', 'didn' and 'wasn.' While those negative words could improve the model's performance, what I was looking for was *why* someone thought a listing was poor or terrible.

I grid searched over a TFIDF Vectorizer and Logistic Regression with balanced class weights, with the best performing model having a recall score of 0.252 and an f1 score of 0.052. Interestingly, even though I gave the model option to use either just the traditional English stop words or the English stop words and the added stop words, the best performing version of this model used just the English stop words.

9. 09_Balance_Oversampling

Because the classes are so imbalanced, using random oversampling was duplicating the minority class too frequently and overfitting by learning about the names of a few hosts. To try to overcome that, I ran a grid search that incorporated a Random Over Sampler, along with the TFIDF Vectorizer and Logistic Regression. The parameters on the Random Over Sampler had the option of having the minority class (being below the 5th percentile in the average review rating) of being 1/10th or 1/4th of the size of the majority class. That would still oversample the minority class, but instead of oversampling it by over 150x times, it would oversample it by just 39 or 16 times respectively. 

The best performing version of this model had a recall score of 0.254 and an f1 score of 0.069. That model had the minority class at 1/4th the size of the majority class, but continued to use only English stop words, as opposed to the added stop words I gave it.

#### Data Dictionary

Because of GitHub's file size limitations, I am not able to upload the data files I worked with. The original data files come from [Insider Air BNB](https://insideairbnb.com/get-the-data/)) for the Washington, D.C. area and are dated as of March 23, 2024.

I used listings.csv.gz to get the detailed listings for all the properties and reviews.csv.gz to get the detailed reviews for all the properties in the Washington, D.C. area. I created 2 data sets myself based on those original data sets.

There are 3,803 unique listings with ratings in the review_scores.csv data set. The values for rating, accuracy, cleanliness, check-in, communication, location, value and avg_rating are reported as decimals that have a minimum value of 1, representing poor, and a maximum value of 5, representing great. The data dictionary for the review_scores.csv is as follows:

|Feature|Type|Description|
|---|---|---|
|id|int65|The unique ID for each listing, from the original AirBnB listings data set|
|rating|float|The summary rating for the listing, from the original AirBnB listings data set|
|accuracy|float|The summary accuracy rating for the listing, from the original AirBnB listings data set|
|cleanliness|float|The summary cleanliness rating for the listing, from the original AirBnB listings data set|
|checkin|float|The summary check-in rating for the listing, from the original AirBnB listings data set|
|communication|float|The summary communication rating for the listing, from the original AirBnB listings data set|
|location|float|The summary location rating for the listing, from the original AirBnB listings data set|
|value|float|The summary value rating for the listing, from the original AirBnB listings data set|
|avg_rating|float|The mean average rating based on each listing's rating, accuracy, cleanliness, check-in, communication, location and value|
|avg_score_below_4|int64|A binarized category indicating whether that listing's average rating is below 4.0 or greater than or equal to 4.0|
|avg_score_below_5th_percentile|int64|A binarized category indicating whether that listing's average rating is below the 5th percentile (approximately 4.429) or not |
|number_of_reviews|int64|The number of reviews for each listing, from the original AirBnB listings data set|

There are 322,807 unique reviews in the reviews_with_ratings_cleaned.csv data set. Please note that the avg_rating is NOT the rating that the reviewer gave the listing, but the listing's average rating across all reviews, as no individual ratings or scores were provided with the reviews. The data dictionary for the reviews_with_ratings_cleaned.csv is as follows:

|Feature|Type|Description|
|---|---|---|
|listing_id|int64|The unique ID for each listing, from the original AirBnB data set|
|id|int64|The unique ID for each review|
|date|object|The date of the review|
|reviewer_id|int64|The unique ID for each reviewer|
|reviewer_name|object|The name of each reviewer|
|comments|object|The original comment left by the reviewer, from the original AirBnB data set|
|avg_rating|float64|The mean average rating based on each listing's rating, accuracy, cleanliness, check-in, communication, location and value (same as on  review_scores.csv)|
|avg_score_below_4|int64|A binarized category indicating whether that listing's average rating is below 4.0 or greater than or equal to 4.0 (same as on review_scores.csv)|
|avg_score_below_5th_percentile|int64|A binarized category indicating whether that listing's average rating is below the 5th percentile (approximately 4.429) or not (same as on review_scores.csv)|
|comments_clean|object|The cleaned review comments with punctuation removed and all text made lower case|

#### Executive Summary

The issue with imbalanced classes made it difficult to produce a model that had a good recall score and balanced accuracy. No model had a balanced accuracy score of above 0.07 and the best performing recall score was just 0.467. Models had an issue with false positives, due to over-learning the names of hosts/listings, particularly when over-sampling the minority class of poorly reviewed listings. 

Some of this may be due to the nature of the data and approach. The rating was for the listing's average rating across ALL reviews, not the rating each reviewer gave for their stay. Additionally, because the ratings were made into a binary classification by drawing a semi-arbitrary line to define "poorly" reviewed listings versus those that aren't, some of the listings that are in majority class of positively rated listings but have lower ratings (for example, those between the bottom 5th and bottom 25th percentile) may use some of the words/features that also appear in the minority class, thereby reducing its importance in a TFIDF Vectorizer.

#### Recommendations and Future Considerations

With additional time, I would look at SMOTE to synthesize minority class of poor reviews to help with the class imbalance, as well as looking at lemmatizing/stemming words (to make "safety" and "unsafe" both be "safe") and the spaCy library to develop context around words.

I was reluctant to do under sampling and get rid of so much data from positively rated listings. But the best solution may be to look at just look at listings with under 100 reviews. Since all the poorly rated listings have under 100 stays, looking at poorly rated listings with under 100 stays and well rated listings with under 100 stays may be a better comparison.