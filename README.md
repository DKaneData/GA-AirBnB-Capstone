# GA-AirBnB-Capstone
by David Kanevsky
General Assembly, Data Science Immersive
June 2024


### Problem Statement
Using natural language processing, the goal is to identify poorly rated AirBnB listings in the Washington, D.C. area and the specific terms that people use to describe poorly rated listings, with the hope that identifying the words people use to describe a poorly reviewed listing can help hosts understand how to improve the guest experience and improve their rating.

#### Detailed Project Overview

Data was acquired from [Insider AirBnB](https://insideairbnb.com/get-the-data/) for the Washington, D.C. metro on June 7, 2024.

The workbooks should be read in the following order:

1. 01_EDA.ipynb

This workbook read in the data on the listings.csv data set, and did some initial exploration and cleaning, like removing null data on the review scores, and looking at the distribution and relationship of the various review scores. The various review categories (rated on a 5 point scale from 1 to 5, with 1 being poor and 5 being great) around overall rating, accuracy, cleanliness, check-in, communication, location and value were combined into a single review category based on the mean average of the other categories for each listing. Because the mean (4.79) and median (4.86) average rating was so high, to identify poorly rated listings, 2 categories were created: one where the listing has an average score below 4 (which has just 56 listings) and one where the listing has an average score below the 5th percentile (172 listings where the average rating is 4.43 or lower). The listings with the average rating and the categories the listing fell into was then exported as review_scores.csv.

2. 02_NL_Cleaning_Init_Model.ipynb

This workbook read in the reviews.csv data set. Since the reviews just have the comments each person left, but not the rating they gave the listing, the averaged rating and the categories it belongs (above or below a 4 rating, and above/below the 5th percentile) was appended onto the review. Reviews that did not have any comments were removed. Some additional analysis showed that the class imbalance on the number of listings above or below 4 (56 listings with an average rating below 4 and 3,747 listings with an average rating above 4) ballooned when the number of reviews were added on (178 reviews for listings with an average rating below 4 and 322,629 reviews for listings with an average rating above 4). Even when we changed the categories to being above or below the 5th percentile, there remained a heavy class imbalance, as there were 172 listings with 2,033 reviews for those below the 5th percentile compared to 3,631 listings and 320,774 reviews for those above the 5th percentile.

The comments were cleaned by removing punctuation and making everything lower case to prepare to run through a TFIDF Vectorizer, resulting in 78,280 unique words/features in the cleaned comments.

An initial Logistic Regression model was built based on those with a review score above or below 4 as the dependent variable and run through a pipeline to determine the best score. However, while this model had good accuracy because it predicted the majority class, because the classes were imbalanced it had a recall score and f1 score of 0 because it failed to predict the minority class.

The cleaned reviews and comments were saved as reviews_with_ratings_cleaned.csv so that I would not need to re-clean the data when running additional models in other notebooks.

3. 03_Model_Balanced_Class

4. 04_Logr_Model_Below_5th_Percentile

5. 05_Bayes_Model_Below_5th_Percentile

6. 06_LogR_CVEC_Model_Below_5th_Percentile

7. 07_Oversample_Model

8. 08_Add_Stopwords

9. 09_Balance_Oversampling