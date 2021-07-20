# Data Scientist TDP takehome screening

## The Biggest Challenges:
1. #### part_one
- time-crunch.
- apartment #'s buried in text.

2. #### part-two
- look through "ML Problem_Set" to see all visualizaitons / EDA / detailed walkthrough
- one voter to many rows. 
- had to make use of .drop_duplicates(subset='myv_van_id')

## Packages
- utilized numpy, pandas, seaborn, and scikitlearn for my ML classifier, and relied on the Jupyter Notebook for quick viz
- for the data-cleaning of part 1, I only used pandas and regex

## Overall: 
- I solved the Machine Learning option, both part 1 and part 2. 
- Due to time constraint, I decided to solve part-2 with the majority of my time.
- Therefore, I was unable to fully extract all matches from part 1, specifically I did not have time to parse our apartment #'s. If I had the time, I would have played with my regular expression to account for #'s within the 'Words' string in which addresses and city are often found.


## Part-One:
- With only 45 minutes left, I did not hit every match. I missed 35 out of 137 without Apartment # accounted for.
- my logic worked quickly and well on what it did hit on without much manipulation / post-processing.
- my Jupyter Notebook file 'part_one.ipynb' showcases a lot of my thinking.
- I would have utilizied "cities.csv" in a case where I needed to prioritize certain matches. I would have used a groupby or a boolean mask to determine which matter the most based on population. 

## Part-Two:
- most of my logic, work, and crucial data visualizaiton are in my Jupyter Notebook file "part_two.ipynb"
- I was very glad with my classifier. Training hit an AUC of .87 with a Gradient Boosted Decision Tree.
- I chose this clf because it's robust to over-fitting, and with the time limit (not having hours to tinker with GridSearch and other algorithims) I opted for GBDT because it's the best 'out-of-the-box' in this case. 
- In my visualizaiton, i noticed that the numerical variables in bivariate relationships were mostly linear, even when i played with taking the log or the sqrt of their values. had some gemoetric relationships arisen from these mods, I was considering an RBF kernel SVM. 
    
    
## Part_Two Thought-Process (from Jupyter Notebook):
1. Join voter_info and survey_respones, keep voter_history separate.
    - use 2 merged df's to generate some numeric variables.
    - initial take: mostlly categorical.
    - check for missing values
2. "master_survey_respone_name" --> try to group it / make useful
    - lot of messiness here / redundancy
3. feature-engineering
    - I.) i want to see a numerical abs(date_registered - election day 2018
        - get an average amount of days from OG registration to Election Day 2018.
        - this at least will correlate to the target variable (did they vote?) 
    - II.) voting history (% of elections voted in)
        - could be cool
4. see relationships with seaborn
    - use groupby with sns.heatmap / pivot_table to check.
    - to see how locations play in with gender, party, etc. later on...
    - precints, districts, etc. will have redundancy. 
    - what is a good balance of categorical variable but not be too sparse?
5. one-hot-encode categoricals (this is why we are trying to limit features AMAP)
6. Quick n dirty Random Forest to verify important features
7. GridsearchCV for best params across Logreg, GradboostDT, RandoForest
    - why? 
    - logreg is an important baseline. we are doing basic classification. always good to start small. 
    - Gradboostdt is very robust against over-fitting, and is a great "out of the box" classifier
    - random forest, tho higher computational cost, like gradboost, does deliver us feature_importances_

