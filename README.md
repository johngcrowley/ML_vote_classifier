# Data Scientist takehome

## The Biggest Challenges:

- look through "ML Problem_Set" to see all visualizaitons / EDA / detailed walkthrough
- one voter to many rows. 
- had to make use of .drop_duplicates(subset='myv_van_id')

## Packages
- utilized numpy, pandas, seaborn, and scikitlearn for my ML classifier, and relied on the Jupyter Notebook for quick viz
- for the data-cleaning of part 1, I only used pandas and regex
- 
## Part-Two.py:
- I was very glad with my classifier. Training hit an AUC of .87 with a Gradient Boosted Decision Tree.
- I chose this clf because it's robust to over-fitting, and with the time limit (not having hours to tinker with GridSearch and other algorithims) I opted for GBDT because it's the best 'out-of-the-box' in this case. 
- In my visualizaiton, i noticed that the numerical variables in bivariate relationships were mostly linear, even when i played with taking the log or the sqrt of their values. had some gemoetric relationships arisen from these mods, I was considering an RBF kernel SVM. 
    
## Thought-Process (from Jupyter Notebook):
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

