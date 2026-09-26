# Loan Default Prediction

**Author:** Neelum Bashir
**Project:** Capstone, MIT Applied Data Science with AI program
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Business problem
Home equity loan defaults can cause large financial losses for lenders. The goal of this project is to predict which loan applicants are likely to default, so credit teams can identify high-risk applicants at the time of application.

## Data
The HMEQ (Home Equity) dataset contains 5,960 loans with 12 features, including loan amount, property value, job type, years at job, delinquency history, credit line age, and debt-to-income ratio. About 20% of applicants defaulted.

## Approach
- Data cleaning: handled missing values (up to 21% in debt-to-income ratio) and created missing-value flags, since missingness itself turned out to be predictive
- Exploratory analysis of default rates by job type, loan reason, and financial features
- Built and compared Logistic Regression, Decision Tree, and Random Forest models
- Tuned models with GridSearchCV and RandomizedSearchCV
- Chose **recall on defaulters** as the key metric, since missing a defaulter costs more than a false alarm

## Results
The **tuned Random Forest (RandomizedSearchCV)** was the best model, correctly identifying **78% of defaulters on the test set** without overfitting. Logistic Regression identified only 4% of defaulters.

## Key insights
- A missing debt-to-income ratio was the strongest warning sign of default, followed by the ratio itself
- Past delinquencies and derogatory reports strongly increase default risk
- Applicants in Sales (35%) and Self-employed (30%) roles had the highest default rates
- Older credit history and longer job tenure are linked to lower risk

## Recommendations
- Require complete debt-to-income and employment information before assessing applications
- Apply stricter review to applicants with delinquencies or derogatory reports
- Strengthen employment verification, especially for short or missing job history

## Notebook
See [Loan_Default_Prediction.ipynb](Loan_Default_Prediction.ipynb) for the full code, charts, and analysis.
