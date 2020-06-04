# LOAN REPAYMENT PREDICION

### AIMS

Find the best classification model to predict wether a loan will be paid off or defaulted

### DATA

Details of 346 persons whose loan are already paid off or defaulted. It includes following fields

| FIELD          | DESCRIPTION                                                                           |
|:---------------|:--------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

### ACHIEVEMENTS

* Data visualization and processing
* Feature selection and extraction
* Converting categorical features to numerical values (One Hot vectors)
* Data normalization
* Training 4 different classification models (KNN, Decision Tree, SVM, Logistic Regression) on training data
* Comparing models with computed metrics (Jaccard, F1 score, LogLoss)

### RESULTS

| Algorithm          |  Jaccard | F1-score |  LogLoss |
|:-------------------|:--------:|:--------:|:--------:|
| KNN                | 0.740741 | 0.714431 |    NA    |
| Decision Tree      | 0.759259 | 0.761886 |    NA    |
| SVM                | 0.759259 | 0.695923 |    NA    |
| LogisticRegression | 0.777778 | 0.708937 | 0.473958 |

#

###### 2020