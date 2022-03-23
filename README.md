# credit risk discretization_ensemble

Source code for experiments replication for the research paper
"Ensamble de métodos de discretización no supervisados para riesgo de crédito"
(Unsupervised discretization methods ensemble for credit risk)

-RawDataCleansing.py: contains all necessary functions to transform data from
its raw form to a ready-to-model data structure

-Clean Data.ipynb: Notebook for applying the previous

-DiscreteCombination.py: All functions, classes and algorithms needed for
replicating the experiment.

-Article Results.ipynb: Notebook for summary tables replication.



Author: José Fuentes <jose.gustavo.fuentes@comunidad.unam.mx>



# credit risk datasets references

Australian: https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
Microfinance Loan Credit Scoring: https://www.kaggle.com/shahrukhkhan/microfinance-loan-credit-scoring
German: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
Give me some credit: https://www.kaggle.com/c/GiveMeSomeCredit
HMEQ: https://www.kaggle.com/ajay1735/hmeq-data
Japanese: https://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening
Lending club: https://www.kaggle.com/wordsforthewise/lending-club
Mortgage: https://github.com/JLZml/Credit-Scoring-Data-Sets/tree/master/6.%20Credit%20risk%20analysis/mortgage
Polish: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
Taiwan: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

All data must be put in a folder called raw, cleansed results will be dumped in a folder named clean in pandas pickle format. Both folders must be inside a folder named data. 
