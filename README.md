# Predict-Default-Loans

The objective of this project is to predict the loans that will be charged-off/default. The dataset is taken from Lending Club with 52 descriptive features with loans over a period of 5 years from 2007-2011.

### Dataset and Modelling

The dataset is imbalanced with fully paid(positive class) to charged off(negative class) ratio of 85:15. Three techniques are implemented to balance the data: Under-sampling, over-sampling and using weighted model.

Three algorithms are used to train the data: Random Forest, XGBoost and Neural network using Pytorch and CUDA. The XGBooost and Neural network are trained using GPU. The models are evaluated using AUC, F1 score and confusion matrix.

### Results

The best model for each of the technique are:

| Technique      | Algorithm      | AUC | F1 score                            | Confusion Matrix                  |
|----------------|----------------|-----|-------------------------------------|-----------------------------------|
| Undersampling  | XGBoost        |0.98 |Charged Off: 0.98<br>Fully paid: 0.98|TP:2156<br>FP:14<br>TN:365<br>FN:14|
| Oversampling   | Neural Network |0.99 |Charged Off: 0.97<br>Fully paid: 1.00|TP:2158<br>FP:12<br>TN:371<br>FN:8 |
| Weighted Model |XGBoost         |0.98 |Charged Off: 0.99<br>Fully paid: 0.99|TP:120<br>FP:1<br>TN:130<br>FN:5   |

### Libraries installed:

pandas, numpy, matplotlib, seaborn, chart_studio, sklearn, xgboost, torch, torchvision.

Using conda install orca to render static plots from plotly. Command to install: 

$ conda install -c plotly plotly-orca

