import Classifiers as cl

# Accuracy_scores holds validation scores from different models
accuracy_scores = {
    "Random Forest Baseline": cl.sklearn_random_forest(),
    "Our MNB validation score": cl.validate(),
    "Sklearn MNB validation score": cl.sklearn_MNB_validate(),
    "Logistic Regression validation score": cl.sklearn_logistic_validate(),
    "Support Vector Classifier validation score": cl.sklearn_svc_validate()
}

# Outputs accuracy scores
for key, value in accuracy_scores.items():
    print(f'{key}: {value}')
