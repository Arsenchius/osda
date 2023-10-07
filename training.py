import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna import trial


# Function to save best parameters to a YAML file
def save_best_parameters_to_yaml(best_params, filename):
    with open(filename, "w") as file:
        yaml.dump(best_params, file)


def training_tuning(data_path, number_of_trials, features, target, path_to_parameters):
    df = pd.read_csv(data_path)

    classifiers = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 1, 32),
            min_samples_split=trial.suggest_float("min_samples_split", 0.1, 1.0),
            min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.1, 0.5),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 1, 32),
            min_samples_split=trial.suggest_float("min_samples_split", 0.1, 1.0),
            min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.1, 0.5),
        ),
        # "XGBoost": XGBClassifier(
        #     n_estimators=trial.suggest_int("n_estimators", 50, 200),
        #     max_depth=trial.suggest_int("max_depth", 1, 32),
        #     learning_rate=trial.suggest_float("learning_rate", 0.001, 1.0),
        #     gamma=trial.suggest_float("gamma", 0.0, 1.0),
        #     tree_method="gpu_hist",
        # ),
        # "CatBoost": CatBoostClassifier(
        #     n_estimators=trial.suggest_int("n_estimators", 50, 200),
        #     max_depth=trial.suggest_int("max_depth", 1, 12),
        #     learning_rate=trial.suggest_float("learning_rate", 0.001, 1.0),
        #     task_type="GPU",
        # ),
        "KNN": KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 20),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        ),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(
            C=trial.suggest_float("C", 0.001, 10),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="liblinear",
        ),
    }

    def optimizing(trial, df, features, target):
        for clf_name, clf in classifiers.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(
                clf, df[features], df[target], cv=kf, scoring="accuracy"
            )
            f1_scores = cross_val_score(
                clf, df[features], df[target], cv=kf, scoring="f1_macro"
            )

            mean_accuracy = np.mean(accuracy_scores)
            mean_f1 = np.mean(f1_scores)

            trial.report(mean_accuracy, step=1)
            trial.report(mean_f1, step=2)

        return mean_f1

    study = optuna.create_study(direction="maximize")
    func = lambda trial: optimizing(trial, df, features, target)
    study.optimize(func, n_trials=number_of_trials)
    best_params = {}
    best_classifiers = {}

    for clf_name in classifiers.keys():
        best_trial = study.best_trial
        best_params[clf_name] = {
            param_name: best_trial.params[param_name]
            for param_name in classifiers[clf_name].get_params().keys()
        }
        # best_trial = study.best_trial_for_key(clf_name)
        # best_params[clf_name] = best_trial.params
        best_classifiers[clf_name] = classifiers[clf_name].set_params(
            **best_trial.params
        )
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    results = {}
    for clf_name, clf in best_classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results[clf_name] = {"accuracy": accuracy, "f1_score": f1}

    save_best_parameters_to_yaml(best_params, path_to_parameters)

    return results
