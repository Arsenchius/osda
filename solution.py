import os
import hydra
from omegaconf import DictConfig
import pandas as pd

from training import training_tuning


# Function to create a classification result table
def create_classification_result_table(results):
    data = {"Classifier": [], "Accuracy": [], "F1-Score": []}
    for clf_name, metrics in results.items():
        data["Classifier"].append(clf_name)
        data["Accuracy"].append(metrics["accuracy"])
        data["F1-Score"].append(metrics["f1_score"])
    result_df = pd.DataFrame(data)
    return result_df


# Function to write classification results to a text file
def write_classification_results_to_file(results, filename):
    result_df = create_classification_result_table(results)
    result_df.to_csv(filename, index=False, sep="\t")


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig):
    exp_folder_name = "exp_" + str(cfg.exps.id)
    exp_path = os.path.join(cfg.reports_path, exp_folder_name)
    path_to_parameters = os.path.join(exp_path, cfg.exps.path_to_parameters)
    path_to_report = os.path.join(exp_path, cfg.exps.path_to_report)
    results = training_tuning(
        data_path=cfg.exps.data_path,
        number_of_trials=cfg.exps.number_of_trials_for_tuning,
        path_to_parameters=path_to_parameters,
        features=cfg.exps.features,
        target=cfg.exps.target,
    )


    # Print the results
    for clf_name, metrics in results.items():
        print(
            f'{clf_name} - Accuracy: {metrics["accuracy"]:.4f}, F1-Score: {metrics["f1_score"]:.4f}'
        )

    # Write the results to a text file
    write_classification_results_to_file(results, path_to_report)

if __name__ == "__main__":
    run()
