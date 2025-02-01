#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import neptune
import os
import re
import pandas as pd

PROJECT_NAME = "kofmanya/HSE-MDS-Kofman-Anna-Diploma"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MmUyOTZjYy1mNzFjLTQ4YzUtYjk4Yi1hZmIxMTk5OWYwMDgifQ==" 

def promote_best_model_to_prod():
    project = neptune.init_project(
        project=PROJECT_NAME,
        api_token=API_TOKEN
    )
    runs_df = project.fetch_runs_table().to_pandas()
    eval_candidates = []

    def parse_model_name_from_test_col(column_name):
        match = re.match(r"^test/(.+)_accuracy$", column_name)
        return match.group(1) if match else None

    for _, row in runs_df.iterrows():
        run_id = row["sys/id"]
        for col in runs_df.columns:
            val = row[col]
            if (
                isinstance(val, (int, float))  
                and not pd.isna(val)          
                and col.startswith("test/")
                and col.endswith("_accuracy")
            ):
                model_name = parse_model_name_from_test_col(col)
                if model_name:
                    accuracy_val = float(val)
                    eval_candidates.append((run_id, model_name, accuracy_val))

    if not eval_candidates:
        print("No runs found that contain a 'test/<model>_accuracy' metric. Exiting.")
        return

    best_eval_run_id, best_model_name, best_accuracy = max(eval_candidates, key=lambda x: x[2])
    print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

    # Define directories
    model_dir = "models"
    data_dir = "data"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    def find_and_download_artifact(artifact_path, local_filepath):
        for _, row2 in runs_df.iterrows():
            r2_id = row2["sys/id"]
            with neptune.init_run(
                project=PROJECT_NAME,
                with_id=r2_id,
                api_token=API_TOKEN,
                mode="read-only"
            ) as run_check:
                if run_check.exists(artifact_path):
                    print(f"Found '{artifact_path}' in run {r2_id}, downloading -> {local_filepath}")
                    run_check[artifact_path].download(local_filepath)
                    return True
        return False

    # Define file paths
    weights_local = os.path.join(model_dir, f"{best_model_name}.pth")
    embeddings_local = os.path.join(data_dir, f"{best_model_name}_embeddings.npz")
    transformations_local = os.path.join(data_dir, f"{best_model_name}_transformations.json")

    # Download artifacts
    find_and_download_artifact(f"model/{best_model_name}_weights", weights_local)
    find_and_download_artifact(f"embeddings/{best_model_name}/path", embeddings_local)
    find_and_download_artifact(f"transformations/{best_model_name}/path", transformations_local)

    print("Best model files downloaded successfully!")

if __name__ == "__main__":
    promote_best_model_to_prod()

