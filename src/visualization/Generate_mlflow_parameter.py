import mlflow
import pandas as pd

def generate_report():
    # Connect to MLflow
    mlflow.set_tracking_uri("/workspaces/NYC_taxi_trip/mlruns")
    
    # Fetch MLflow experiments and runs
    experiments = mlflow.list_experiments()
    runs = [run for exp in experiments for run in mlflow.search_runs(experiment_ids=[exp.experiment_id])]
    
    # Extract relevant information
    metrics = pd.DataFrame([(run.run_id, run.data.metrics) for run in runs], columns=['Run ID', 'Metrics'])
    parameters = pd.DataFrame([(run.run_id, run.data.params) for run in runs], columns=['Run ID', 'Parameters'])
    
    # Generate report (example: save as CSV)
    metrics.to_csv('metrics.csv', index=False)
    parameters.to_csv('parameters.csv', index=False)

if __name__ == "__main__":
    generate_report()