"""
MLflow Model Registry Management Script
Demonstrates model versioning, staging, and deployment tracking
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import json


def setup_model_registry():
    """
    Setup and configure MLflow model registry for cats_vs_dogs_classifier
    """
    print("="*60)
    print("MLflow Model Registry Setup")
    print("="*60)
    
    # Initialize MLflow client
    tClient = MlflowClient()
    
    # Model name
    tModel_Name = "cats_dogs_classifier"
    
    try:
        # Create registered model if it doesn't exist
        tClient.create_registered_model(
            name=tModel_Name,
            description="Binary image classifier for cats vs dogs - MLOps Assignment 2",
            tags={"framework": "pytorch", "task": "binary_classification"}
        )
        print(f"Created registered model: {tModel_Name}")
    except Exception as tErr:
        print(f"Model registry already exists: {tModel_Name}")
    
    return tClient, tModel_Name


def list_model_versions(tClient, tModel_Name):
    """
    List all versions of the registered model
    """
    print(f"\n{'='*60}")
    print(f"Available Model Versions: {tModel_Name}")
    print(f"{'='*60}")
    
    try:
        tVersions = tClient.get_latest_versions(tModel_Name)
        
        if not tVersions:
            print("No model versions found in registry")
            return []
        
        print(f"{'Version':<10} {'Stage':<15} {'Run ID':<20} {'Status'}")
        print("-" * 60)
        
        for tVersion in tVersions:
            tVersion_Num = tVersion.version
            tStage = tVersion.current_stage
            tRun_ID = tVersion.run_id
            tStatus = "Active" if tVersion.current_stage in ["staging", "production"] else "Archived"
            
            print(f"{tVersion_Num:<10} {tStage:<15} {tRun_ID:<20} {tStatus}")
        
        return tVersions
    except Exception as tErr:
        print(f"Error listing model versions: {str(tErr)}")
        return []


def promote_model_to_production(tClient, tModel_Name, tVersion_Num):
    """
    Promote a specific model version to production stage
    """
    print(f"\n{'='*60}")
    print(f"Promoting Model {tModel_Name} v{tVersion_Num} to Production")
    print(f"{'='*60}")
    
    try:
        # Archive current production model
        tCurrent_Production = tClient.get_latest_versions(
            tModel_Name, stages=["production"]
        )
        
        if tCurrent_Production:
            tOld_Version = tCurrent_Production[0].version
            tClient.transition_model_version_stage(
                name=tModel_Name,
                version=tOld_Version,
                stage="archived"
            )
            print(f"Archived previous production model: v{tOld_Version}")
        
        # Promote new version to production
        tClient.transition_model_version_stage(
            name=tModel_Name,
            version=tVersion_Num,
            stage="production"
        )
        print(f"Promoted model v{tVersion_Num} to production")
        
        # Add production tag
        tClient.set_model_version_tag(
            name=tModel_Name,
            version=tVersion_Num,
            key="deployment_status",
            value="production"
        )
        
        return True
    except Exception as tErr:
        print(f"Error promoting model: {str(tErr)}")
        return False


def get_production_model_info(tClient, tModel_Name):
    """
    Get information about the current production model
    """
    print(f"\n{'='*60}")
    print(f"Current Production Model: {tModel_Name}")
    print(f"{'='*60}")
    
    try:
        tProduction_Models = tClient.get_latest_versions(
            tModel_Name, stages=["production"]
        )
        
        if not tProduction_Models:
            print("No production model found")
            return None
        
        tModel_Version = tProduction_Models[0]
        tModel_Details = tClient.get_model_version(
            tModel_Name, tModel_Version.version
        )
        
        print(f"Version: {tModel_Details.version}")
        print(f"Stage: {tModel_Details.current_stage}")
        print(f"Run ID: {tModel_Details.run_id}")
        print(f"Creation Time: {tModel_Details.creation_timestamp}")
        print(f"Last Updated: {tModel_Details.last_updated_timestamp}")
        
        # Get run details
        tRun = tClient.get_run(tModel_Details.run_id)
        print(f"\nModel Performance Metrics:")
        for tKey, tValue in tRun.data.metrics.items():
            print(f"  {tKey}: {tValue:.4f}")
        
        return tModel_Details
    except Exception as tErr:
        print(f"Error getting production model info: {str(tErr)}")
        return None


def compare_model_versions(tClient, tModel_Name, tVersion1, tVersion2):
    """
    Compare two model versions based on their metrics
    """
    print(f"\n{'='*60}")
    print(f"Comparing Model Versions: v{tVersion1} vs v{tVersion2}")
    print(f"{'='*60}")
    
    try:
        tModel1 = tClient.get_model_version(tModel_Name, tVersion1)
        tModel2 = tClient.get_model_version(tModel_Name, tVersion2)
        
        tRun1 = tClient.get_run(tModel1.run_id)
        tRun2 = tClient.get_run(tModel2.run_id)
        
        print(f"\n{'Metric':<25} {'v'+tVersion1:<15} {'v'+tVersion2:<15} {'Difference'}")
        print("-" * 70)
        
        # Compare common metrics
        tCommon_Metrics = set(tRun1.data.metrics.keys()) & set(tRun2.data.metrics.keys())
        
        for tMetric in sorted(tCommon_Metrics):
            tVal1 = tRun1.data.metrics[tMetric]
            tVal2 = tRun2.data.metrics[tMetric]
            tDiff = tVal2 - tVal1
            tIndicator = "+" if tDiff > 0 else "-" if tDiff < 0 else "="
            
            print(f"{tMetric:<25} {tVal1:<15.4f} {tVal2:<15.4f} {tIndicator}{abs(tDiff):.4f}")
        
    except Exception as tErr:
        print(f"Error comparing model versions: {str(tErr)}")


def export_model_registry_info(tClient, tModel_Name):
    """
    Export model registry information to JSON for documentation
    """
    print(f"\n{'='*60}")
    print(f"Exporting Model Registry Information")
    print(f"{'='*60}")
    
    try:
        tVersions = tClient.get_latest_versions(tModel_Name)
        tRegistry_Info = {
            "model_name": tModel_Name,
            "total_versions": len(tVersions),
            "versions": []
        }
        
        for tVersion in tVersions:
            tRun = tClient.get_run(tVersion.run_id)
            tVersion_Info = {
                "version": tVersion.version,
                "stage": tVersion.current_stage,
                "run_id": tVersion.run_id,
                "creation_timestamp": tVersion.creation_timestamp,
                "metrics": tRun.data.metrics,
                "params": tRun.data.params,
                "tags": tVersion.tags
            }
            tRegistry_Info["versions"].append(tVersion_Info)
        
        # Save to file
        import os
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model_registry_info.json", "w") as tF:
            json.dump(tRegistry_Info, tF, indent=2)
        
        print(f"Model registry info exported to: artifacts/model_registry_info.json")
        return tRegistry_Info
        
    except Exception as tErr:
        print(f"Error exporting registry info: {str(tErr)}")
        return None


if __name__ == "__main__":
    # Set MLflow tracking URI (can be local or remote)
    mlflow.set_tracking_uri("file:///./mlruns")
    
    print("MLflow Model Registry Management Tool")
    print("="*60)
    
    # Setup model registry
    tClient, tModel_Name = setup_model_registry()
    
    # List available versions
    tVersions = list_model_versions(tClient, tModel_Name)
    
    if tVersions:
        # Get production model info
        get_production_model_info(tClient, tModel_Name)
        
        # Compare latest two versions if available
        if len(tVersions) >= 2:
            compare_model_versions(tClient, tModel_Name, 
                               tVersions[-1].version, 
                               tVersions[-2].version)
        
        # Export registry information
        export_model_registry_info(tClient, tModel_Name)
        
        print(f"\n{'='*60}")
        print("Model Registry Management Complete")
        print(f"{'='*60}")
        print(f"Use MLflow UI to view: mlflow ui")
        print(f"Or promote models programmatically using promote_model_to_production()")
    else:
        print("\nNo model versions found. Train a model first using train.py")
