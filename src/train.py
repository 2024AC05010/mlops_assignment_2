import torch
import json
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
from src.model import CatDogCNN
from src.data_preprocessing import get_dataloaders
import os
import pickle
import datetime

# hyperparams
tEpochs = 10
tLearning_Rate = 0.001
tBatch_Size = 32
tOptimizer_Name = "Adam"

tDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(pModel, pLoader, pCriterion, pOptimizer):
    pModel.train()
    tRunning_Loss = 0.0
    tCorrect = 0
    tTotal = 0

    for tImages, tLabels in pLoader:
        tImages = tImages.to(tDevice)
        tLabels = tLabels.float().unsqueeze(1).to(tDevice)

        pOptimizer.zero_grad()
        tOutputs = pModel(tImages)
        tLoss = pCriterion(tOutputs, tLabels)
        tLoss.backward()
        pOptimizer.step()

        tRunning_Loss += tLoss.item() * tImages.size(0)
        tPreds = (torch.sigmoid(tOutputs) > 0.5).float()
        tCorrect += (tPreds == tLabels).sum().item()
        tTotal += tLabels.size(0)

    tEpoch_Loss = tRunning_Loss / tTotal
    tEpoch_Acc = tCorrect / tTotal
    return tEpoch_Loss, tEpoch_Acc


def validate(pModel, pLoader, pCriterion):
    pModel.eval()
    tRunning_Loss = 0.0
    tCorrect = 0
    tTotal = 0

    with torch.no_grad():
        for tImages, tLabels in pLoader:
            tImages = tImages.to(tDevice)
            tLabels = tLabels.float().unsqueeze(1).to(tDevice)

            tOutputs = pModel(tImages)
            tLoss = pCriterion(tOutputs, tLabels)

            tRunning_Loss += tLoss.item() * tImages.size(0)
            tPreds = (torch.sigmoid(tOutputs) > 0.5).float()
            tCorrect += (tPreds == tLabels).sum().item()
            tTotal += tLabels.size(0)

    tEpoch_Loss = tRunning_Loss / tTotal
    tEpoch_Acc = tCorrect / tTotal
    return tEpoch_Loss, tEpoch_Acc


def main():
    tTrain_Loader, tVal_Loader, tTest_Loader = get_dataloaders(tBatch_Size)

    tModel = CatDogCNN().to(tDevice)
    tCriterion = torch.nn.BCEWithLogitsLoss()
    tOptimizer = torch.optim.Adam(tModel.parameters(), lr=tLearning_Rate)

    tTrain_Losses = []
    tVal_Losses = []

    # Set MLflow experiment with enhanced tracking
    mlflow.set_experiment("cats-vs-dogs-classification")
    
    # Generate unique run name with timestamp
    tRun_Name = f"cnn_v1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=tRun_Name):
        # Log hyperparameters with enhanced metadata
        mlflow.log_params({
            "epochs": tEpochs,
            "learning_rate": tLearning_Rate,
            "batch_size": tBatch_Size,
            "optimizer": tOptimizer_Name,
            "model_architecture": "SimpleCNN_4layer",
            "input_size": "224x224x3",
            "output_classes": "binary (cat/dog)",
            "device": str(tDevice)
        })

        # Log additional training metadata
        mlflow.set_tag("training_framework", "pytorch")
        mlflow.set_tag("task", "binary_image_classification")
        mlflow.set_tag("dataset", "cats_vs_dogs_kaggle")
        mlflow.set_tag("author", "mlops_assignment")

        for tEpoch in range(tEpochs):
            tTrain_Loss, tTrain_Acc = train_one_epoch(tModel, tTrain_Loader, tCriterion, tOptimizer)
            tVal_Loss, tVal_Acc = validate(tModel, tVal_Loader, tCriterion)

            tTrain_Losses.append(tTrain_Loss)
            tVal_Losses.append(tVal_Loss)

            mlflow.log_metric("train_loss", tTrain_Loss, step=tEpoch)
            mlflow.log_metric("val_loss", tVal_Loss, step=tEpoch)
            mlflow.log_metric("train_accuracy", tTrain_Acc, step=tEpoch)
            mlflow.log_metric("val_accuracy", tVal_Acc, step=tEpoch)

            print(f"Epoch {tEpoch+1}/{tEpochs} | Train Loss: {tTrain_Loss:.4f} Acc: {tTrain_Acc:.4f} | Val Loss: {tVal_Loss:.4f} Acc: {tVal_Acc:.4f}")

        # test eval
        tTest_Loss, tTest_Acc = validate(tModel, tTest_Loader, tCriterion)
        mlflow.log_metric("test_accuracy", tTest_Acc)
        mlflow.log_metric("test_loss", tTest_Loss)
        print(f"Test Accuracy: {tTest_Acc:.4f}")

        # save model locally
        os.makedirs("models", exist_ok=True)
        torch.save(tModel.state_dict(), "models/cat_dog_model.pt")
        
        # Log model artifact
        mlflow.log_artifact("models/cat_dog_model.pt", artifact_path="pytorch_models")
        
        # Create example input for signature inference
        tModel.eval()
        tExample_Input = torch.randn(1, 3, 224, 224).to(tDevice)
        with torch.no_grad():
            tExample_Output = tModel(tExample_Input)
        tSignature = infer_signature(
            tExample_Input.cpu().numpy(),
            tExample_Output.cpu().numpy()
        )
        
        # Log PyTorch model with enhanced metadata
        tModel_Info = mlflow.pytorch.log_model(
            pytorch_model=tModel,
            artifact_path="cats_dogs_cnn_model",
            signature=tSignature,
            input_example=tExample_Input.cpu().numpy(),
            registered_model_name="cats_dogs_classifier"
        )
        
        # Log model version metadata
        mlflow.set_tag("model_stage", "staging")
        
        # Log final metrics as model performance metrics
        mlflow.log_metrics({
            "final_train_accuracy": round(tTrain_Acc, 4),
            "final_val_accuracy": round(tVal_Acc, 4),
            "final_test_accuracy": round(tTest_Acc, 4),
            "final_train_loss": round(tTrain_Loss, 4),
            "final_val_loss": round(tVal_Loss, 4),
            "final_test_loss": round(tTest_Loss, 4)
        })

        # save metrics json
        tMetrics = {
            "train_accuracy": round(tTrain_Acc, 4),
            "val_accuracy": round(tVal_Acc, 4),
            "test_accuracy": round(tTest_Acc, 4),
            "train_loss": round(tTrain_Loss, 4),
            "val_loss": round(tVal_Loss, 4),
            "test_loss": round(tTest_Loss, 4),
            "epochs": tEpochs,
            "mlflow_run_id": mlflow.active_run().info.run_id
        }
        with open("metrics.json", "w") as tF:
            json.dump(tMetrics, tF, indent=2)
        mlflow.log_artifact("metrics.json")

        # save losses for plotting
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/losses.pkl", "wb") as tF:
            pickle.dump({"train": tTrain_Losses, "val": tVal_Losses}, tF)
        mlflow.log_artifact("artifacts/losses.pkl")
        
        print(f"\n{'='*60}")
        print(f"MLflow Model Registration Complete")
        print(f"{'='*60}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model Name: cats_dogs_classifier")
        print(f"Model Stage: production")
        print(f"Test Accuracy: {tTest_Acc:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()