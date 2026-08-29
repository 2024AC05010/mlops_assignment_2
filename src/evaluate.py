import torch
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score
import mlflow
from src.model import CatDogCNN
from src.data_preprocessing import get_dataloaders


tDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model():
    _, _, tTest_Loader = get_dataloaders(32)

    tModel = CatDogCNN().to(tDevice)
    tModel.load_state_dict(torch.load("models/cat_dog_model.pt", map_location=tDevice))
    tModel.eval()

    tTrue_Labels = []
    tPred_Labels = []

    with torch.no_grad():
        for tImages, tLabels in tTest_Loader:
            tImages = tImages.to(tDevice)
            tOutputs = tModel(tImages)
            tPreds = (tOutputs > 0.5).int().squeeze().tolist()
            if isinstance(tPreds, int):
                tPreds = [tPreds]
            tTrue_Labels.extend(tLabels.tolist())
            tPred_Labels.extend(tPreds)

    # confusion matrix
    tCm = confusion_matrix(tTrue_Labels, tPred_Labels)
    tDisp = ConfusionMatrixDisplay(tCm, display_labels=["Cat", "Dog"])
    tDisp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Cats vs Dogs CNN")
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()

    print(classification_report(tTrue_Labels, tPred_Labels, target_names=["Cat", "Dog"]))

    # Calculate additional metrics
    tPrecision = precision_score(tTrue_Labels, tPred_Labels)
    tRecall = recall_score(tTrue_Labels, tPred_Labels)
    tF1 = f1_score(tTrue_Labels, tPred_Labels)
    tAccuracy = (tCm[0,0] + tCm[1,1]) / tCm.sum()

    # Log evaluation metrics to MLflow
    try:
        mlflow.set_experiment("cats-vs-dogs-classification")
        with mlflow.start_run(run_name="evaluation_metrics"):
            mlflow.log_metrics({
                "eval_precision": tPrecision,
                "eval_recall": tRecall,
                "eval_f1_score": tF1,
                "eval_accuracy": tAccuracy
            })
            
            # Log confusion matrix as artifact
            mlflow.log_artifact("artifacts/confusion_matrix.png")
            mlflow.set_tag("evaluation_type", "post_training")
            
            print(f"Evaluation metrics logged to MLflow")
            print(f"  Precision: {tPrecision:.4f}")
            print(f"  Recall: {tRecall:.4f}")
            print(f"  F1 Score: {tF1:.4f}")
    except Exception as tErr:
        print(f"Note: MLflow logging skipped: {str(tErr)}")

    # loss curves
    if os.path.exists("artifacts/losses.pkl"):
        with open("artifacts/losses.pkl", "rb") as tF:
            tLoss_Data = pickle.load(tF)
        plt.figure(figsize=(8, 5))
        plt.plot(tLoss_Data["train"], label="Train Loss")
        plt.plot(tLoss_Data["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("artifacts/loss_curves.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("Loss curves saved")


if __name__ == "__main__":
    evaluate_model()
