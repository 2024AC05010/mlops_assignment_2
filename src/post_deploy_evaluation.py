import requests
import json
from sklearn.metrics import accuracy_score, classification_report

tApi_Url = "http://localhost:8000/predict"

tTest_Samples = [
    {"image_path": "data/sample/cat_sample.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_1.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_2.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_3.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_4.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_5.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_6.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_7.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_8.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_9.jpg", "true_label": "cat"},
    {"image_path": "data/sample/cat_sample_10.jpg", "true_label": "cat"},
    {"image_path": "data/sample/dog_sample.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_1.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_2.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_3.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_4.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_5.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_6.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_7.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_8.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_9.jpg", "true_label": "dog"},
    {"image_path": "data/sample/dog_sample_10.jpg", "true_label": "dog"},
]

if __name__ == "__main__":
    tTrue_Labels = []
    tPredicted_Labels = []

    print(f"\nStarting post-deployment evaluation with {len(tTest_Samples)} samples...")
    print(f"API Endpoint: {tApi_Url}")
    print(f"{'='*60}")

    for tIdx, tSample in enumerate(tTest_Samples, 1):
        try:
            with open(tSample["image_path"], "rb") as tF:
                tResponse = requests.post(tApi_Url, files={"pFile": tF})

            if tResponse.status_code == 200:
                tPred = tResponse.json()["prediction"]
                tPredicted_Labels.append(tPred)
                tTrue_Labels.append(tSample["true_label"])
                tStatus = "PASS" if tPred == tSample["true_label"] else "FAIL"
                print(f"[{tIdx}/{len(tTest_Samples)}] {tStatus} {tSample['image_path']} | True: {tSample['true_label']:4s} | Predicted: {tPred:4s}")
            else:
                print(f"[{tIdx}/{len(tTest_Samples)}] ERROR: {tSample['image_path']} | Status: {tResponse.status_code}")
        except Exception as tErr:
            print(f"[{tIdx}/{len(tTest_Samples)}] ERROR: {tSample['image_path']} | {str(tErr)}")

    if len(tTrue_Labels) > 0:
        tAccuracy = accuracy_score(tTrue_Labels, tPredicted_Labels)
        print(f"\n{'='*60}")
        print(f"POST-DEPLOYMENT MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples evaluated: {len(tTrue_Labels)}")
        print(f"Correct predictions: {sum(tT == tP for tT, tP in zip(tTrue_Labels, tPredicted_Labels))}")
        print(f"Post-deployment accuracy: {tAccuracy:.2%}")
        print(f"\nClassification Report:")
        print(classification_report(tTrue_Labels, tPredicted_Labels))
        print(f"{'='*60}")

        tResults = {
            "post_deployment_accuracy": round(tAccuracy, 4),
            "total_samples_evaluated": len(tTrue_Labels),
            "correct_predictions": sum(tT == tP for tT, tP in zip(tTrue_Labels, tPredicted_Labels)),
            "incorrect_predictions": sum(tT != tP for tT, tP in zip(tTrue_Labels, tPredicted_Labels)),
            "cat_samples": tTrue_Labels.count("cat"),
            "dog_samples": tTrue_Labels.count("dog"),
            "evaluation_timestamp": "2026-08-24"
        }
        
        import os
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/post_deploy_metrics.json", "w") as tF:
            json.dump(tResults, tF, indent=2)
        
        print(f"\nDetailed results saved to: artifacts/post_deploy_metrics.json")
        print(f"Sample distribution: {tResults['cat_samples']} cats, {tResults['dog_samples']} dogs")
