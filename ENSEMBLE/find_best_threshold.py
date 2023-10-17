import json
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt

prediction_sheet_path = "ENSEMBLE/ensemble_output_442_prob.jsonl"
answer_sheet_path = "ENSEMBLE/test_hand_labeled.jsonl"

with open(prediction_sheet_path, 'r', encoding='utf8') as file:
    prediction_data = [json.loads(line) for line in file]

with open(answer_sheet_path, 'r', encoding='utf8') as file:
    answer_data = [json.loads(line) for line in file]

emotions = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

pr_data = {}

thresholds_path = "ENSEMBLE/emotion_thresholds.json"

for emotion in emotions:
    true_labels = [int(record['output'][emotion] == 'True') for record in answer_data]
    predicted_scores = [record['output'][emotion] for record in prediction_data]
    
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)
    pr_data[emotion] = {'precision': precision, 'recall': recall, 'thresholds': thresholds}

def show_precision_recall_plt():
    # Plotting the precision-recall curve for each emotion
    plt.figure(figsize=(14, 10))

    for emotion in emotions:
        plt.plot(pr_data[emotion]['recall'], pr_data[emotion]['precision'], label=emotion)

    plt.title('Precision-Recall Curve for Each Emotion')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_best_threshold(get_details=False, save_output=True):
    best_thresholds = {}
    for emotion in emotions:
        precision = pr_data[emotion]['precision'][:-1]  # excluding the last value which is 1
        recall = pr_data[emotion]['recall'][:-1]        # excluding the first value which is 0
        thresholds = pr_data[emotion]['thresholds']
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        # Find the threshold that gives the highest F1 score
        best_index = np.argmax(f1_scores)

        if get_details:
            best_thresholds[emotion] = {
                'threshold': thresholds[best_index],
                'f1_score': f1_scores[best_index]
            }

        else:
            best_thresholds[emotion] = thresholds[best_index]

    print(best_thresholds)

    if save_output:
        with open(thresholds_path, 'w', encoding='utf8') as file:
            json.dump(best_thresholds, file, ensure_ascii=False)

    return best_thresholds

# show_precision_recall_plt()
best_thresholds = find_best_threshold()
