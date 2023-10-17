import json

twhinbert_output_path = "ENSEMBLE/twhinbert_output_꼬링크.jsonl"
t5_output_path = "ENSEMBLE/t5_output_epoch10.jsonl"
kcelectra_output_path = "ENSEMBLE/kcelectra_output_펄기아.jsonl"

model_weights = {
    "twhinbert": 0.4,
    "t5": 0.2,
    "kcelectra": 0.4
}

emotion_thresholds = {
    "joy": 0.5,
    "anticipation": 0.5,
    "trust": 0.5,
    "surprise": 0.5,
    "disgust": 0.5,
    "fear": 0.5,
    "anger": 0.5,
    "sadness": 0.5
}

ensemble_output = {
    "joy": "True",
    "anticipation": "True",
    "trust": "True",
    "surprise": "True",
    "disgust": "True",
    "fear": "True",
    "anger": "True",
    "sadness": "True"
}

emotion_lists = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

ensemble_output_file_path = "ENSEMBLE/ensemble_output_442_postprocessed.jsonl"


def do_ensemble(get_label=True, post_process=False):
    with open(twhinbert_output_path, 'r', encoding='utf-8') as twhinbert_file, \
            open(t5_output_path, 'r', encoding='utf-8') as t5_file, \
            open(kcelectra_output_path, 'r', encoding='utf-8') as kcelectra_file, \
            open(ensemble_output_file_path, 'w', encoding='utf-8') as ensemble_output_file:

        print("Reading files...")
        twhinbert_lines = twhinbert_file.readlines()
        t5_lines = t5_file.readlines()
        kcelectra_lines = kcelectra_file.readlines()

        print("Processing files...")
        for idx, line in enumerate(twhinbert_lines):
            twhinbert_entry = json.loads(line)
            t5_entry = json.loads(t5_lines[idx])
            kcelectra_entry = json.loads(kcelectra_lines[idx])

            all_false = True
            min_gap = float('inf')
            min_gap_emotion = None

            for emotion in emotion_lists:
                twhinbert_probs = twhinbert_entry["output"][emotion][1]
                t5_probs = t5_entry["output"][emotion][1]
                kcelectra_probs = kcelectra_entry["output"][emotion][1]

                ensemble_probs = model_weights["twhinbert"] * twhinbert_probs + \
                                model_weights["t5"] * t5_probs + \
                                model_weights["kcelectra"] * kcelectra_probs

                if get_label:
                    if ensemble_probs > emotion_thresholds[emotion]:
                        ensemble_output[emotion] = "True"
                        all_false = False
                    else:
                        ensemble_output[emotion] = "False"

                    gap = abs(ensemble_probs - emotion_thresholds[emotion])
                    if gap < min_gap:
                        min_gap = gap
                        min_gap_emotion = emotion

                else:
                    ensemble_output[emotion] = ensemble_probs

            if post_process and all_false and min_gap_emotion:
                ensemble_output[min_gap_emotion] = "True"
            
            ensemble_entry = {
                "id": twhinbert_entry["id"],
                "input": twhinbert_entry["input"],
                "output": ensemble_output
            }

            print(f'{idx + 1} / 4738')
            ensemble_output_file.write(json.dumps(ensemble_entry, ensure_ascii=False) + "\n")
    print("Done!")

do_ensemble(get_label=False, post_process=False)

def get_threshold():
    pass

# do_ensemble(get_label=True, post_process=True)
