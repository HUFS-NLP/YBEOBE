import json
import os
from collections import Counter

# config 불러옴
with open("ensemble_settings.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def dialga_ensemble(model_name="디아루가_v5"):
    # outputs_backup\펼기아_with_probs.jsonl
    palkia_path = "outputs_backup/펼기아_with_probs.jsonl"
    charmander_path = "outputs_backup/파이리_with_probs.jsonl"

    model_weights = {"palkia": 0.5, "charmander": 0.5}
    emotion_thresholds = {"joy": 0.5, "anticipation": 0.5, "trust": 0.5, "surprise": 0.5, "disgust": 0.5, "fear": 0.5, "anger": 0.5, "sadness": 0.5}

    ensemble_output = {"joy": "True", "anticipation": "True", "trust": "True", "surprise": "True", "disgust": "True", "fear": "True", "anger": "True", "sadness": "True"}

    emotion_lists = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

    ensemble_output_file_path = f"outputs/{model_name}.jsonl"

    with open(palkia_path, "r", encoding="utf-8") as palkia_file, \
            open(charmander_path, "r", encoding="utf-8") as charmander_file, \
            open(ensemble_output_file_path, "w", encoding="utf-8") as ensemble_output_file:
        
        palkia_lines = palkia_file.readlines()
        charmander_lines = charmander_file.readlines()

        for idx, line in enumerate(palkia_lines):
            palkia_entry = json.loads(line)
            charmander_entry = json.loads(charmander_lines[idx])

            all_false = True
            min_gap = float("inf")
            min_gap_emotion = None

            for emotion in emotion_lists:
                palkia_probs = palkia_entry["output"][emotion][1]
                charmander_probs = charmander_entry["output"][emotion][1]

                ensemble_probs = (model_weights["palkia"] * palkia_probs) + (model_weights["charmander"] * charmander_probs)

                if ensemble_probs > emotion_thresholds[emotion]:
                    ensemble_output[emotion] = "True"
                    all_false = False
                else:
                    ensemble_output[emotion] = "False"

                gap = abs(ensemble_probs - emotion_thresholds[emotion])
                if gap < min_gap:
                    min_gap = gap
                    min_gap_emotion = emotion

            if all_false and min_gap_emotion:
                ensemble_output[min_gap_emotion] = "True"

            ensemble_entry = {
                "id": palkia_entry["id"],
                "input": palkia_entry["input"],
                "output": ensemble_output
            }

            ensemble_output_file.write(json.dumps(ensemble_entry, ensure_ascii=False) + "\n")

def old_voting(model_name):
    input_dir = "outputs_backup"
    output_dir = f"outputs/{model_name}.jsonl"
    input_files = [os.path.join(input_dir, f) for f in config[model_name]["composition"]]

    def read_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        return data

    def write_jsonl(data, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            for entry in data:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def hard_voting(*outputs):
        emotions = list(outputs[0].keys())
        votes = Counter({emotion: sum([output[emotion] == "True" for output in outputs]) for emotion in emotions})

        max_votes = votes.most_common(1)[0][1]
        top_voted_emotions = [emotion for emotion, vote_count in votes.items() if vote_count == max_votes]
        ensemble_output = {emotion: 'False' for emotion in emotions}

        for emotion in top_voted_emotions:
            ensemble_output[emotion] = 'True'

        return ensemble_output

    data_list = [read_jsonl(file_path) for file_path in input_files]
    ensemble_results = []
    for entries in zip(*data_list):
        ensemble_output = hard_voting(*[entry["output"] for entry in entries])
        ensemble_results.append({
            "id": entries[0]["id"],
            "input": entries[0]["input"],
            "output": ensemble_output
        })

    write_jsonl(ensemble_results, output_dir)

def new_voting(model_name):
    input_dir = "outputs_backup"
    output_dir = f"outputs/{model_name}.jsonl"
    input_files = [os.path.join(input_dir, f) for f in config[model_name]["composition"]]

    def read_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        return data

    def write_jsonl(data, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            for entry in data:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def hard_voting(*outputs):
        emotions = list(outputs[0].keys())
        votes = Counter({emotion: sum([output[emotion] == "True" for output in outputs]) for emotion in emotions})

        max_votes = votes.most_common(1)[0][1]
        top_voted_emotions = [emotion for emotion, vote_count in votes.items() if vote_count == max_votes]
        ensemble_output = {emotion: 'False' for emotion in emotions}

        num = len(config[model_name]["composition"])/2

        for emotion in emotions:
            if votes[emotion] > num:
                ensemble_output[emotion] = 'True'
            else:
                pass

        if 'True' not in [ensemble_output[emotion] for emotion in emotions]:
            ensemble_output[votes.most_common(1)[0][0]] = 'True'

        return ensemble_output

    data_list = [read_jsonl(file_path) for file_path in input_files]
    ensemble_results = []
    for entries in zip(*data_list):
        ensemble_output = hard_voting(*[entry["output"] for entry in entries])
        ensemble_results.append({
            "id": entries[0]["id"],
            "input": entries[0]["input"],
            "output": ensemble_output
        })

    write_jsonl(ensemble_results, output_dir)

def freeze_voting(model_name="넷볼"):
    original_dro_path = "outputs_backup/오박사.jsonl"
    original_ej_path = "outputs_backup/ej_v3.jsonl"

    dro_strong_emotions_path = "outputs/오박사_strong_emotions.jsonl"
    ej_strong_emotions_path = "outputs/ej_v3_strong_emotions.jsonl"

    new_voting("temp_넷볼")

    temp_net_ball_path = "outputs/temp_넷볼.jsonl"

    # dro freezing (trust, disgust)
    with open(original_dro_path, "r", encoding="utf-8") as file:
        with open(dro_strong_emotions_path, "w", encoding="utf8") as writer:
            for line in file:
                entry = json.loads(line)
                output = {}
                output["trust"] = entry["output"]["trust"]
                output["disgust"] = entry["output"]["disgust"]

                writer.write(json.dumps({"id":entry["id"], "input":entry["input"], "output":output}, ensure_ascii=False) + "\n")

    # ej freezing (anticipation, surprise)
    with open(original_ej_path, "r", encoding="utf-8") as file:
        with open(ej_strong_emotions_path, "w", encoding="utf8") as writer:
            for line in file:
                entry = json.loads(line)
                output = {}
                output["anticipation"] = entry["output"]["anticipation"]
                output["surprise"] = entry["output"]["surprise"]

                writer.write(json.dumps({"id":entry["id"], "input":entry["input"], "output":output}, ensure_ascii=False) + "\n")

    # freeze voting
    with open(dro_strong_emotions_path, "r", encoding="utf-8") as dro_file:
        with open(ej_strong_emotions_path, "r", encoding="utf8") as ej_file:
            with open(temp_net_ball_path, "r", encoding="utf8") as temp_file:
                with open(f"outputs/raw_{model_name}.jsonl", "w", encoding="utf8") as writer:
                    for dro_line, ej_line, temp_line in zip(dro_file, ej_file, temp_file):
                        dro_entry = json.loads(dro_line)
                        ej_entry = json.loads(ej_line)
                        temp_entry = json.loads(temp_line)

                        outputs = {}
                        outputs['joy'] = temp_entry['output']['joy']
                        outputs['anticipation'] = ej_entry['output']['anticipation']
                        outputs['trust'] = dro_entry['output']['trust']
                        outputs['surprise'] = ej_entry['output']['surprise']
                        outputs['disgust'] = dro_entry['output']['disgust']
                        outputs['fear'] = temp_entry['output']['fear']
                        outputs['anger'] = temp_entry['output']['anger']
                        outputs['sadness'] = temp_entry['output']['sadness']

                        writer.write(json.dumps({"id":temp_entry['id'], "input":temp_entry['input'], "output":outputs}, ensure_ascii=False) + '\n')

    # all false 후처리
    with open(f"outputs/raw_{model_name}.jsonl", "r", encoding="utf8") as raw_file:
        with open(f"outputs_backup/마이볼.jsonl", "r", encoding="utf8") as my_ball_file:
            with open(f"outputs/{model_name}.jsonl", "w", encoding="utf8") as writer:
                for raw_line, my_ball_line in zip(raw_file, my_ball_file):
                    raw_entry = json.loads(raw_line)
                    my_ball_entry = json.loads(my_ball_line)
                    outputs = raw_entry['output']

                    if not any(value == "True" for value in outputs.values()):
                        outputs = my_ball_entry['output']

                    writer.write(json.dumps({"id":raw_entry['id'], "input":raw_entry['input'], "output":outputs}, ensure_ascii=False) + '\n')

    os.remove(f"outputs/raw_{model_name}.jsonl")
    os.remove(f"outputs/temp_{model_name}.jsonl")

dialga_ensemble()
# old_voting("output_top_9")
# old_voting("output_top_14")
# new_voting("ej_v3")
# new_voting("오박사")
# new_voting("t9_new_voting")
# new_voting("토게피_v7")
# new_voting("토게피_v8")
# new_voting("only")
# new_voting("뮤츠_v1")
# new_voting("마이볼")
# freeze_voting()
# new_voting("플랑크톤")
new_voting("뮤츠_v2")
