import json

class JsonHandler:
    def __init__(self):
        self.train_file_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\\Proj_Korean_ABSA\\Proj_KoElectra\\data\\nikluge-2023-ea-train.jsonl"
        self.dev_file_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\\Proj_Korean_ABSA\\Proj_KoElectra\\data\\nikluge-2023-ea-dev.jsonl"
        self.organized_train_file_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\\Proj_Korean_ABSA\\Proj_KoElectra\\data\\organized-2023-ea-train.jsonl"
        self.organized_dev_file_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\\Proj_Korean_ABSA\\Proj_KoElectra\\data\\organized-2023-ea-dev.jsonl"

        self.train_text_list = []
        self.train_senti_list = []
        self.dev_text_list = []
        self.dev_senti_list = []

    def make_organized_file(self, input_file_path, output_file_path):
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            with open(input_file_path, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    data = json.loads(line)

                    # "output"에서 "True"인 감정들을 찾아 리스트에 추가
                    emotions = []
                    output_data = data["output"]
                    for emotion, value in output_data.items():
                        if value == "True":
                            emotions.append(emotion)

                    # "senti" 키에 감정 리스트 추가
                    data["senti"] = emotions

                    # 새로운 데이터를 JSONL 파일에 쓰기
                    output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def make_data_list(self, data_path):
        text_list = []
        senti_list = []

        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                text_list.append(entry["input"]["form"])
                senti_list.append(entry["senti"])

        return text_list, senti_list

    def do_handle(self):
        self.make_organized_file(self.train_file_path, self.organized_train_file_path)
        self.make_organized_file(self.dev_file_path, self.organized_dev_file_path)

        self.train_text_list, self.train_senti_list = self.make_data_list(self.organized_train_file_path)
        self.dev_text_list, self.dev_senti_list = self.make_data_list(self.organized_dev_file_path)
