import json

class CreatePrompt:
    def __init__(self):
        self.train_dataset_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\Proj_Korean_ABSA\\Proj_KoElectra\\data\\nikluge-2023-ea-train.jsonl"
        self.train_prompt = ""

        self.test_dataset_path = "C:\\Users\\JongbeenSong\\Desktop\\23-S\\LAB\Proj_Korean_ABSA\\Proj_KoElectra\\data\\nikluge-2023-ea-test.jsonl"
        self.test_prompt = ""

        self.base_prompt = ""
        self.before_train_prompt = ""
        self.before_test_prompt = ""
        self.final_prompt = ""

    def open_train_dataset(self):
        with open(self.train_dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[:2]

    def open_test_dataset(self):
        with open(self.test_dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines

    def create_train_prompt(self):
        lines = self.open_train_dataset()
        for line in lines:
            data = json.loads(line)
            input_form = data['input']['form']
            input_target_form = data['input']['target']['form']
            output_data = data['output']
            self.train_prompt += f"input_form: {input_form}\ninput_target_form: {input_target_form}\noutput_data: {output_data}\n\n"

        with open("train_prompt.txt", "w", encoding="utf-8") as f:
            f.write(self.train_prompt)

    def open_train_prompt(self):
        with open("train_prompt.txt", "r", encoding="utf-8") as f:
            self.train_prompt = f.read()

    def create_test_prompt(self):
        lines = self.open_test_dataset()
        for line in lines:
            print("line: ", line)
            data = json.loads(line)
            input_form = data['input']['form']
            input_target_form = data['input']['target']['form']
            self.test_prompt += f"input_form: {input_form}\ninput_target_form: {input_target_form}\n\n"

        with open("test_prompt.txt", "w", encoding="utf-8") as f:
            f.write(self.test_prompt)

    def create_base_prompt(self, CREATE_TRAIN_PROMPT=True):
        self.base_prompt = """For the following tasks, I need you to perform Target-based Sentiment Analysis on Korean input forms. I will provide you with an "input_form" which is a Korean sentence and an "input_target_form" which is the target word or phrase from that sentence. Based on the sentiment expressed towards the target in the sentence, classify the sentiment into the following categories: 'joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', and 'sadness'. Please provide an 'output_data' dictionary with 'True' or 'False' values for each of the sentiment categories.\n\n"""

        self.before_train_prompt = """For clarity, here are examples:\n"""

        self.before_test_prompt = """Now, please analyze the following sentences:\n"""

        if CREATE_TRAIN_PROMPT:
            self.create_train_prompt()
        else:
            self.open_train_prompt()

        # len(self.base_prompt) = 1,111
        self.base_prompt = self.base_prompt + self.before_train_prompt + self.train_prompt + self.before_test_prompt

    def create_final_prompt(self):
        self.final_prompt = self.base_prompt
        lines = self.open_test_dataset()
        for line in lines:
            data = json.loads(line)
            input_form = data['input']['form']
            input_target_form = data['input']['target']['form']
            input_line = f"input_form: {input_form}\ninput_target_form: {input_target_form}\n\n"

            if len(self.final_prompt) <= 2048:
                self.final_prompt += input_line
            else:
                with open(f"prompt\\final_prompt_{lines.index(line)}.txt", "w", encoding="utf-8") as f:
                    f.write(self.final_prompt)
                self.final_prompt = self.base_prompt

cp = CreatePrompt()
cp.create_base_prompt()
cp.create_final_prompt()
