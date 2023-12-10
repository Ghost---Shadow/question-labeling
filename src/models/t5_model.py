from transformers import AutoTokenizer, T5ForConditionalGeneration


class T5ModelForQuestionGeneration:
    def __init__(self, config):
        self.config = config
        size = self.config["architecture"]["question_generator_model"]["size"]
        device = self.config["architecture"]["question_generator_model"]["device"]

        full_name = f"google/flan-t5-{size}"

        self.model = T5ForConditionalGeneration.from_pretrained(full_name)
        self.tokenizer = AutoTokenizer.from_pretrained(full_name, model_max_length=512)
        self.model = self.model.to(device)
        self.device = device

    def generate_question(self, passage):
        prompt = f"Generate a question for the given passage.\nPassage: {passage}\nQuestion: "
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        outputs = self.model.generate(input_ids, max_new_tokens=400)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_question_batch(self, passages):
        prompts = []
        for passage in passages:
            prompt = f"Generate a question for the given passage.\nPassage: {passage}\nQuestion: "
            prompts.append(prompt)

        input_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_new_tokens=400)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
