class Agent1:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sys_prompt = """
            You are an agent meant to extract claims and classify their verifiabilities.
            A Smishing-Oriented Claim is defined as an assertive proposition within an SMS that attributes a specific, verifiable state or event to a target entity. Formally, a claim C is a tuple of information slots extracted from the message:
            C = <Subject, Predicate, Condition>

            Your task:
            1. Decompose the provided raw text, providing both the raw claim (plaintext) as well as the parsed claim ([Subject, Predicate, Condition] tuple).
            2. Determine the Verifiability Category for each claim.
            
            Guidelines:
            - Return JSON only
            - Extract atomic claims only
            - Split compound statements into multiple claims
            - Do not paraphrase unless necessary for separation

            Output format (JSON):
                [{"raw_claim": "...", "parsed_claim": [{Subject}, {Predicate}, {Condition}], "Category": "Verifiable / Unverifiable"}, ...]
            """

    def format_user_prompt(self, data_example):
        description = data_example['description']
        
        user_prompt = f"""
        Description: {description}
        """
        return user_prompt

    def parse_input(self, data):
        user_prompt = self.format_user_prompt(data)

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        batch_texts = []

        for ex in data:
            user_prompt = self.format_user_prompt(ex)
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_texts.append(text)

        model_inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)

         # Decode each output
        outputs = []
        for i, gen_ids in enumerate(generated_ids):
            input_len = model_inputs["input_ids"][i].size(0)  # length of input tokens
            output_ids = gen_ids[input_len:]  # skip the input tokens
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(text)

        return outputs