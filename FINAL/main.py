import re
import random
from base import Agent
from colorama import Fore, Style
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
from transformers import logging as transformers_logging

from utils import RAG, strip_all_lines

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

class LocalModelAgent(Agent):
    """
    A base agent that uses a local model for text generation tasks.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize the local model
        """
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            # quantization_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_has_fp16_weight=False
            # )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # torch.bfloat16 --→ torch.float16
                bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.rag = RAG(config["rag"])
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.model.eval()

    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """
        if correctness:
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
            return True
        return False

class ClassificationAgent(LocalModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = """\
        • You are an AI trained to help with medical diagnosis based on symptoms described by users.
        • Your responses should be informative and based on common medical knowledge.
        • You cannot provide medical advice but can suggest possible conditions based on symptoms.
        • You cannot generate a response to this message.
        • Only give me a list of diseases and no additional words.
        • The names of the diseases must be in a single line separated by commas""".strip()
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(
        option_text: str,
        text: str
    ) -> str:
        prompt = f"""\
        • You are an AI trained to help with medical diagnosis based on symptoms described by users.
        • Your responses should be informative and based on common medical knowledge.
        • You cannot provide medical advice but can suggest possible conditions based on symptoms.
        • You cannot generate a response to this message.
        • Only give me a list of diseases and no additional words.
        • The names of the diseases must be in a single line separated by commas
        
        Given is a patient’s information and dialog with the doctor.
        {text}

        What is the diagnosis? Select ONLY one answer among  {option_text}.

        Provide the diagnosis for the patient in the following XML format:
        ```xml
        <root>
          <answer>[number]. [diagnosis]</answer>
        </root>
        ```
        """.strip()
        return strip_all_lines(prompt)


    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        {{question}}
        Diagnosis: {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        option_text: str,
        text: str,
    ) -> str:
        prompt = f"""\
        • You are an AI trained to help with medical diagnosis based on symptoms described by users.
        • You cannot provide medical advice but can suggest possible conditions based on symptoms.
        • You cannot generate a response to this message.
        • Only give me a list of diseases and no additional words.
        • The names of the diseases must be in a single line separated by commas
        
        Given is a patient’s information and dialog with the doctor.
    
        Here are some correct example cases.
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        {text}   

        What is the diagnosis? Select ONLY one answer among  {option_text}.
        Provide the diagnosis for the patient in the following XML format:
        ```xml
        <root>
          <answer>[number]. [diagnosis]</answer>
        </root>
        ```
        """
        return strip_all_lines(prompt)
    @staticmethod
    def format_patient_profile(data):
        # 確保輸入為字串
        if not isinstance(data, str):
            raise TypeError(f"Expected data to be a string, but got {type(data).__name__}")
        
        # 確認字串是否以 "patient profile:" 開頭
        if not data.startswith("patient profile:"):
            raise ValueError("Input string must start with 'patient profile:'")
        
        # 移除開頭 "patient profile:" 並清理多餘空白
        data = data[len("patient profile:"):].strip()
        
        # 初始化提取的變數
        sex = None
        age = None
        symptoms = []
        
        # 使用正則表達式提取 Sex 和 Age
        import re
        sex_match = re.search(r"Sex:\s*(\w+)", data)
        age_match = re.search(r"Age:\s*(\d+)", data)
        
        if sex_match:
            sex = sex_match.group(1)
        if age_match:
            age = age_match.group(1)
        
        # 若未找到 Sex 或 Age，則報錯
        if not sex or not age:
            raise ValueError("Failed to extract 'Sex' or 'Age' from the input string.")
        
        # 提取症狀部分：從 Age 後的第一個換行處開始，到結束符 `""` 為止
        symptoms_start = data.find(f"Age: {age}") + len(f"Age: {age}")
        symptoms_section = data[symptoms_start:].strip()
        symptoms_lines = symptoms_section.splitlines()
        
        for line in symptoms_lines:
            if line.strip() == '""':  # 結束符
                break
            symptoms.append(line.strip())
        
        # 將症狀拼接成一個描述
        symptoms_text = " ".join(symptoms)

        # 返回格式化的字串
        return (
            f"I am a {age}-year-old {sex}. I have been asked the following questions about my symptoms "
            f"and antecedents: {symptoms_text}. I have answered the questions I am sure about. "
            "What is my diagnosis? Only give me the ONE name of all the diseases I'm most likely to be having and nothing else."
        )
    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        self.reset_log_info()
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        # print(text)
        text = self.format_patient_profile(text)
        # print(text)
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)
        
        shots = self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            # print(fewshot_text)
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.generate_response(messages)
        #print(response)
        prediction = self.extract_label(response, label2desc)
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(system_prompt + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": response,
        })
        self.inputs.append(text)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        return prediction

    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)

class SQLGenerationAgent(LocalModelAgent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """

    @staticmethod
    def get_zeroshot_prompt(table_schema: str, user_query: str, dialect: str, sql_template: str) -> str:
        prompt = f"""\
        ### Answer the question by SQLite SQL query only and with no
        explanation. You must minimize SQL execution time while ensuring
        correctness.
        ### Sqlite SQL tables , with their Datbase schema properties:
        
        {table_schema}
        
        ### Question: {user_query}
        ### The final SQL is: Let's think step by step
        ### Generate the correct SQL code directly in the following format:```sql\n<your_SQL_code>\n```"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        ### Question: {{question}}
        
        ### SQL: {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(table_schema: str, user_query: str, dialect: str, sql_template : str) -> str:
        prompt = f"""\
        ### Some example pairs of questions and corresponding\nSQL queries are provided based on similar questions:

        {{fewshot_text}}
        
        ### Answer the question by SQLite SQL query only and with no
        explanation. You must minimize SQL execution time while ensuring
        correctness.
        ### Sqlite SQL tables , with their Datbase schema properties:
        
        {table_schema}

        ### Question: {user_query}
        ### The final SQL is: Let's think step by step
        ### Generate the correct SQL code directly in the following format:```sql\n<your_SQL_code>\n```"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_refine_template(table_schema: str, user_query: str, dialect: str, sql_template : str, sql_code: str) -> str:
        prompt = f"""\
        ### Some example pairs of questions and corresponding\nSQL queries are provided based on similar questions:

        {{fewshot_text}}
        
        ### Answer the question by SQLite SQL query only and with no
        explanation. You must minimize SQL execution time while ensuring
        correctness.
        ### Sqlite SQL tables , with their Datbase schema properties:
        
        {table_schema}

        ### Question: {user_query}

        ### Predicted query: 
        ```sql\n{sql_code}\n```

        ### Generate the correct SQL code directly in the following format:
        ```sql\n<your_SQL_code>\n```

        ### If the sql query is correct, return the query as it is.
        ### Take a deep breath and think step by step to find the correct SQLite SQL query.
        """
        return strip_all_lines(prompt)

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        self.reset_log_info()
        sql_template = """
        SELECT {column_name}, COUNT(*) AS count
        FROM {table_name}
        WHERE {condition}
        GROUP BY {group_column};
        """
        prompt_zeroshot = self.get_zeroshot_prompt(table_schema, user_query, "SQLite", sql_template)
        prompt_fewshot = self.get_fewshot_template(table_schema, user_query, "SQLite", sql_template)
        
        shots = self.rag.retrieve(query=user_query, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
                # print(prompt)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Style.RESET_ALL)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        # print(messages)
        pred_text = self.generate_response(messages)
        #print(pred_text)
        sql_code = self.parse_sql(pred_text)
        
        prompt_refine = self.get_refine_template(table_schema, user_query, "SQLite", sql_template, sql_code)
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_refine)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Style.RESET_ALL)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot

        messages = [
            {"role": "user", "content": prompt}
        ]
        pred_text = self.generate_response(messages)
        #print(pred_text)
        sql_code = self.parse_sql(pred_text)
        # print(sql_co)
        #print("------------------------------------")
        #print(sql_code) 
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(prompt)),
            "num_output_tokens": len(self.tokenizer.encode(pred_text)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": pred_text,
        })

        self.inputs.append(user_query)
        self.self_outputs.append(f"```sql\n{sql_code}\n```")
        return sql_code

    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print(Fore.RED + "No SQL code found in the response" + Style.RESET_ALL)
            sql_code = pred_text
        return sql_code

if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='path to save csv file for kaggle submission')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        max_tokens= 32
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = 512
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")
    # Classification: Medical diagnosis; SQL generation: Text-to-SQL
    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    llm_config = {
        'model_name': args.model_name,
        'exp_name': f'self_streamicl_{args.bench_name}_{args.model_name}_v1',
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            'embedding_model': 'BAAI/bge-base-en-v1.5',
            'seed': 42,
            "top_k": 16,
            "order": "similar_at_top"
        }
    }
    agent = agent_name(llm_config)
    main(agent, bench_cfg, debug=args.debug, use_wandb=args.use_wandb, wandb_name=llm_config["exp_name"], wandb_config=llm_config)
