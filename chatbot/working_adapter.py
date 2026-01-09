## Python libraries #########################################################################################################################################
#

import configparser
import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

## Setup ####################################################################################################################################################
#

class WorkingAdapter:
    def __init__(self, config_path=None):
        # 1. Laad environment variables (.env)
        load_dotenv()
        
        self.config = configparser.ConfigParser()
        self.client = None
        self.model = "gpt-3.5-turbo"
        self.is_huggingface = False

        # 2. Lees config bestand(en)
        paths_to_check = []
        if config_path:
            paths_to_check.append(Path(config_path))
        
        base_dir = Path(__file__).resolve().parent
        paths_to_check.append(base_dir / 'config.ini')
        paths_to_check.append(base_dir.parent / 'config.ini')
        
        for path in paths_to_check:
            if path.exists():
                self.config.read(path)
                break

## Main #####################################################################################################################################################
#

	## Option A -> OpenAI Key in .env

        env_key = os.getenv("OPENAI_API_KEY")
        
        if env_key:
            print("OpenAI API Key found in .env")
            self.client = OpenAI(api_key=env_key)
            
            if 'OPENAI' in self.config and 'model_name' in self.config['OPENAI']:
                self.model = self.config['OPENAI']['model_name']

            else: # Default if [OPENAI]-section is not found
                self.model = "gpt-3.5-turbo"
            
            print(f"OpenAI model: {self.model}")
            return

	## Option B -> OpenAI Key in config.ini

        if 'OPENAI' in self.config and 'api_key' in self.config['OPENAI']:
            print("OpenAI API Key found in config.ini.")
            self.client = OpenAI(api_key=self.config['OPENAI']['api_key'])

            self.model = self.config['OPENAI'].get('model_name', 'gpt-3.5-turbo')
            return

	## Option C -> HuggingFace fallback

        if 'AI' in self.config and 'hf_token' in self.config['AI']:
            print("No OpenAI key, switch to HuggingFace.")
          
            self.token = self.config['AI']['hf_token']
            self.model = self.config['AI'].get('model_judge', 'HuggingFaceH4/zephyr-7b-beta')
            self.client = InferenceClient(token=self.token)
            self.is_huggingface = True
            return

        raise ValueError("No API keys found")

    def chat_completion(self, system_prompt, user_prompt, max_tokens=200, temperature=0.7):
        """
        Stuurt bericht naar de geconfigureerde client.
        """
        if self.is_huggingface: # if llm = Huggingface
            try:
                formatted_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
                response = self.client.text_generation(
                    formatted_prompt,
                    model=self.model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_full_text=False
                )
                return response.strip()

            except Exception as e:
                print(f"HuggingFace Error: {e}")
                return "Error"

        try: # if llm = OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return "Error generating response"

    def generate(self, system_prompt_or_prompt, user_prompt=None, max_tokens=100):
        if user_prompt is None: # 1 argument = user prompt            
            return self.chat_completion("You are a helpful assistant.", system_prompt_or_prompt, max_tokens)

        else: # 2 arguments = system + user prompts            
            return self.chat_completion(system_prompt_or_prompt, user_prompt, max_tokens)