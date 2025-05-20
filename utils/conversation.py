import os
import json
import tiktoken
from openai import OpenAI
import anthropic
import pickle as pkl 
from ratelimit import limits, sleep_and_retry
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


OTHER_LLM_APIs = False
if OTHER_LLM_APIs : 
    # pip install -q -U google-generativeai
    import google.generativeai as genai
    # pip install anthropic
    # pip install transformers




# we should restructure this to abstract Conversation -> and subsequent specific models. but it's ok for now. 
class AbstractConversation : 
    def __init__(self, model_name, save_path, config) : 
        self.model_name = model_name 
        self.save_path = save_path 
        self.config = config 

    
    def llm_actor(self, prompt, prefix, task_name, step, stop, temperature, role) : 
        pass 

class HFLLMConversation(AbstractConversation):
    def __init__(self, model_name, save_path, config=None) : 
        super().__init__(model_name, save_path, config)
        
    def llm_actor(self, prompt, prefix, task_name, step, stop, temperature, role) : 
        pass 

class GeminiConversation(AbstractConversation):
    def __init__(self, model_name, save_path, config=None):
        super().__init__(model_name, save_path, config)
        self.model = genai.GenerativeModel(model_name="gemini-pro-vision")

    def _reply(self, prompt, config):
        response = self.model.generate_content(["What's in this photo?", img])
        return response




class OpenAIConversation:
    def __init__(self, llm_model, save_path) -> None:
        self.llm_prompt = []
        self.log_history = []
        self.llm_model =  llm_model
        self.tokens_per_min = 0
        self.max_tokens = 256
        self.input_token_cost = 0.5 / 1e6
        self.output_token_cost = 1.5 / 1e6
        self.total_cost = 0
        self.cost_limit = 20
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.save_path = save_path
        
        self.setup_client(llm_model)
    
    def setup_client(self, llm_model):
        if llm_model == 'gpt-3.5-turbo' or llm_model == 'gpt-4':
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
            )
            self.openai_client = client
        elif 'claude' in llm_model:
            anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            self.anthropic_client = anthropic_client
            self.max_tokens_per_min = 50000 #TODO: get this from the API
            self.max_tokens_per_day = 1000000

    def count_tokens(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
       
    def construct_message(self, prompt, role):
        assert role in ["user", "assistant"]
        new_message = {"role": role, "content": prompt}
        self.llm_prompt = []
        message = self.llm_prompt + [new_message]
        input_tokens = self.count_tokens(message[0]['content'],  'cl100k_base')
        if 'gpt' in self.llm_model:
            self.total_input_tokens += input_tokens
            self.total_cost += self.input_token_cost * input_tokens
        return message, input_tokens
    
    def llm_actor(self, prompt, prefix, task_name, step, stop, temperature=0, role="user"): 
            
        self.task_save_dir = self.save_path + prefix + '/'
        if not os.path.exists(self.task_save_dir):
            os.makedirs(self.task_save_dir)
        
        task_save_file = f"{self.task_save_dir}{task_name}.json"
        
        task = {"task_name": task_name}     
        
        if not os.path.exists(task_save_file):
            os.makedirs(os.path.dirname(task_save_file), exist_ok=True)
            with open(task_save_file, 'a') as f:
                if f.tell() == 0:
                    f.write('[\n')
                f.write(json.dumps(task) + ',\n')  
        
        if self.llm_model != 'None':
            # chat model
            message, input_tokens = self.construct_message(prompt, role)  

            if self.llm_model == 'gpt-3.5-turbo' or self.llm_model == 'gpt-4':
                client = self.openai_client

                if self.total_cost > self.cost_limit:
                    return {"response_message": "[WARNING] COST LIMIT REACHED!"}
                else:
                    response = client.chat.completions.create(
                    model=self.llm_model,
                    messages = message,
                    temperature=temperature,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop
                    )

                answer = response.choices[0].message.content
                output_tokens = self.count_tokens(answer, 'cl100k_base')
                self.total_output_tokens += output_tokens
                self.total_cost += self.output_token_cost * output_tokens
                
            elif self.llm_model == 'text-davinci-002': #deprecated
                client = self.openai_client
                response = client.completions.create(
                    model=self.llm_model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                    )
                answer = response["choices"][0]["text"]
                output_tokens = self.count_tokens(answer, 'cl100k_base')
                self.total_output_tokens += output_tokens
                self.total_cost += self.output_token_cost * output_tokens
            
            elif 'claude' in self.llm_model : 
                anthropic_client = self.anthropic_client
                local_config = {"max_tokens": 100, "temperature": 0}
                
                
                @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6))
                def get_claude_response():
                    tokens_per_min = 0
                    tokens_per_day = 0
                    response = anthropic_client.messages.create(
                        model=self.llm_model,
                        max_tokens=local_config['max_tokens'],
                        temperature=local_config['temperature'],
                        messages=message
                    )
                    claude_input_tokens = response.usage.input_tokens
                    self.total_input_tokens += claude_input_tokens
                    claude_output_tokens = response.usage.output_tokens
                    self.total_output_tokens += claude_output_tokens
                    answer = response.content[0].text
                    tokens_per_min += claude_input_tokens + claude_output_tokens
                    tokens_per_day += claude_input_tokens + claude_output_tokens
                    if tokens_per_min > self.max_tokens_per_min:
                        print("[INFO] Sleeping for 60 seconds to avoid rate limit.")
                        time.sleep(60)
                        tokens_per_min = 0
                    if tokens_per_day > self.max_tokens_per_day:
                        print("[INFO] Sleeping for 24 hours to avoid rate limit.")
                        time.sleep(86400)
                        tokens_per_day = 0
                    return answer, claude_input_tokens, claude_output_tokens
                
                answer, input_tokens, output_tokens = get_claude_response()
        
        else:
            answer = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        
        
        self.log_history.append(answer)
        self.llm_prompt.append(prompt + answer + "\n")
        
        task_step = {"step": step, "prompt": prompt, "response": answer, "input_tokens": self.total_input_tokens, "output_tokens": self.total_output_tokens}
        with open(task_save_file, 'a') as f:
            f.write(json.dumps(task_step) + ',\n') 
            
        print("[TOTAL INPUT TOKENS]: ", self.total_input_tokens)
        print("[TOTAL OUTPUT TOKENS]: ", self.total_output_tokens)   
            
        return answer, task_save_file
    
    def store_trajectory(self, step, action, observation, reward, done, task_name):
        
        trajectory_save_file = f'{self.task_save_dir}{task_name}_trajectory.json'
        
        with open(trajectory_save_file, 'a') as f:
            if f.tell() == 0:
                    f.write('[\n')
            step = {"step": step, "action": action, "observation": observation, "reward": reward, "done": done}
            f.write(json.dumps(step) + ',\n')
            if done or step == 49:
                done = {"done": done}
                f.write(json.dumps(done) + '\n')
                f.write(']')

if __name__ == "__main__":
    # c = OpenAIConversation("gpt-3.5-turbo")
    c = OpenAIConversation("claude-3-haiku-20240307", "./data/claude-3-haiku-20240307/")
    PROMPT_1 = "Hello, how are you?"
    PROMPT_2 = "I've been better."

    response = c.llm_actor(PROMPT_1, "testprefix string", task_name="test", step=1, stop=['\n'])
    print("Final resp : ", response)