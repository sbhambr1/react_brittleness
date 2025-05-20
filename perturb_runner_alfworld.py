import os

from openai import OpenAI
import anthropic
 

# CLAUDE MODELS 'claude-3-sonnet-20240229'  'claude-3-haiku-20240307', 'claude-3-opus-20240229'

LLM_MODEL = "gpt-3.5-turbo"
# LLM_MODEL="gpt-3.5-turbo-instruct"
# LLM_MODEL = 'claude-3-haiku-20240307'
# LLM_MODEL = 'claude-3-sonnet-20240229'
# LLM_MODEL = 'claude-3-opus-20240229'

class LLM : 
    def __init__(self, model, file_path) -> None:
        self.model = model
        if 'gpt' in self.model : 
            self.client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                )
        elif 'claude' in self.model : 
            self.client = anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

        self.file_path = file_path
        with open(file_path, 'w') as json_file:
            json.dump([], json_file)
    

    def llm(self, prompt, stop=["\n"]):
        try : 
            res = self._llm(prompt, stop)
            return res
        except Exception as e:
            print(f"Error: {e}")
            print("Waiting 100 seconds and Retrying")
            time.sleep(100)
            res = self.llm(prompt, stop)
            return res
     
    def _llm(self, prompt, stop=["\n"]):
        message, answer = '', ''
        if LLM_MODEL == "gpt-3.5-turbo" : 
            message = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model= LLM_MODEL, 
                messages= message,
                temperature=0.0,
                max_tokens=100,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
            )
            answer = response.choices[0].message.content

        elif LLM_MODEL == "gpt-3.5-turbo-instruct":
            response = self.client.completions.create(
                model=LLM_MODEL,
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop
                )
            answer = response.choices[0].text
        
        elif 'claude' in LLM_MODEL : 
            message = [{"role": "user", "content": prompt}]
            response = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=100,
                temperature=0,
                messages=message
            )
            answer = response.content[0].text

        self.save_response(message, response)
        return answer
    
    def save_response(self, message, response):
        data = {
            "message": message,
            "response": json.loads(response.to_json()),
        }
        # append response to an existing json file
        file_path = self.file_path            

        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.append(data)
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file)
    


import yaml
import alfworld
import alfworld.agents.environment
with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution"

env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob



import json
folder = './prompts/perturb_example_prompts/'

# list of all valid are : 
# content/domain
# content/problem
# content/problem-partial
# content/problem-all
# content/instance
# abstraction/global
# abstraction/local
# nature/explanation
# nature/failure
# nature/success
# nature/magic
# structure/freeform
# structure/ordering
# structure/structured


# abstraction/global-problem-partial
# abstraction/global-problem
# abstraction/global-problem-all


PERTURB_MODE_MAPPING = {
    'put':'examine',
    'clean':'cool',
    'heat':'put',
    'cool':'puttwo',
    'examine':'clean',
    'puttwo':'heat'
}


# PERTURB_MODE = 'nature/optimal_plan_length'
PERTURB_MODE = 'actonly'
prompt_file =  PERTURB_MODE + '.json'
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)

import sys

def alfworld_run(prompt, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action = llm.llm(init_prompt + prompt, stop=['\n']).strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if done:
            return reward
    return 0




prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
cnts = [0] * 6
rs = [0] * 6

import time 

SAVE_PATH = f'./perturb_outputs/{LLM_MODEL}/{PERTURB_MODE}/{str(time.time()).replace(".", "_")}'
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Saving at {SAVE_PATH}")

for idx in range(134):
    llm = LLM(LLM_MODEL, SAVE_PATH + f'/response_{idx}.json')

    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(name)
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            eg1 = d[f'react_{v}_1']
            eg2 = d[f'react_{v}_0']
            if PERTURB_MODE == 'content/problem' or PERTURB_MODE == 'abstraction/global-problem':
                # use two examples of the another task
                v = PERTURB_MODE_MAPPING[v]
                eg1 = d[f'react_{v}_1']
                eg2 = d[f'react_{v}_0']
            elif PERTURB_MODE == 'content/problem-partial' or PERTURB_MODE == 'abstraction/global-problem-partial':
                # use one example of the same task, and one example of another task
                v2 = PERTURB_MODE_MAPPING[v]
                eg1 = d[f'react_{v}_1']
                eg2 = d[f'react_{v2}_1']
            elif PERTURB_MODE == 'content/problem-all' or PERTURB_MODE == 'abstraction/global-problem-all':
                # use one example of all tasks
                egs = []
                tasks = PERTURB_MODE_MAPPING.keys()
                for task in tasks:
                    if task == v:
                        continue
                    egs.append(d[f'react_{task}_1'])
                egs.append(d[f'react_{v}_1']) # add the example of the current task in the end
                eg1 = ' '.join(egs) # eg1 is the concatenation of all examples
                eg2 = '' # eg2 is placeholder


            INSTRUCTION = ''
            if 'claude' in LLM_MODEL : 
                INSTRUCTION = " Only provide the one action at a time. Be concise, and do not provide any extra information. Always start with the action. For example, 'pick up apple' is correct, but 'I want to pick up the apple' is not."

            prompt = 'Interact with a household to solve a task. Here are two examples.\n' + eg1 + eg2 + INSTRUCTION + '\nHere is the task.\n'

            print(k, v)
            r = alfworld_run(prompt, ob=ob)
            rs[i] += r
            cnts[i] += 1
            break
    print(idx+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    print('------------\n')

    # save metrics in a json file 
    with open( SAVE_PATH + f'/metrics_{idx}.json', 'w') as writer:
        json.dump({'rs': rs, 'cnts': cnts}, writer)
