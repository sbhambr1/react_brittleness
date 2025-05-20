import os
import sys
import openai
import yaml
import json
import alfworld
import alfworld.agents.environment
from utils.conversation import Conversation

EXAMPLE_TO_RUN = 'pick_and_place_simple-Mug-None-Desk-308/trial_T20190908_125200_737896'
EXAMPLE_INDEX = 64 # found using runners/example_finder.py
PILLAR_INDEX = 4 # Content: 1, Abstraction: 2, Nature: 3, Format: 4
VARIATION_INDEX = 3 # Content/Domain: 1, Content/Problem: 2, Content/Instance: 3; Abstraction/Global: 1, Abstraction/Local: 2; Nature/Success: 1, Nature/Failure: 2, Nature/Magic: 3, Nature/Explanation:4; Format/Freeform: 1, Format/Structured: 2, Format/Ordering: 3

expt_name = 'example_format_ordering_variation'
llm_model = 'gpt-3.5-turbo'

openai.api_key = os.environ["OPENAI_API_KEY"]

with open('./data/configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution" # train (3553), eval_in_distribution (140), eval_out_of_distribution (134)
env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

folder = './data/example_prompts/' + f'pillar_{PILLAR_INDEX}'
if PILLAR_INDEX == 1:
    if VARIATION_INDEX == 1:
        prompt_file = '/domain_variation.json'
    elif VARIATION_INDEX == 2:
        prompt_file = '/problem_variation.json'
    elif VARIATION_INDEX == 3:
        prompt_file = '/instance_variation.json'
elif PILLAR_INDEX == 2:
    if VARIATION_INDEX == 1:
        prompt_file = '/global_info.json'
    elif VARIATION_INDEX == 2:
        prompt_file = '/local_info.json'
elif PILLAR_INDEX == 3:
    if VARIATION_INDEX == 1:
        prompt_file = '/success_info.json'
    elif VARIATION_INDEX == 2:
        prompt_file = '/failure_info.json'
    elif VARIATION_INDEX == 3:
        prompt_file = '/magic_info.json'
    elif VARIATION_INDEX == 4:
        prompt_file = '/explanation_info.json'
elif PILLAR_INDEX == 4:
    if VARIATION_INDEX == 1:
        prompt_file = '/freeform_info.json'
    elif VARIATION_INDEX == 2:
        prompt_file = '/structured_info.json'
    elif VARIATION_INDEX == 3:
        prompt_file = '/ordering_info.json'
        
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)
    
save_path = './data/' + expt_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
conversation = Conversation(llm_model, save_path=save_path)

def alfworld_run(prompt, prefix, task_name, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action, task_save_file = conversation.llm_actor(init_prompt + prompt, prefix=prefix, task_name=task_name, step=i, stop=['\n'])
        action = action.strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if done:
            with open(task_save_file, 'a') as f:
                done = {'done': done}
                f.write(json.dumps(done) + '\n')
                f.write(']')
            return reward
        elif i == 49:
            with open(task_save_file, 'a') as f:
                done = {'done': done}
                f.write(json.dumps(done) + '\n')
                f.write(']')
    return 0


if __name__ == '__main__':
    
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

    for idx in range(134):        
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        if name == EXAMPLE_TO_RUN:
            print(name)
            for i, (k, v) in enumerate(prefixes.items()):
                if PILLAR_INDEX == 1:
                    if VARIATION_INDEX == 2:
                        v = 'examine' # 1: clean, 2: examine
                elif PILLAR_INDEX == 2:
                    v = 'put'
                elif PILLAR_INDEX == 3:
                    if VARIATION_INDEX == 1: # baseline ReAct
                        v = 'put'
                    elif VARIATION_INDEX == 3: # think: Take a deep breath and work on this problem step-by-step.
                        v = 'put'
                elif PILLAR_INDEX == 4:
                    if VARIATION_INDEX == 1: # freeform
                        v = 'put'
                    elif VARIATION_INDEX == 2: # structured - baseline ReAct
                        v = 'put'
                    elif VARIATION_INDEX == 3: # ordering shuffled
                        v = 'put'
                else:
                    raise NotImplementedError
                prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
                print(k, v)
                r = alfworld_run(prompt, ob=ob, task_name=name, prefix=k)
                rs[i] += r
                cnts[i] += 1
                break
            print(idx+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
            print('------------\n')
            break
