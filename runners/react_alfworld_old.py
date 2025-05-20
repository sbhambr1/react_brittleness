import os
import sys
import openai
import yaml
import json
import alfworld
import alfworld.agents.environment
from utils.conversation import Conversation
 
openai.api_key = os.environ["OPENAI_API_KEY"]

with open('./ReAct/base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution" # train (3553), eval_in_distribution (140), eval_out_of_distribution (134)

env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

folder = './ReAct/prompts/'
prompt_file = 'alfworld_3prompts.json'
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)
    
llm_model = 'testing'
    
save_path = './data/' + llm_model + '/'
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

    for _ in range(134):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
                print(k, v)
                r = alfworld_run(prompt, ob=ob, task_name=name, prefix=k)
                rs[i] += r
                cnts[i] += 1
                break
        print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')