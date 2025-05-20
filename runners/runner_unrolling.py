import os
import sys
import openai
import yaml
import json
import alfworld
import alfworld.agents.environment
from utils.conversation import OpenAIConversation

# read instances from data/instances_selected.json as list
with open('./data/instances_selected.json', 'r') as f:
    instances_selected = json.load(f)
    
# CATEGORY_TO_RUN = 'put' # put, clean, heat, cool, examine, puttwo, all

# INSTANCES_TO_RUN = instances_selected # list of strings of instances to run
# EXAMPLE_INDEX = [] # list of instance indices to run
# PILLAR_INDEX = 2 # Content: 1, Abstraction: 2, Nature: 3, Format: 4
# VARIATION_INDEX = 1 # Content/Domain: 1, Content/Problem: 2, Content/Instance: 3; Abstraction/Global: 1, Abstraction/Local: 2; Nature/Success: 1, Nature/Failure: 2, Nature/Magic: 3, Nature/Explanation:4; Format/Freeform: 1, Format/Structured: 2, Format/Ordering: 3
# NUM_AGENT_STEPS = 50

pillar_variation_idx2name = {
    1: ['domain', 'problem', 'instance'],
    2: ['global', 'local'],
    3: ['success', 'failure', 'magic', 'explanation'],
    4: ['freeform', 'structured', 'ordering']
}
pillar_idx2name = {
    1: 'content',
    2: 'abstraction',
    3: 'nature',
    4: 'format'
}

# expt_name = 'nature_local_variation'
# # llm_model = 'gpt-3.5-turbo' # 'gpt-3.5-turbo', 'None' for testing ANYTHING
# # llm_model = 'claude-3-haiku-20240307' # haiku < sonnet < opus model size. OTHERS are : 'claude-3-sonnet-20240229', 
# llm_model = 'claude-3-opus-20240229'

expt_config = None
USE_EXAMPLES = None
CATEGORY_TO_RUN = None
INSTANCES_TO_RUN = instances_selected
EXAMPLE_INDEX = []
PILLAR_INDEX = None
VARIATION_INDEX = None
NUM_AGENT_STEPS = 50
save_path = None
expt_name = None 
llm_model = None
conversation = None
d = None


with open('./data/configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution" # train (3553), eval_in_distribution (140), eval_out_of_distribution (134)
env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def run_setup():
    global expt_config, USE_EXAMPLES, CATEGORY_TO_RUN, INSTANCES_TO_RUN, EXAMPLE_INDEX, PILLAR_INDEX, VARIATION_INDEX, NUM_AGENT_STEPS, llm_model, expt_name, save_path, conversation, d

    # # repeat this to set USE_EXAMPLES
    if PILLAR_INDEX == 1 and VARIATION_INDEX == 2:
        if CATEGORY_TO_RUN == 'put':
            USE_EXAMPLES = 'puttwo'
        elif CATEGORY_TO_RUN == 'puttwo':
            USE_EXAMPLES = 'put'
    
    # setup directory for saving
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
        
    save_path = './data/' + llm_model + '/' + expt_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    expt_config = {
        'CATEGORY_TO_RUN': CATEGORY_TO_RUN,
        'INSTANCES_TO_RUN': INSTANCES_TO_RUN,
        'EXAMPLE_INDEX': EXAMPLE_INDEX,
        'PILLAR_INDEX': PILLAR_INDEX,
        'VARIATION_INDEX': VARIATION_INDEX,
        'NUM_AGENT_STEPS': NUM_AGENT_STEPS,
        'expt_name': expt_name,
        'llm_model': llm_model
    }

    conversation = OpenAIConversation(llm_model, save_path=save_path)









def alfworld_run(prompt, prefix, task_name, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, NUM_AGENT_STEPS):
        action, task_save_file = conversation.llm_actor(init_prompt + prompt, prefix=prefix, task_name=task_name, step=i, stop=['\n'])
        action = action.strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        conversation.store_trajectory(step=i, action=action, observation=observation, reward=reward, done=done, task_name=task_name)
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
        elif i == NUM_AGENT_STEPS - 1:
            with open(task_save_file, 'a') as f:
                done = {'done': done}
                f.write(json.dumps(done) + '\n')
                f.write(']')
    return 0


if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser(description="Process variables llm_model, category, pillar, and variation.")
    parser.add_argument("--llm_model", type=str, help="The llm_model variable", default='gpt-3.5-turbo')
    parser.add_argument("--category", type=str, help="The category variable", default='put')
    parser.add_argument("--pillar", type=int, help="The pillar variable", default=1)
    parser.add_argument("--variation", type=int, help="The variation variable", default=2)

    args = parser.parse_args()

    model = args.llm_model
    model2LLMmodel = {
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'haiku': 'claude-3-haiku-20240307',
        'sonnet': 'claude-3-sonnet-20240229',
        'opus': 'claude-3-opus-20240229'
    }
    if model in model2LLMmodel:
        llm_model = model2LLMmodel[model]
    else:
        llm_model = model

    category = args.category
    pillar = args.pillar
    variation = args.variation

    exp_name = f"UNROLLING_{pillar_idx2name[args.pillar]}_{pillar_variation_idx2name[args.pillar][args.variation-1]}"

    # setting global vars...
    CATEGORY_TO_RUN = category 
    PILLAR_INDEX = pillar
    VARIATION_INDEX = variation
    expt_name = exp_name
    llm_model = llm_model

    run_setup()

    print("Model to use : ", llm_model)
    

    prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
    }
    
    # FIND KEY OF 'PUT' IN PREFIXES
    if CATEGORY_TO_RUN != 'all':
        for k, v in prefixes.items():
            if v == CATEGORY_TO_RUN:
                CATEGORY_TO_RUN = k
                config_save_dir = f'{save_path}' + CATEGORY_TO_RUN + '/'
                if not os.path.exists(config_save_dir):
                    os.makedirs(config_save_dir)
                with open(config_save_dir+'config.json', 'w') as f:
                    json.dump(expt_config, f)
                break
    
        for i in range(len(INSTANCES_TO_RUN)):
            if INSTANCES_TO_RUN[i][0].startswith(CATEGORY_TO_RUN):
                instances = INSTANCES_TO_RUN[i]
            
    else:
        instances = INSTANCES_TO_RUN
    
    cnts = [0] * 6
    rs = [0] * 6

    for idx in range(134):        
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])            
        for instance_name in instances:
            if instance_name == name:                     
                for i, (k, v) in enumerate(prefixes.items()):
                    if instance_name.startswith(k):
                        if PILLAR_INDEX == 1 and VARIATION_INDEX == 2:
                            v = USE_EXAMPLES
                        # else: v = v
                        #     v = CATEGORY_TO_RUN
                        print(name)
                        INSTRUCTION = " Only provide the one action at a time. Be concise, and do not provide any extra information. Always start with the action. For example, 'pick up apple' is correct, but 'I want to pick up the apple' is not."
                        prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + INSTRUCTION + '\nHere is the task.\n'
                        print(k, v)
                        reward = alfworld_run(prompt, ob=ob, task_name=name, prefix=k)
                        rs[i] += reward
                        cnts[i] += 1
                        print(idx+1, 'r', reward, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
                        print('------------\n')
                        break
                    
                
    print('Final:', 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    
    with open(config_save_dir + 'results.json', 'w') as f:
        json.dump({'rs': rs, 'cnts': cnts, 'sum(rs)/sum(cnts)': sum(rs) / sum(cnts)}, f)
