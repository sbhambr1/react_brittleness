import json 
import alfworld 
import sys , os 

# base_path = "./outputs/gpt-3.5-turbo-instruct/1713395922_3327913"
base_path = "perturb_outputs/gpt-3.5-turbo-instruct/content/problem/1713462738_8277943"

def response_from_saved_interaction(path):
    try : 
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            answers = []
            for d in data:
                answers.append(d['response']['choices'][0]['text'])
            return answers
    except Exception as e:
        print("Exception found {e}")
        return None



def alfworld_run(responses, to_print=True, ob=''):
    if to_print:
        print(ob)
        sys.stdout.flush()

    prev = None

    metrics = {
        'valid_think' : 0,
        'valid_think_total' : 0,
    }

    for i in range(0, 49):
        action = responses[i].strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

        
        # metrics....
        # think followed by valid action v/s think followed by invalid action...
        if prev and 'think' in prev[0] and 'think' not in action:
            if observation != 'Nothing happens.' : 
                metrics['valid_think'] += 1
            metrics['valid_think_total'] += 1
        
        if prev and action == prev[0] and observation == prev[1]:
            return 0, metrics
        if 'think' not in action and ('apologize' in action or 'Apologize' in action):
            return 0, metrics
        
        prev = (action, observation)

        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()

        if done:
            return reward, metrics
    return 0, metrics


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



m = {
    'valid_think' : 0, 
    'valid_think_total' : 0, 
}

class_metrics = {k : m.copy() for k in prefixes.keys()}

for idx in range(134):
    print(f"At index {idx}")
    path = f'{base_path}/response_{idx}.json'
    
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(name)


    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            answers = response_from_saved_interaction(path)
            if answers is None : 
                break # handle incomplete runs etc...

            r, metrics = alfworld_run(answers, ob=ob)
            class_metrics[k]['valid_think'] += metrics['valid_think']
            class_metrics[k]['valid_think_total'] += metrics['valid_think_total']

            break

for x in class_metrics : 
    k = class_metrics[x]
    val = k['valid_think']/(k['valid_think_total'] + 1e-6)
    print(x, val, 1 - val)

print("Done")