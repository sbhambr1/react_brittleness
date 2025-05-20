import yaml
import json
import alfworld
import alfworld.agents.environment

NUM_INSTANCES = 10 # 10 instances per task, currently taking first 10 in each category

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
    }
    
cnts = [0] * 6
instances_selected = [[] for _ in range(6)]

with open('./ReAct/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution" # train (3553), eval_in_distribution (140), eval_out_of_distribution (134)

env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

for _ in range(134):
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k): 
            if cnts[i] < NUM_INSTANCES:
                cnts[i] += 1
                instances_selected[i].append(name)
                
print(instances_selected)

# Save the selected instances to a file
instance_save_file = './data/instances_selected.json'

with open(instance_save_file, 'w') as f:
    json.dump(instances_selected, f)