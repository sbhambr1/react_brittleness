import os
import json

react_dir = './data/gpt-3.5-turbo/REACT_baseline/'

total_input_tokens = 0
total_output_tokens = 0

for task_dir in os.listdir(react_dir):
        if os.path.isdir(react_dir + task_dir):
            for inst_dir in os.listdir(react_dir + task_dir):
                for file in os.listdir(react_dir + task_dir + '/' + inst_dir):
                    file_path = react_dir + task_dir + '/' + inst_dir + '/' + file
                    if file_path.endswith('trajectory.json'):
                        continue
                    else:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for i in range(1, len(data)):
                                try:
                                    total_input_tokens += data[i]['input_tokens']
                                    total_output_tokens += data[i]['output_tokens']
                                except:
                                    pass
                        
print(total_input_tokens, total_output_tokens)
        
    
