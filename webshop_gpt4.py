import sys
import os
import boto3
from botocore.exceptions import ClientError
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

os.environ["OPENAI_API_KEY"] = None
os.environ["AWS_ACCESS_KEY"] = None
os.environ["AWS_SECRET_ACCESS_KEY"] = None
os.environ["ANTHROPIC_API_KEY"] = None

from openai import OpenAI
import time 
import json 
import anthropic

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # f

LLM_MODEL = "gpt-4"

class LLM : 
    def __init__(self, model, file_path) -> None:
        self.model = model
        if 'gpt' in self.model:
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
            )
        elif 'llama3-1' in self.model:
            self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
            
        elif 'claude' in self.model:
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            self.client = self.anthropic_client
            
        self.file_path = file_path
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
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
        if LLM_MODEL in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4"]: 
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
            
        elif 'llama3' in self.model:
            prompt += "Only provide the one action at a time. Be concise, and do not provide any extra information. Always start with the action."
            formatted_prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>user<|end_header_id|>
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

            # Format the request payload using the model's native structure.
            native_request = {
                "prompt": formatted_prompt,
                "max_gen_len": 512,
                "temperature": 0,
            }
            
            request = json.dumps(native_request)
            try:
                # Invoke the model with the request.
                response = self.client.invoke_model(modelId=self.model, body=request)

            except (ClientError, Exception) as e:
                print(f"ERROR: Can't invoke '{self.model}'. Reason: {e}")
                exit(1)

            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            answer = model_response["generation"]
            if answer[0] == ' ':
                answer = answer[1:]
            response = answer
            
        elif 'claude' in self.model:
        
            anthropic_client = self.anthropic_client
            local_config = {"max_tokens": 100, "temperature": 0}
            
            
            @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
            def get_claude_response():
                # tokens_per_min = 0
                # tokens_per_day = 0
                response = anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=local_config['max_tokens'],
                    temperature=local_config['temperature'],
                    messages=message
                )
                answer = response.content[0].text
                # if tokens_per_min > self.max_tokens_per_min:
                #     print("[INFO] Sleeping for 60 seconds to avoid rate limit.")
                #     time.sleep(60)
                #     tokens_per_min = 0
                # if tokens_per_day > self.max_tokens_per_day:
                #     print("[INFO] Sleeping for 24 hours to avoid rate limit.")
                #     time.sleep(86400)
                #     tokens_per_day = 0
                return answer
            
            answer = get_claude_response()

        self.save_response(message, response)
        return answer
    
    def save_response(self, message, response):
        if 'gpt' in self.model:
            data = {
                "message": message,
                "response": json.loads(response.to_json()),
            }
        elif 'llama3' in self.model:
            data = {
                "message": message,
                "response": response,
            }
        # append response to an existing json file
        file_path = self.file_path            

        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.append(data)
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file)
            

# WEBSHOP_URL = "http://3.83.245.205:3000"
# WEBSHOP_URL = "http://172.17.0.3:3000/"
WEBSHOP_URL = "http://127.0.0.1:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}
  
  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done




def main():
    
    env = webshopEnv()
    
    #REACT baseline trivial search & item, choose option
    prompt1 = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
    Observation: OK.

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
    Observation: OK.

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    # ACT Only : trivial search & item, choose option
    prompt1_actonly = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    #Rq1 - exemplar COT - prompt1_cot
    prompt1_cot = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 

    Action: think[Once I find items that are bright citrus deodorant less then 50 dollars, I can check them one by one. For 3 ounce bottle of bright citrus deodorant for sensitive skin, if the item has options 'bright citrus' and '3 ounce (pack of 1)', it will be good to buy.]
    Observation: OK.

    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    #Rq1 - exemplar anon COT - prompt1_cot_anon
    prompt1_cot_anon = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 

    Action: think[Once I find the desired items, I can check them one by one. For 3 ounce bottle of bright citrus deodorant for sensitive skin, if the item has the desired options, it will be good to buy.]
    Observation: OK.

    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    #Rq2 - failure - prompt1_failure
    prompt1_failure = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: click[B999WW3JKJ]
    Observation: Nothing happens

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: click[ginger]
    Observation: Nothing happens

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    #Rq2 - failure + exp - prompt1_failure_exp
    prompt1_failure_exp = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: click[B999WW3JKJ]
    Observation: Nothing happens

    Action: think[Nothing happens because I clicked on an invalid item.]
    Observation: OK.

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: click[ginger]
    Observation: Nothing happens

    Action: think[Nothing happens because I clicked on the incorrect fragrance.]
    Observation: OK.

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    #Rq2 - placebo - prompt1_placebo
    prompt1_placebo = """Webshop 
    Instruction:  
    i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
    [Search]  

    Action: search[3 ounce bright citrus deodorant sensitive skin]
    Observation: 
    [Back to Search] 
    Page 1 (Total results: 50) 
    [Next >] 
    [B078GWRC1J] 
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B078GTKVXY] 
    Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    $10.99 
    [B08KBVJ4XN] 
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
    $15.95  

    Action: think[Take a deep breath and work on this problem step-by-step.]
    Observation: OK.

    Action: click[B078GWRC1J]
    Observation: 
    [Back to Search] 
    [< Prev] 
    scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
    size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
    Price: $10.99 
    Rating: N.A. 
    [Description] 
    [Features] 
    [Reviews] 
    [Buy Now]  

    Action: think[Take a deep breath and work on this problem step-by-step.]
    Observation: OK.

    Action: click[bright citrus]
    Observation: You have clicked bright citrus. 

    Action: click[3 ounce (pack of 1)]
    Observation: You have clicked 3 ounce (pack of 1). 

    Action: click[Buy Now]
    """

    
    EXPT_index = 0
    expt_name = ''
    
    for expt in range(1,7):
        EXPT_index = expt

        if EXPT_index == 0:
            prompt1 = prompt1
            expt_name = '/react_results_50'
        elif EXPT_index == 1:
            prompt1 = prompt1_actonly
            expt_name = '/actonly_result_50'
        elif EXPT_index == 2:
            prompt1 = prompt1_cot
            expt_name = '/cot_results_50'
        elif EXPT_index == 3:
            prompt1 = prompt1_cot_anon
            expt_name = '/cot_anon_results_50'
        elif EXPT_index == 4:
            prompt1 = prompt1_failure
            expt_name = '/failure_results_50'
        elif EXPT_index == 5:
            prompt1 = prompt1_failure_exp
            expt_name = '/failure_exp_results_50'
        elif EXPT_index == 6:
            prompt1 = prompt1_placebo
            expt_name = '/placebo_results_50'
            
            

        SAVE_PATH= f"./test_webshopoutput/{LLM_MODEL}"
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        expt_file_path = SAVE_PATH + expt_name + '.txt'

        llm = LLM(LLM_MODEL, SAVE_PATH + expt_name + '.json')

        def webshop_run(idx, prompt, to_print=True):
            action = 'reset'
            init_prompt = prompt
            prompt = ''
            ACTIONS_LIMIT = 15
            for i in range(ACTIONS_LIMIT):
                try:
                    res = env.step(idx, action)
                    observation = res[0]
                except AssertionError:
                    observation = 'Invalid action!'

                if action.startswith('think'):
                    observation = 'OK.'


                if to_print:
                    print(f'Action: {action}\nObservation: {observation}\n')
                    sys.stdout.flush()
                if i:
                    prompt += f' {action}\nObservation: {observation}\n\nAction:'
                else:
                    prompt += f'{observation}\n\nAction:'
                
                if res[2]:  
                    return res[1]

                action = llm.llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')

            return 0

        def run_episodes(prompt, n=100, expt_file_path=None):
            rs = []
            cnt = 0
            for i in range(n):
                print('-----------------')
                print(i)
                try:
                    r = webshop_run(f'fixed_{i}', prompt, to_print=True)
                except AssertionError:
                    r = 0
                cnt += 1
                rs.append(r)
                if (i+1) % 1 == 0:
                    r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
                print(i+1, r, sr, fr)
                print('-------------')
            r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
            
            with open(expt_file_path, 'w') as f:
                f.write(f"{r}, {sr}, {fr}")
            
            print(r, sr, fr)
            return rs
                
        if not os.path.exists(expt_file_path):
            os.makedirs(os.path.dirname(expt_file_path), exist_ok=True)
        res1 = run_episodes(prompt1, 50, expt_file_path)
        
    
if __name__ == '__main__':
    main()