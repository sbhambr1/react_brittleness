# Code for the TMLR paper: "Do Think Tags Really Help LLMs Plan? A Critical Evaluation of ReAct-Style Prompting"


## Use VSCode DevContainers

Setup : 

1. Make sure VSCode has devcontainer extension installed. 
2. You have docker that is already setup (you can run `docker ps`, `docker images`) easily.

Running : 
1. Clone the repository : `git clone https://github.com/sbhambr1/react_brittleness`
2. Run the devcontainer : VSCode should give a popup to run the code within a devcontainer. If not, then do Cmd + Shift + P to open VSCode command pallete and search for `Rebuild Container` which should start the devcontainer. 
3. Specify `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` as environment variable. 


**Running Webshop**
1. In the devcontainer use docker image : `famishedrover/taxonomy_llm:webshop` 
2. Run the webshop by running.
```cmd 
source /webvenv/bin/activate 
cd /webshop/
./run_dev.sh 
```
3. Open the webpage. VSCode should prompt you, otherwise Flask will also log a message that the website is accessible on link like : `172.0.0.6:3000` (Use the link mentioned in the message!)

4. Run OpenAI code using native python (not webvenv)


## Installation for Local setup

```bash
pip install openai anthropic ratelimit alfworld
```


```bash
git clone https://github.com/sbhambr1/react_brittleness
conda create -n react_test python=3.9
conda activate react_test
pip install -r requirements.txt
```

## Directory Setup

```bash
mkdir data
```

## Run ReAct Baseline

```bash
python runners/react_alfworld.py
```


## Running Webshop 

Run `patchfix.sh` for each container. It updates the `/webshop/web_agent_site/utils.py` to use the larger dataset and downloads it using `webvenv` virtual environment present in the container. 
