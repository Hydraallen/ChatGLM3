# ChatGLM3 Web Demo

## Installation

We recommend managing environments through [Conda](https://docs.conda.io/en/latest/).

Execute the following commands to create a new conda environment and install the necessary dependencies:

```bash
git clone git@github.com:openai/whisper.git
conda create -n chatglm3-voice-demo python=3.10
conda activate chatglm3-voice-demo
python3 -m pip install --upgrade pip
conda install -c anaconda portaudio
pip install -r requirements.txt
```

Please note that this project requires Python 3.10 or higher.

Additionally, installing the Jupyter kernel is required for using the Code Interpreter:

```bash
ipython kernel install --name chatglm3-demo --user
```

### Test speech recognition

```bash
python3 speech_recognition_test.py
```

## Execution

Run the following command to load the model locally and start the demo:

```bash
python voiceassistant.py -r {PATH/TO/FOLDER}/whisper-tiny -m {PATH/TO/FOLDER}/chatglm3-6b  -l english
```


# Enjoy!