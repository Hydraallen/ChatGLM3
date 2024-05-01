# ChatGLM3 Web Demo


## 安装

我们建议通过 [Conda](https://docs.conda.io/en/latest/) 进行环境管理。

执行以下命令新建一个 conda 环境并安装所需依赖：

```bash
git clone git@github.com:openai/whisper.git
conda create -n chatglm3-voice-demo python=3.10
conda activate chatglm3-voice-demo
python3 -m pip install --upgrade pip
conda install -c anaconda portaudio
pip install -r requirements.txt
```

请注意，本项目需要 Python 3.10 或更高版本。

此外，使用 Code Interpreter 还需要安装 Jupyter 内核：

```bash
ipython kernel install --name chatglm3-demo --user
```

### 测试speech recognition

```bash
python3 speech_recognition_test.py
```

## 运行

运行以下命令在本地加载模型并启动 demo：

```bash
python voiceassistant.py -r {PATH/TO/FOLDER}/whisper-tiny -m {PATH/TO/FOLDER}/chatglm3-6b  -l english
```


# Enjoy!