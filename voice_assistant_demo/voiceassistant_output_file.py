from langchain import LLMChain, PromptTemplate
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.memory import ConversationBufferWindowMemory
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import speech_recognition as sr
import numpy as np
import pyttsx3
import argparse
import time

english_template = """
{history}
Q: {human_input}
A:"""

chinese_template = """{history}\n\n问：{human_input}\n\n答："""


template_dict = {
    "english": english_template,
    "chinese": chinese_template
}

llm_load_methods = (
    TransformersLLM.from_model_id,
    TransformersLLM.from_model_id_low_bit,
)

def prepare_chain(args):

    llm_model_path = args.llm_model_path

    # Use a easy prompt could bring good-enough result
    # For Chinese Prompt
    # template = """{history}\n\n问：{human_input}\n\n答："""
    template = template_dict[args.language]
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    method_index = 1 if args.directly else 0
    llm = llm_load_methods[method_index](
            model_id=llm_model_path,
            model_kwargs={
                          "trust_remote_code": True},
    )

    # Following code are complete the same as the use-case
    voiceassitant_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        llm_kwargs={"max_new_tokens":args.max_new_tokens},
        memory=ConversationBufferWindowMemory(k=2),
    )

    recog_model_path = args.recog_model_path
    processor = WhisperProcessor.from_pretrained(recog_model_path,local_files_only=True)
    recogn_model = AutoModelForSpeechSeq2Seq.from_pretrained(recog_model_path, load_in_4bit=True)
    recogn_model.config.forced_decoder_ids = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

    return voiceassitant_chain, processor, recogn_model, forced_decoder_ids

def listen(chain, output_file):

    voiceassitant_chain, processor, recogn_model, forced_decoder_ids = chain

    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=10)

        print("Okay, go!")
        while True:
            text = ""
            print("Listening now...")
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=60)
                frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
                print("Recognizing...")
                input_features = processor(frame_data,
                                           sampling_rate=audio.sample_rate,
                                           return_tensors="pt").input_features
                predicted_ids = recogn_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: \n {e}"
                )
                text = unrecognized_speech_text
            
            response_text = voiceassitant_chain.predict(human_input=text, stop="\n\n")
            print(response_text)
            output_file.write("User: " + text + "\n")
            output_file.write("AI: " + response_text + "\n")
            output_file.flush()  # Make sure data is written immediately

def main(args):
    chain = prepare_chain(args)
    with open("conversation.txt", "w") as output_file:
        listen(chain, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDL-LLM Transformer Int4 Langchain Voice Assistant Example')
    parser.add_argument('-r', '--recog-model-path', type=str, required=True,
                        help="the path to the huggingface speech recognition model")
    parser.add_argument('-m','--llm-model-path', type=str, required=True,
                        help='the path to the huggingface llm model')
    parser.add_argument('-x','--max-new-tokens', type=int, default=320,
                        help='the max new tokens of model tokens input')
    parser.add_argument('-l', '--language', type=str, default="english",
                        help='the language to be transcribed')
    parser.add_argument('-d', '--directly', action='store_true',
                        help='whether to load low bit model directly')
    args = parser.parse_args()

    main(args)
