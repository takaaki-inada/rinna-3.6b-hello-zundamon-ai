import asyncio
import itertools
import subprocess
from threading import Thread
from typing import AsyncIterator

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

LORA_WEIGHTS='./models/rinna_3b_zundamon_prof_homu_20230615/checkpoint-800'

TITLE=f'ずんだもんLoRA ({LORA_WEIGHTS})'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# NOTE: transformers 4.30.0.dev0
subprocess.run(f"cp scripts/adapter_config.json {LORA_WEIGHTS}", shell=True)
subprocess.run(f"cp {LORA_WEIGHTS}/pytorch_model.bin {LORA_WEIGHTS}/adapter_model.bin", shell=True)

model_id = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_base_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(peft_base_model, LORA_WEIGHTS, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

streamer = TextIteratorStreamer(tokenizer)

@torch.no_grad()
def inference_func(prompt, max_new_tokens=128, temperature=1.0, repetition_penalty=1.1, top_p=0.9):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    generation_kwargs = dict(
        input_ids=token_ids.to(model.device),
        streamer=streamer,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()


def make_prompt(message, chat_history, max_context_size: int = 10, is_push_history=True):
    if is_push_history:
        contexts = chat_history + [[message, ""]]
        contexts = list(itertools.chain.from_iterable(contexts))
        if max_context_size > 0:
            context_size = max_context_size - 1
        else:
            context_size = 100000
        contexts = contexts[-context_size:]
    # NOTE: chat_historyは一旦使用していない
    prompt = f"""ユーザ: {message}
システム: """
    prompt = prompt.replace('\n', '<NL>')
    return prompt


def interact_func(message, chat_history, max_context_size, max_new_tokens, temperature, top_p, repetition_penalty):
    if chat_history is None:
        chat_history = []
    prompt = make_prompt(message, chat_history, max_context_size)
    _ = inference_func(prompt, max_new_tokens, temperature, top_p=top_p)
    chat_history.append((message, ""))
    return gr.update(value="", interactive=False), chat_history


async def bot(history) -> AsyncIterator[str]:
    prompt = make_prompt(history[-1][0], history, max_context_size=0, is_push_history=False)

    preprocess = False
    for output in streamer:
        await asyncio.sleep(0)
        if not output:
            continue
        # output = output.replace("<NL>", "\n")
        # if not output:
        #     continue
        history[-1][1] += output
        # FIXME: ひどいコード
        if not preprocess:
            # prompt = prompt.replace("<NL>", "\n")
            if len(history[-1][1]) > len(prompt):
                # NOTE: len(chat_history[-1][1]) - len(prompt) でサイズが取得できる
                output = output[-(len(history[-1][1]) - len(prompt)):]
                history[-1][1] = history[-1][1][len(prompt):]
                preprocess = True
            else:
                continue
        yield history


# NOTE: zundamon_fastapi.py から呼び出す
async def generate(message, chat_history=[], max_context_size=10, max_new_tokens=128, temperature=1.0, repetition_penalty=1.1, top_p=0.9) -> AsyncIterator[str]:
    prompt = make_prompt(message, chat_history, max_context_size)
    inference_func(prompt, max_new_tokens, temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    chat_history.append([message, ""])

    preprocess = False
    for output in streamer:
        await asyncio.sleep(0)
        if not output:
            continue
        print(output)
        chat_history[-1][1] += output
        if not preprocess:
            if len(chat_history[-1][1]) > len(prompt):
                # NOTE: len(chat_history[-1][1]) - len(prompt) でサイズが取得できる
                output = output[-(len(chat_history[-1][1]) - len(prompt)):]
                chat_history[-1][1] = chat_history[-1][1][len(prompt):]
                preprocess = True
            else:
                continue
        output = output.replace("<NL>", "\n")
        output = output.replace("</s>", "\n")
        yield output


with gr.Blocks() as demo:
    gr.Markdown(f"# {TITLE}")
    with gr.Accordion("Configs", open=False):
        # max_context_size = the number of turns * 2
        max_context_size = gr.Number(value=10, label="max_context_size", precision=0, visible=False)
        max_new_tokens = gr.Number(value=128, label="max_new_tokens", precision=0)
        temperature = gr.Slider(0.0, 1.9, value=1.0, step=0.1, label="temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="top_p")
        repetition_penalty = gr.Slider(0.0, 5.0, value=1.1, step=0.1, label="repetition_penalty")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    response = msg.submit(interact_func, [msg, chatbot, max_context_size, max_new_tokens, temperature, top_p, repetition_penalty], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True, share=False)
