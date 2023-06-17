import itertools

import ctranslate2
import gradio as gr
import torch
from transformers import AutoTokenizer

LORA_WEIGHTS = "./models/rinna_3b_zundamon_prof_homu_20230615_checkpoint-800_ct2"

TITLE=f'[WIP]ずんだもんLoRA ({LORA_WEIGHTS})'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

model_id = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
generator = ctranslate2.Generator(LORA_WEIGHTS, device=device)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

@torch.no_grad()
def inference_func(prompt, max_new_tokens=128, temperature=0.7, repetition_penalty=1.1, top_p=0.9):
    token_ids = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False))
    results = generator.generate_batch(
        [token_ids],
        max_length=max_new_tokens,
        sampling_temperature=temperature,
        #top_p=top_p,
        sampling_topk=10,
        # include_prompt_in_result
        # 長い入力のパフォーマンスを上げるにはFalseを設定するのが良い
        # Falseの場合、promptをdecoderの内部状態の初期化に使われる
        # Trueの場合、promptが結果に含まれるように生成を制約
        include_prompt_in_result=False,
        # beam_size=1でgreedy search
        # beam_size=4,
        repetition_penalty=repetition_penalty,
        # end_token=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        # return_end_token=True,
    )
    output = tokenizer.decode(results[0].sequences_ids[0])
    output = output.replace("<NL>", "\n")
    return output


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
    generated = inference_func(prompt, max_new_tokens, temperature, top_p=top_p)
    chat_history.append((message, generated))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Chat with `rinna/japanese-gpt-neox-3.6b-instruction-ppo`")
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

    msg.submit(interact_func, [msg, chatbot, max_context_size, max_new_tokens, temperature, top_p, repetition_penalty], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
