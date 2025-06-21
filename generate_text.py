from .adapters import *
import torch
import pathlib
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
MODULE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "module"

def generate_sample_and_log(model, tokenizer, prompt_str, device, max_gen_tokens=256, temperature=1.0, top_p=0.95):
    model.eval()
    with torch.no_grad():
        prompt_ids = tokenizer.encode(prompt_str)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        eos_token_id = tokenizer.vocab_to_id.get("<|endoftext|>".encode('utf-8'), None)

        gen_ids = model.generate(
            input_tensor,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        full_ids = prompt_ids + gen_ids[0].tolist()
        output_text = tokenizer.decode(full_ids)
        print(output_text)
    model.train()

if __name__ == '__main__':
    vocab_name = 'TinyStories_vocab.pkl'
    merges_name = 'TinyStories_merges.pkl'
    max_gen_tokens = 256
    temperature = 0.8
    top_p = 0.95
    special_tokens = ["<|endoftext|>"]
    prompt_str = "Once upon a time"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)
    model = Transformer_lm.from_pretrained(MODULE_PATH).to(device)
    generate_sample_and_log(model=model,
        tokenizer=tokenizer,
        prompt_str=prompt_str,  
        device=device,
        max_gen_tokens=max_gen_tokens,
        temperature=temperature,
        top_p=top_p,
    )
