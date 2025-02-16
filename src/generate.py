from transformers import (
    GenerationConfig
)

import torch

default_generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
    
    # do_sample=True,
    # temperature=.7,
    # top_p=.8,
    # top_k=20,
    
    # eos_token_id=tokenizer.eos_token_id,
    # pad_token_id=tokenizer.pad_token_id,    
)

def generate_next_turn(turns, tokenizer, llm, generation_config=default_generation_config):

    # Apply template with the tokenizer. Be careful to return pt tensors on the same device than `llm`.
    input_ids = tokenizer.apply_chat_template(turns, return_tensors='pt').to(llm.device)

    # Generate with llm using the given generation config.
    with torch.no_grad():
        output_ids = llm.generate(
            input_ids,
            generation_config
        )

    answer = tokenizer.decode(output_ids[0, input_ids.shape[1]:-1]).strip()
    turns.append({"role":"assistant", "content":answer})
    
    # Decode and select the answer to return.
    return turns