import datasets
import random
import re

def do_pretokenized(text):
    text = re.sub(r"\s*(@@|##|[\-\+\(\{\[\]\}\)\.\,\;])\s*", r" \1 ", text)
    return text

def undo_pretokenized(text):
    text = " " + text + " "

    text = re.sub(r"\s+(\@\@)\s+", r" \1", text)
    text = re.sub(r"\s+(\#\#)\s+", r"\1 ", text)
    
    text = re.sub(r"\s+(\-\+)\s+", r"\1", text)
    text = re.sub(r"\s+([\(\{\[])\s+", r" \1", text)
    text = re.sub(r"\s+([\)\}\]])\s+", r"\1 ", text)

    text = re.sub(r"\s+([\.\,\;])\s+", r"\1 ", text)

    return text.strip()


def format_entities(example):

    tokens = []
    is_inside = False

    for token, tag in zip(example["tokens"], example["ner_tags"]):

        if tag == 1:
            is_inside = True
            tokens.append("@@")

        if is_inside and tag == 0:
            is_inside = False
            tokens.append("##")

        tokens.append(token)

    return tokens

def unformat_entities(text):

    tokens = []
    tags = []
    
    current_tag = 0

    for token in text.split():

        if token == "@@":
            current_tag = 1
            continue
        
        if token == "##":
            current_tag = 0
            continue

        tokens.append(token)
        tags.append(current_tag)

        if current_tag == 1:
            current_tag = 2
    
    return tokens, tags


def make_texts(example, text_key="text", tagged_key="tagged_text", text_transforms=[]):

    text =  " ".join(example["tokens"]).strip() 
    tagged_text =  " ".join(format_entities(example)).strip()

    for transform in text_transforms:
        text = transform(text)
        tagged_text = transform(tagged_text)

    example[text_key] = text
    example[tagged_key] = tagged_text

    return example


def get_random_demos(examples, filter_func, k=1):
    return random.choices(list(filter(filter_func, examples)), k=k)

def get_stratified_random_demos(examples, filter_funcs, k=1):
    demos = []
    
    for filter_func in filter_funcs:
        demos += get_random_demos(examples, filter_func, k)

    random.shuffle(demos)

    return demos


def format_multi_turns(example, demos, system_prompt="", user_key="text", assistant_key="tagged_text"):
    turns = [{"role":"system", "content":system_prompt}]
    
    for demo in demos:
        turns.append({"role":"user", "content":demo[user_key]})
        turns.append({"role":"assistant", "content":demo[assistant_key]})
    
    turns.append({"role":"user", "content":example[user_key]})

    return turns


