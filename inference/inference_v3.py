from pathlib import Path
import os
import torch
import random
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from transformers import PreTrainedTokenizerFast
import typer
from scipy.spatial.transform import Rotation

def decode_v1(output, tokenizer):
    tokens, ldr_lines, line, text = [], [], [], ""
    
    #creating lines
    for token_id in output[0]: tokens.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False, clean_up_tokenization_spaces=False))
    for token in tokens:
        if token == "\n" or token[0] == "<":
            line.append(token)
            ldr_lines.append(line)
            line = []
        else: line.append(token)
        
    #formatting lines
    for ldr_line in ldr_lines:
        #class token
        if ldr_line[0][0] == "<": 
            text += ldr_line[0] + "\n"
            continue

        #normal brick line
        for i in range(len(ldr_line)):
            token = ldr_line[i]
            next = "" if i != len(ldr_line) else ldr_line[i+1]
            #1_
            if i == 0 and ldr_line[i] == '1': text += '1' + " "
            #colour_
            elif i == 1: text += token + " "
            #potential -
            elif ldr_line[i] == "-": text += "-"    
            #(-)int(.000)
            elif '.' in next: text += token
            #.000 or .dat
            elif '.' in token: text += token + " "
            #\n
            else: text += token   

    return text

def decode_v3(output, tokenizer):
    tokens, ldr_lines, line, text = [], [], [], ""
    
    #creating lines
    for token_id in output[0]: tokens.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False, clean_up_tokenization_spaces=False))
    for token in tokens:
        if token == "\n" or token[0] == "<":
            line.append(token)
            ldr_lines.append(line)
            line = []
        else: line.append(token)
        
    #formatting lines
    for ldr_line in ldr_lines:
        #class token
        if ldr_line[0][0] == "<": 
            text += ldr_line[0] + "\n"
            continue

        #normal brick line
        for i in range(len(ldr_line)):
            token = ldr_line[i]
            next = '' if i == len(ldr_line)-1 else ldr_line[i+1]
            #1_
            if i == 0 and token == '1': text += '1' + " "
            #colour_ 
            elif i==1: text += token + ' '
            #potential -
            elif token == "-": text += "-"    
            #(-)int(.000) or \n
            elif (next != '' and next[0] == '.') or token=='\n': text += token
            #.000 or .dat or x y z
            else: text += token + " "

    return text

def valid_ldr(text):
    ldr = ""
    lines = text.split("\n")

    for line in lines:
        entries = line.split(" ")
        #class token
        if entries[0] == '': continue

        if entries[0][0] == "<":
            ldr += '0 ' + entries[0] + "\n"
            continue

        #invalid num entries
        if len(entries) != 11: 
            print(f"invalid line: {line}")
            continue
        
        #if valid line
        quats = [float(entries[5]), float(entries[6]), float(entries[7]), float(entries[8])]
        rotation = Rotation.from_quat(quats)
        rot_matrix = rotation.as_matrix().tolist()
        rot_matrix = rot_matrix[0] + rot_matrix[1] + rot_matrix[2]

        for i in range(len(rot_matrix)):
            float_val = float(rot_matrix[i])
            rot_matrix[i] = f"{float_val:.6f}"

        ldr += " ".join(entries[0:5] + rot_matrix + entries[9:]) + "\n"
    
    print(ldr)
    return ldr

def load_ldr_data(ldr_dir: Path, num_augments_per_file, is_eval_set):
    """
    Loads all LDR files from the ldr_dir and calls process_file() to augement them
    """
    src_files = list(ldr_dir.glob("*.ldr")) + list(ldr_dir.glob("*.mpd"))
    all_lines = []

    #setting upper limit for augmenting evaluation dataset
    if is_eval_set: 
        print("\n--- LOADING EVAL LINES ---")
        num_augments_per_file = min(num_augments_per_file, 1)
    else: print("--- LOADING TRAIN LINES ---")

    #iterating through files and augmenting
    for src_file in src_files:
        #determing class label
        label_token = "<|ANY|>"
        if 'creature' in src_file.name: label_token = "<|CR|>"
        elif 'building' in src_file.name: label_token = "<|BU|>"
        elif 'nature' in src_file.name: label_token = "<|NA|>"
        elif 'vehicle' in src_file.name: label_token = "<|VE|>"
        
        #augmenting & appending file
        with src_file.open('r') as file:
            file_lines = file.readlines()
            process_file(file_lines, all_lines, num_augments_per_file, label_token)

    print(f"Total Lines: {len(all_lines)}")
    return all_lines

def process_file(file_lines, all_lines, num_augments_per_file, label_token, bricks_per_window=75):
    #worst case 25 tokens per brick: 2048 / 25 = 81 bricks, rounding to 75 for extra safety
    """
    Cleans up LDR by:
    -removing metadata/comments, rounds floats to 3 decimal places
    -adds EOS token and label token
    -converts rotation matrix to quaternions
    Creates multiple versions of a file:
    -shuffles brick lines (except first pass)
    -adds translation for entire assembly (except first pass)
    """

    processed_file_lines = []
    #determining transation for file augment
    
    for line in file_lines:
        entries = line.split()
        if len(entries) == 0: continue
        if entries[0] != '1': continue
        # Round LDU x,y,z unit offset from origin to LDU integer unit
        for j in range(2, 5):
            coord = float(entries[j])
            entries[j] = str(round(coord))
        #converting rotation matrix to quaternions
        rot_matrix = [
            [float(entries[5]), float(entries[6]), float(entries[7])],
            [float(entries[8]), float(entries[9]), float(entries[10])],
            [float(entries[11]), float(entries[12]), float(entries[13])]
        ]
        rotation = Rotation.from_matrix(rot_matrix)
        quaternions = rotation.as_quat().tolist()
        for j in range(len(quaternions)):
            quat = float(quaternions[j])
            quaternions[j] = f"{quat:.3f}"

        #creating and adding processed line to curr file lines
        final_entries = entries[0:5] + quaternions + entries[14:]
        final_processed_line = " ".join(final_entries) + " \n"
        processed_file_lines.append(final_processed_line)
    
    all_lines.append(" ".join([label_token] + processed_file_lines))
     
def main(
    vlads_machine: bool = False,
    trained_tokenizer_dir: Path = Path('tokenizers/v3/trained_tokenizer_overfit'),
    trained_model_dir: Path = Path('models/v3/trained_model_overfit/final'),
):
    
    #Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer_dir)
    print(f"Tokenizer Loaded: {trained_tokenizer_dir.name}\n")
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not vlads_machine and torch.backends.mps.is_available(): torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(trained_model_dir).to(device)
    print(f"Model Loaded: {trained_model_dir.name}\n")

    #PROMPT AND CONFIG
    text_prompt = """<|VE|>"""

    #IF WANT TO PROMPT WITH BRICKS NEED TO ENSURE THEY ARE IN SAME FORMAT AS WHEN TRAINING
    # test_assemblies = load_ldr_data(Path("data/test"), 1, True)
    # rand_index = random.randint(0, len(test_assemblies) - 1)
    # print('potential prompt: \n', test_assemblies[rand_index])
    prompt = torch.tensor([tokenizer.encode(text_prompt)], dtype=torch.long).to(device)
    output_tokens = prompt.clone().detach().cpu().numpy().tolist()[0]

    no_EOS_generated = True
    max_tokens_per_step = 512
    i = 0
    while no_EOS_generated and i < 5:
        #i += 1
        generation_config = GenerationConfig(
            max_length=len(prompt[0]) + max_tokens_per_step,  # Ensure length is dynamic
            max_new_tokens=max_tokens_per_step,  # Generate up to 1024 tokens at a time
            do_sample=True,
            top_k=50,
            top_p=0.8,  # Adjust for more diversity
            repetition_penalty=1.1,  # Encourage diverse generations
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        print(f"GENERATING UP TO {max_tokens_per_step} TOKENS...\n")

        # Generate the next chunk of tokens
        out = model.generate(prompt, generation_config=generation_config)
        new_tokens = out[0].tolist()[len(output_tokens):]  # Extract only new tokens generated
        print("New Tokens:", len(new_tokens))
        # Append new tokens to the output list
        output_tokens.extend(new_tokens)
        print("\nTOTAL NUMBER OF TOKENS GENERATED:")
        print(len(output_tokens))
        if(len(new_tokens) == 0):
            break

        # Print generated tokens
        for new_token in new_tokens:
            #decoded_token = tokenizer.decode(new_token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            #print(f"Generated token: {decoded_token}")

            # Check if the EOS token is generated
            if new_token == tokenizer.eos_token_id:
                no_EOS_generated = False  # Stop generation if EOS token is generated
                break

        # Update prompt with the last 1024 tokens for the next iteration if not EOS
        if no_EOS_generated:
            max_tokens = 2048 -  max_tokens_per_step # Maximum allowed length for promp (should be 2048 - generation max)
            prompt_length = min(len(output_tokens), max_tokens)  # Use only the available number of tokens up to 1540
            prompt = torch.tensor([output_tokens[-prompt_length:]], dtype=torch.long).to(device)
    #decoding tokens
    formatted_output = decode_v3(out, tokenizer)    

    #CONVERTING TO VALID LDR
    print("\nTOTAL NUMBER OF TOKENS GENERATED:")
    print(len(output_tokens))
    for token in output_tokens:
        print(token)
    print("\nFINAL LDR:")
    ldr = valid_ldr(formatted_output)

    #saving ldr in .ldr file
    with open("output/output.ldr", "w") as f:
        f.write(ldr)
    
if __name__ == "__main__":
    typer.run(main)