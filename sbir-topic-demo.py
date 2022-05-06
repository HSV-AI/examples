import streamlit as st
import pandas as pd
import altair as alt
import s3fs
from transformers import GPT2LMHeadModel, GPT2Tokenizer

weights_dir = "localModel"

@st.cache
def load_model(device = 'cpu'):
    model_bucket = "s3://hsv-ai/gpt2-sbir/"
    s3 = s3fs.S3FileSystem()
    s3.get(model_bucket, weights_dir, recursive=True)
    model = GPT2LMHeadModel.from_pretrained(weights_dir)
    return model

def generate_messages(
    model,
    tokenizer,
    prompt_text,
    stop_token,
    length,
    num_return_sequences,
    temperature = 0.7,
    k=20,
    p=0.9,
    repetition_penalty = 1.0,
    device = 'cpu'
):

    MAX_LENGTH = int(10000)
    def adjust_length_to_model(length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length
        
    length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
    return generated_sequences


model = load_model(device='cpu')
tokenizer = GPT2Tokenizer.from_pretrained(weights_dir)


st.image("https://hsv.ai/wp-content/uploads/2022/03/logo_v11_2022.png")
st.title('SBIR Topic Generator')
st.markdown("""
Please forgive the performance of this demo, as the model was trained for free on Google Colab and is currently hosted for free by Streamlit using only CPU resources.

Also - this free web instance only allows for 3 concurrent users, so if you don't mind closing this browser tab after you're done - that would be helpful!
""")
prompt_text = st.text_input('Topic Prompt')
if not prompt_text:
  st.warning('Please enter a prompt to create your topic.')
  st.markdown("""
    Example prompts:

    - Design and develop a family of Acoustic Doppler Current Profilers.
    - Develop innovative solutions for the collection of high resolution, high frame rate, and low distortion imagery.
    - The object is to develop a solid-state switch modulator.
  """)
  st.stop()

temperature = 1.0
k=400
p=0.9
repetition_penalty = 1.0
num_return_sequences = 1
length = 700
stop_token = '|EndOfText|'

topics = generate_messages(
    model,
    tokenizer,
    prompt_text,
    stop_token,
    length,
    num_return_sequences,
    temperature = temperature,
    k=k,
    p=p,
    repetition_penalty = repetition_penalty
)

st.write("Generated Topic:")
st.write(topics[0])