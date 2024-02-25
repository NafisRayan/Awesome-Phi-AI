import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Phi 2 model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
except:
    st.error("Failed to load the model or tokenizer. Please check your internet connection and try again.")
    st.stop()

# Streamlit UI
st.title("Microsoft Phi 2 Streamlit App")
st.markdown("Enter your prompt in the text area below. For example, you can ask for a list of 13 words that have 9 letters.")
prompt = st.text_area("Enter your prompt", "Give me a list of 13 words that have 9 letters.")
temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

if st.button("Generate Output"):
    # Code for generating output based on user input
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature
        )
        output = tokenizer.decode(output_ids[0][token_ids.size(1):])
    st.text("Generated Output:")
    st.write(output)

# Add some basic styling
st.markdown("""
<style>
.stApp {
  background-color: #f5f5f5;
  padding: 20px;
}
.stTextarea {
  font-family: 'Open Sans', sans-serif;
  font-size: 16px;
  line-height: 1.5;
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ddd;
}
.stButton {
  font-family: 'Open Sans', sans-serif;
  font-size: 16px;
  padding: 10px 20px;
  border-radius: 5px;
  border: none;
  background-color: #0077cc;
  color: #fff;
  cursor: pointer;
}
.stButton:hover {
  background-color: #005fa3;
}
</style>
""", unsafe_allow_html=True)
