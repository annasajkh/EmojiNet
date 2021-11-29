from minGPT.mingpt.model import GPT, GPTConfig
from dall_e import map_pixels, unmap_pixels, load_model
from mingpt.utils import sample

import matplotlib.pyplot as plt
import torchvision.transforms as T
import streamlit as st

import os

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

@st.cache(allow_output_mutation=True)
def load_data():
    os.system("git clone https://github.com/karpathy/minGPT.git") 
    os.system("gdown --id 1LjvV1oGjPxZULM_RsMlU1Qs-24t-MuOS")
    os.system("gdown --id 1-AiVgEzuozPSE-_ngJ0COYJHY2sHJ6NC")
    return np.load("emoji_dataset.npy").flatten()

data = load_data()


@st.cache(allow_output_mutation=True)
def load_emoji_model():
    mconf = GPTConfig(8192, 1024, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    model.load_state_dict(torch.load("emoji_model.pth"))
    model.eval()
    return model

model = load_emoji_model()

@st.cache(allow_output_mutation=True)
def load_dalle():
    return load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)


dec = load_dalle()


st.title("EmojiNet")
st.text("AI that can generate emoji")

form = st.form(key="submit-form")
emoji_reference_id = form.slider(label="emoji reference id", min_value=0, max_value=14253, value=0)
emoji_reference_token_length = form.slider(label="emoji reference token length", min_value=0, max_value=1024, value=512)
temperature = form.slider(label="Temperature (Controls the \"craziness\" of the text)", min_value=0.01, max_value=1.0, value=0.7)
top_k = form.slider(label="Top K (If nonzero, limits the sampled tokens to the top k values)", min_value=0, max_value=100, value=40)


complete_text = form.form_submit_button("generate")

if complete_text:
    with st.spinner("Generating..."):
        emoji_reference = [i for i in data[32*32*emoji_reference_id:32*32*(emoji_reference_id+1)][:emoji_reference_token_length]]

        x = torch.tensor([8193,*emoji_reference], dtype=torch.long)[None,...].to(trainer.device)
        y = sample(model, x, 1024 - 512 + 1, temperature=temperature, sample=True, top_k=top_k)[0]

        z = y[1:-1].view(1, 32, 32).cuda().long()
        z = F.one_hot(z, num_classes=8192).permute(0, 3, 1, 2).float()

        x_stats = dec(z).float()


        result = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        result = T.ToPILImage(mode='RGB')(result[0])

        st.image(result)