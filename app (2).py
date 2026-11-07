
import streamlit as st
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)

st.set_page_config(
    page_title="Balochi NLP Studio",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #080b10;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111722;
}

.stButton button {
    background-color: #13877e;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# Hugging Face model repositories

BERT_ID = "khadija123it/mbert-multilingual-ner"



GPT_ID = "khadija123it/balochi-gpt2"


@st.cache_resource
def load_bert():

    return pipeline(
        "token-classification",
        model=BERT_ID,
        tokenizer=BERT_ID,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )


@st.cache_resource
def load_gpt():

    tokenizer = AutoTokenizer.from_pretrained(GPT_ID)

    model = AutoModelForCausalLM.from_pretrained(GPT_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


st.sidebar.title("Balochi NLP Studio")

page = st.sidebar.radio(
    "SECTIONS",
    ["Analyze", "Dataset", "Evaluation", "About"]
)

st.sidebar.caption("Research baseline - v1")

st.title("Balochi NLP Studio")

st.caption(
    "Multilingual understanding and Balochi text generation"
)


if page == "Analyze":

    mode = st.radio(
        "Select Model",
        ["BERT - Named Entity Recognition",
         "GPT - Balochi Text Generation"]
    )

    left, right = st.columns([2, 1])

    if mode.startswith("BERT"):

        with left:

            st.subheader("Your text")

            text = st.text_area(
                "Enter your sentence",
                placeholder="Ali visited Paris and met the Google team.",
                height=150
            )

            if st.button("Analyze text →"):

                if text.strip():

                    with st.spinner("BERT is analyzing..."):

                        model = load_bert()
                        result = model(text)

                    st.subheader("BERT Prediction")

                    if result:

                        st.dataframe([
                            {
                                "Word": item["word"],
                                "Entity": item["entity_group"],
                                "Confidence": round(
                                    float(item["score"]), 3
                                )
                            }
                            for item in result
                        ])

                    else:
                        st.info("No named entities found.")

                else:
                    st.warning("Please enter text.")

        with right:

            st.subheader("Read the result")

            st.write("PER: Person name")

            st.write("LOC: Location")

            st.write("ORG: Organization")

            st.write("O: Other words")


    else:

        with left:

            st.subheader("Balochi text")

            prompt = st.text_area(
                "Enter a Balochi starting sentence",
                placeholder="بلوچی زبان ءِ",
                height=150
            )

            if st.button("Generate Balochi text →"):

                if prompt.strip():

                    with st.spinner("GPT is generating..."):

                        tokenizer, model = load_gpt()

                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt"
                        )

                        with torch.no_grad():

                            output = model.generate(
                                **inputs,
                                max_new_tokens=50,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id
                            )

                        answer = tokenizer.decode(
                            output[0],
                            skip_special_tokens=True
                        )

                    st.subheader("GPT Generated Text")

                    st.write(answer)

                else:
                    st.warning("Please enter Balochi text.")

        with right:

            st.subheader("Read the result")

            st.write(
                "GPT reads your Balochi starting text "
                "and predicts the next words."
            )

            st.write("Training corpus: Southern Balochi.")


elif page == "Dataset":

    st.header("Datasets")

    st.write("BERT: WikiANN multilingual NER dataset.")

    st.write("GPT: Southern Balochi Instructions dataset.")

    st.write("BERT languages: English, Arabic, German, Spanish and French.")

    st.write("GPT training text: Balochi only.")


elif page == "Evaluation":

    st.header("Model Evaluation")

    st.write("BERT: Precision, Recall and F1-score.")

    st.write("GPT: Evaluation loss and manual text quality review.")


else:

    st.header("About the Project")

    st.write(
        "This application combines an Encoder-only "
        "multilingual BERT model and a Decoder-only "
        "GPT model for Balochi text generation."
    )

    st.write("Research baseline - v1")
