
import streamlit as st
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)

# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Balochi NLP Studio",
    page_icon="🌐",
    layout="wide"
)

# --------------------------------
# DARK THEME
# --------------------------------

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
    width: 100%;
}

h1, h2, h3 {
    color: white;
}

.balochi-input textarea {
    direction: rtl;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# HUGGING FACE MODEL IDs
# --------------------------------

BERT_ID = "khadija123it/mbert-multilingual-ner"

GPT_ID = "khadija123it/balochi-gpt2"

# --------------------------------
# LOAD BERT MODEL
# --------------------------------

@st.cache_resource
def load_bert():

    bert = pipeline(
        "token-classification",
        model=BERT_ID,
        tokenizer=BERT_ID,
        aggregation_strategy="simple",
        device=-1
    )

    return bert

# --------------------------------
# LOAD GPT MODEL
# --------------------------------

@st.cache_resource
def load_gpt():

    tokenizer = AutoTokenizer.from_pretrained(
        GPT_ID
    )

    model = AutoModelForCausalLM.from_pretrained(
        GPT_ID
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return tokenizer, model

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("🌐 Balochi NLP Studio")

page = st.sidebar.radio(
    "SECTIONS",
    [
        "Analyze",
        "Dataset",
        "Evaluation",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Research baseline - v1")

# --------------------------------
# MAIN HEADER
# --------------------------------

st.title("Balochi NLP Studio")

st.caption(
    "Multilingual Understanding and Balochi Text Generation"
)

st.markdown("---")

# =================================
# ANALYZE SECTION
# =================================

if page == "Analyze":

    mode = st.radio(
        "SELECT MODEL",
        [
            "BERT - Named Entity Recognition",
            "GPT - Balochi Text Generation"
        ],
        horizontal=True
    )

    left, right = st.columns([2, 1])

    # --------------------------------
    # BERT MODE
    # --------------------------------

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

                    try:

                        with st.spinner("BERT is analyzing..."):

                            bert = load_bert()

                            result = bert(text)

                        st.markdown("---")

                        st.subheader("MODEL PREDICTION")

                        if result:

                            st.dataframe(
                                [
                                    {
                                        "Word": item["word"],
                                        "Entity": item["entity_group"],
                                        "Confidence": round(
                                            float(item["score"]), 3
                                        )
                                    }
                                    for item in result
                                ],
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "No named entities found."
                            )

                    except Exception as e:

                        st.error(
                            "BERT model loading error: " + str(e)
                        )

                else:

                    st.warning(
                        "Please enter a sentence."
                    )

        with right:

            st.subheader("Read the result thoughtfully")

            st.markdown("""
            **PER - Person**

            Identifies names of people.

            **LOC - Location**

            Identifies places and cities.

            **ORG - Organization**

            Identifies companies and organizations.

            **O - Other**

            Words that are not named entities.
            """)

    # --------------------------------
    # GPT MODE
    # --------------------------------

    else:

        with left:

            st.subheader("Your Balochi text")

            prompt = st.text_area(
                "Enter a Balochi starting sentence",
                placeholder="بلوچی زبان ءَ",
                height=150
            )

            if st.button("Generate Balochi text →"):

                if prompt.strip():

                    try:

                        with st.spinner("GPT is generating..."):

                            tokenizer, model = load_gpt()

                            inputs = tokenizer(
                                prompt,
                                return_tensors="pt"
                            )

                            with torch.no_grad():

                                output = model.generate(
                                    **inputs,
                                    max_new_tokens=60,
                                    do_sample=True,
                                    temperature=0.8,
                                    top_p=0.9,
                                    repetition_penalty=1.2,
                                    no_repeat_ngram_size=3,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id
                                )

                            # Extract only newly generated tokens

                            new_tokens = output[
                                0,
                                inputs["input_ids"].shape[1]:
                            ]

                            answer = tokenizer.decode(
                                new_tokens,
                                skip_special_tokens=True
                            )

                        st.markdown("---")

                        st.subheader("MODEL PREDICTION")

                        st.markdown(
                            '<div style="direction:rtl; text-align:right;">'
                            + prompt + answer +
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.caption(
                            "Generated text should be checked for "
                            "Balochi language accuracy."
                        )

                    except Exception as e:

                        st.error(
                            "GPT model loading error: " + str(e)
                        )

                else:

                    st.warning(
                        "Please enter Balochi text."
                    )

        with right:

            st.subheader("Read the result thoughtfully")

            st.markdown("""
            **Balochi Text Generation**

            GPT reads your Balochi starting sentence.

            It predicts the next words and creates a continuation.

            **Important:**

            Generated text may contain repeated or incorrect words.
            """)

# =================================
# DATASET SECTION
# =================================

elif page == "Dataset":

    st.header("Dataset Information")

    st.subheader("Encoder-Only: BERT")

    st.write(
        "Dataset: WikiANN multilingual NER"
    )

    st.write(
        "Languages: English, Arabic, German, Spanish and French."
    )

    st.write(
        "Task: Person, Location and Organization identification."
    )

    st.markdown("---")

    st.subheader("Decoder-Only: GPT")

    st.write(
        "Dataset: Southern Balochi Instructions"
    )

    st.write(
        "Task: Balochi text completion and generation."
    )

    st.write(
        "Fine-tuning data: Balochi-script text."
    )

# =================================
# EVALUATION SECTION
# =================================

elif page == "Evaluation":

    st.header("Model Evaluation")

    st.subheader("BERT Evaluation")

    st.write(
        "Evaluation metrics: Precision, Recall and F1-score."
    )

    st.write(
        "Task: Multilingual Named Entity Recognition."
    )

    st.markdown("---")

    st.subheader("GPT Evaluation")

    st.write(
        "Evaluation metric: Test loss."
    )

    st.write(
        "Manual evaluation: Language accuracy, "
        "sentence meaning and repetition."
    )

# =================================
# ABOUT SECTION
# =================================

else:

    st.header("About the Project")

    st.write("""
    Balochi NLP Studio is a unified NLP application
    combining two different language models.

    1. Encoder-Only BERT:
       Multilingual text understanding and
       Named Entity Recognition.

    2. Decoder-Only GPT:
       Balochi text completion and generation.

    Both models are connected through Hugging Face
    and deployed using Streamlit.
    """)

    st.caption("Research baseline - v1")

# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.caption(
    "Balochi NLP Studio | Encoder-Only BERT + Decoder-Only GPT"
)
