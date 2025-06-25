import streamlit as st
from predictor import load_model_and_tokenizer, predict_sentiment

st.set_page_config(page_title="Tweet Sentiment Analysis", layout="centered")

# === Session state default menu ===
if "page" not in st.session_state:
    st.session_state.page = "Home"

# === Sidebar manual menu ===
with st.sidebar:
    st.markdown("## Menu")
    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Project Overview"):
        st.session_state.page = "Project Overview"

# === Load Model & Tokenizer ===
model, tokenizer, label_names = load_model_and_tokenizer()

# === Halaman Home ===
if st.session_state.page == "Home":
    st.title("Analisis Sentimen Tweet")
    st.markdown("Masukkan tweet di bawah ini dan lihat prediksi sentimennya. (note: in english)")

    tweet = st.text_area("✍️ Tulis tweet di sini:", placeholder="Contoh: I love this book")

    if st.button("🔍 Analisa Sentimen"):
        if tweet.strip():
            sentiment = predict_sentiment(tweet, model, tokenizer, label_names)
            st.success(f"Sentimen: **{sentiment}**")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

# === Halaman Overview ===
elif st.session_state.page == "Project Overview":
    st.title("📊 Project Overview")

    tab1, tab2, tab3, tab4 = st.tabs(["🧭 Overview", "📊 Data & Model", "📚 Literature", "📝 Paper & Slides"])

    with tab1:
        st.markdown("""
        ### 🧠 Deskripsi Singkat
        Proyek ini bertujuan untuk menganalisis sentimen tweet menggunakan model pre-trained dari Hugging Face.

        - Input: tweet teks pendek
        - Output: kategori sentimen (*Positive, Neutral, Negative*)
        - Framework: Streamlit, PyTorch, Hugging Face Transformers
        - Target user: Mahasiswa, peneliti, pengembang NLP
        """)
        st.markdown("""
        ### 📌 Deskripsi Proyek
        Proyek ini menggunakan model pre-trained dari Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment`) untuk mengklasifikasikan tweet ke dalam tiga kategori sentimen:

        - **Negative**
        - **Neutral**
        - **Positive**

        ### Model
        - Arsitektur: RoBERTa Base
        - Dataset pelatihan: TweetEval (Cardiff NLP)
        - Bahasa: Bahasa Inggris

        ### ⚠️ Catatan
        Karena model ini dilatih menggunakan Bahasa Inggris, kalimat berbahasa Indonesia mungkin tidak selalu diprediksi secara akurat. Gunakan terjemahan untuk hasil optimal.

        ### 🛠️ Teknologi
        - Streamlit
        - Hugging Face Transformers
        - PyTorch
        """)

    with tab2:
        st.markdown("### 📂 Dataset & Model Detail")
        from predictor import (
            load_model_and_tokenizer,
            predict_sentiment,
            load_tweet_dataset,   
            get_label_map         
        )
        # Load dataset via predictor.py
        if "tweet_dataset" not in st.session_state:
            st.session_state.tweet_dataset = load_tweet_dataset()

        ds = st.session_state.tweet_dataset

        # Hitung proporsi dataset
        train_size = len(ds["train"])
        test_size = len(ds["test"])
        total_size = train_size + test_size
        train_pct = train_size / total_size * 100
        test_pct = test_size / total_size * 100

        st.markdown(f"""
        **📊 Statistik Dataset:**
        - Jumlah Data Train: **{train_size}** tweet
        - Jumlah Data Test: **{test_size}** tweet
        """)

        # Tampilkan contoh data
        st.markdown("### 🔍 Contoh Data")
        num_samples = st.slider("Tampilkan berapa contoh data?", 1, 10, 3)

        label_map = get_label_map()
        sample_data = ds["train"].select(range(num_samples))
        for i, row in enumerate(sample_data):
            st.markdown(f"""
            **Contoh {i+1}:**
            - 📝 Tweet: *{row['text']}*
            - 🏷️ Label: **{label_map[row['label']]}**
            """)

        # Informasi model & preprocessing
        st.markdown("### 🤖 Arsitektur Model & Tokenisasi")
        st.markdown("""
        - Model: `cardiffnlp/twitter-roberta-base-sentiment`
        - Arsitektur: `roberta-base` (fine-tuned untuk tweet)
        - Tokenizer: AutoTokenizer dari model yang sama
        - Preprocessing: padding, truncation, special tokens
        - Output: logits → argmax → label
        - Akurasi dasar model (benchmark TweetEval): ~65-70%
        """)

        # Link sumber dataset
        st.markdown("""
        ### 🔗 Sumber Dataset
        - Dataset Hugging Face: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
        """)

    with tab3:
        st.markdown("""
        ### 📚 Literatur Terkait
        - Barbieri et al. (2020). [TweetEval: Unified Benchmark for Tweet Classification](https://arxiv.org/abs/2010.12421)
        - Devlin et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
        - Wolf et al. (2020). [Transformers: State-of-the-Art NLP](https://arxiv.org/abs/1910.03771)

        """)

    with tab4:
        st.subheader("📄 Final Paper")
        st.markdown("Berikut adalah paper penelitian yang ditulis dalam format IEEE berdasarkan proyek ini:")

        with open("PAPER_SentimenAnalysis_AdvDamin.pdf", "rb") as f:
            st.download_button(
                label="📥 Download Paper (PDF)",
                data=f,
                file_name="Sentiment_Analysis_Paper.pdf",
                mime="application/pdf"
            )

        st.divider()
        st.markdown("""
        ### 📝 Paper & Presentasi

        - Paper IEEE Format: [Sharelatex](https://www.overleaf.com/read/jgvvvppjmwyk#119cf3)
        - Slide Presentasi: [Canva](https://www.canva.com/design/DAGlgd6NJcs/13a6Ezvop1ocBAKt0I2WNg/edit?utm_content=DAGlgd6NJcs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

        """)
