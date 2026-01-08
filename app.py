# =========================================
# Streamlit LSTM 감정분류 앱 (무한로딩 방지)
# - Review.xlsx (Sentiment, Review)
# - 학습은 버튼으로 실행 + 캐싱해서 1회만
# =========================================

import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

st.set_page_config(page_title="리뷰 감정 분석", layout="centered")
st.title("📊 리뷰 감정 분석 (LSTM)")
st.caption("Positive / Negative (2-class)")

# ----------------------------
# 0) 경로/데이터 로드
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Review.xlsx"

if not DATA_PATH.exists():
    st.error("❌ Review.xlsx를 레포 루트(app.py 옆)에 올려줘야 해요.")
    st.stop()

df = pd.read_excel(DATA_PATH)

required_cols = {"Sentiment", "Review"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ 필요한 컬럼이 없습니다: {required_cols} / 현재: {list(df.columns)}")
    st.stop()

df = df[["Sentiment", "Review"]].dropna().copy()
df["Sentiment"] = df["Sentiment"].astype(str)
df["Review"] = df["Review"].astype(str)

st.success(f"✅ 데이터 로드 완료: {len(df)} rows")
st.write("라벨 분포:", df["Sentiment"].value_counts().to_dict())

# ----------------------------
# 1) 전처리/불용어
# ----------------------------
stopwords = set([
    "이", "가", "을", "를", "은", "는", "에", "에서", "에게",
    "의", "와", "과", "도", "로", "으로",
    "하다", "되다", "있다", "없다",
    "그", "저", "것", "수",
    "좀", "잘", "매우", "정말", "너무",
    "때문", "같다"
])

def clean_text(text: str) -> str:
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(tokens)

# ----------------------------
# 2) 학습 함수 (캐싱: 한 번만 학습)
# ----------------------------
@st.cache_resource(show_spinner=False)
def train_once(df_in: pd.DataFrame, vocab_size=15000, max_len=120, epochs=4):
    # 준비
    texts = [clean_text(t) for t in df_in["Review"].tolist()]
    labels = df_in["Sentiment"].tolist()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def encode(texts_list):
        seq = tokenizer.texts_to_sequences(texts_list)
        return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    X_train_pad = encode(X_train)
    X_test_pad = encode(X_test)

    # 모델
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, 128, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(le.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # 학습 (너무 길면 cloud에서 답답해져서 epochs 낮게)
    model.fit(
        X_train_pad, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=16,
        verbose=0
    )

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)

    return model, tokenizer, le, (loss, acc), max_len

# ----------------------------
# 3) 학습 버튼 (여기서만 학습 시작)
# ----------------------------
st.subheader("1) 모델 학습")

col1, col2, col3 = st.columns(3)
with col1:
    EPOCHS = st.slider("epochs", 1, 10, 4)
with col2:
    VOCAB_SIZE = st.selectbox("vocab_size", [5000, 10000, 15000, 20000], index=2)
with col3:
    MAX_LEN = st.selectbox("max_len", [80, 100, 120, 150], index=2)

train_btn = st.button("🧠 학습하기 (1회만)")

if train_btn:
    with st.spinner("학습 중... (한 번만 돌고 캐시에 저장됩니다)"):
        model, tokenizer, le, metrics, max_len = train_once(
            df, vocab_size=VOCAB_SIZE, max_len=MAX_LEN, epochs=EPOCHS
        )
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["le"] = le
    st.session_state["max_len"] = max_len
    st.session_state["metrics"] = metrics

# 이미 학습된 캐시가 있으면 자동으로 가져오기 (첫 실행 후부터)
if "model" not in st.session_state:
    st.info("아직 학습 전입니다. 위의 **학습하기** 버튼을 눌러주세요.")
else:
    loss, acc = st.session_state["metrics"]
    st.success(f"✅ 학습 완료 (Test Acc={acc:.3f}, Loss={loss:.3f})")
    st.write("라벨 매핑:", dict(zip(st.session_state["le"].classes_, range(len(st.session_state["le"].classes_)))))

# ----------------------------
# 4) 입력 예측
# ----------------------------
st
