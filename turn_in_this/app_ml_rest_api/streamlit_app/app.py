"""Streamlit-інтерфейс до Housing Price API.

Запуск:  streamlit run streamlit_app/app.py
URL API задається змінною середовища API_URL (за замовчуванням http://localhost:8000).
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠")
st.title("Прогноз ціни житла")

with st.sidebar:
    st.caption(f"API: {API_URL}")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=3).json()
        st.write("**Модель:**", info["best_model"])
        m = info["test_metrics"]
        st.metric("R² (test)", f"{m['r2']:.3f}")
        st.metric("MAE", f"{m['mae']:,.0f}")
    except Exception:
        st.warning("API недоступне")

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Площа (кв. фути)", 500, 20000, 5000, step=100)
    bedrooms = st.slider("Спальні", 1, 8, 3)
    bathrooms = st.slider("Ванні", 1, 6, 2)
    stories = st.slider("Поверхи", 1, 5, 2)
    parking = st.slider("Паркомісця", 0, 5, 1)
    furnishing = st.selectbox(
        "Меблювання", ["furnished", "semi-furnished", "unfurnished"]
    )
with col2:
    yn = ["yes", "no"]
    mainroad = st.radio("Біля головної дороги", yn, index=0, horizontal=True)
    guestroom = st.radio("Гостьова кімната", yn, index=1, horizontal=True)
    basement = st.radio("Підвал", yn, index=1, horizontal=True)
    hotwater = st.radio("Підігрів води", yn, index=1, horizontal=True)
    ac = st.radio("Кондиціонер", yn, index=1, horizontal=True)
    prefarea = st.radio("Престижний район", yn, index=1, horizontal=True)

if st.button("Розрахувати ціну", type="primary"):
    payload = {
        "area": int(area),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwater,
        "airconditioning": ac,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishing,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        st.success(f"Прогнозована ціна: **{r.json()['predicted_price']:,.0f}**")
    except Exception as e:
        st.error(f"Помилка запиту до API: {e}")
