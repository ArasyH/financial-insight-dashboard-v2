import os
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --- Load Environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# --- Constants ---
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.55,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# ====== STREAMLIT APP ======
st.set_page_config(page_title="Revenue & Cost Segments Dashboard", layout="wide")
st.title("📊 Revenue & Cost Segments Dashboard")

# ===================== UTILS ===================== #
def fetch_data(endpoint: str, params: dict = None):
    """Generic function to fetch data from Sectors API."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def run_llm(prompt_template: str, data: pd.DataFrame):
    """Format prompt with data and invoke LLM."""
    prompt = PromptTemplate.from_template(prompt_template).format(data=data.to_string(index=False))
    return llm.invoke(prompt).content


def clean_python_code(raw_code: str):
    """Cleans LLM-generated Python code block."""
    return raw_code.strip().strip("```").replace("python", "").strip()

# ===================== SECTIONS ===================== #
def segments_summary(ticker: str):
    """Ringkasan keuangan per segmen dari LLM."""
    st.header(f"🏢 Segments Summary for {ticker}")
    
    raw_data_segments = pd.DataFrame(fetch_data(f"company/get-segments/{ticker}/"))
    # Convert bagian revenue_breakdown ke DataFrame
    data_segments = pd.DataFrame(raw_data_segments["revenue_breakdown"])
    # Tambahkan info symbol & year biar tetap nyambung
    data_segments["symbol"] = raw_data_segments["symbol"]
    data_segments["financial_year"] = raw_data_segments["financial_year"]

    prompt = """
    Anda adalah analis keuangan.
    Berikut adalah breakdown revenue dan cost perusahaan:

    {data}

    Buat insight singkat dalam 3 poin:
    1. Segmen revenue terbesar dan kontribusinya.
    2. Apakah ada risiko ketergantungan pada satu segmen?
    3. Analisis keseimbangan revenue vs cost.
    """

    summary = run_llm(prompt, data_segments)  

    with st.expander("💡 Ringkasan Kinerja Finansial"):
        st.markdown(summary)

    return data_segments

def visualize_segments(ticker: str, data_segments: pd.DataFrame):
    """Generate plot untuk melihat segmen revenue & cost."""
    data_sample = data_segments
    # Convert bagian revenue_breakdown ke DataFrame
    prompt = f""" 
    Anda adalah seorang programmer Python yang ahli dalam visualisasi data.
    Berikut adalah data segmen pendapatan dan biaya perusahaan:
    {data_sample.to_string(index=False)}
    
    Buat sebuah skrip Python menggunakan matplotlib untuk menghasilkan bar plot yang rapih.
    Instruksi:
    - Sumbu X adalah 'source' (segmen pendapatan/biaya)
    - Sumbu Y adalah 'value' dalam triliun IDR (gunakan pembagian value/1e12)
    - Gunakan warna berbeda untuk membedakan setiap segmen
    - Putar label X agar lebih terbaca
    - Beri judul 'Revenue & Cost Segments {ticker}'
    
    Tulis HANYA kode python yang bisa langsung dieksekusi. Jangan sertakan penjelasan apapun.
    Pastikan untuk taruh hasilnya dalam variabel bernama 'fig' dan pastikan untuk mengimpor semua library yang diperlukan.
    """

    code = clean_python_code(llm.invoke(prompt).content)

    with st.expander("📊 Visualisasi Tren Pendapatan"):
        exec_locals = {}
        exec(code, {}, exec_locals)
        st.pyplot(exec_locals["fig"])

def interpret_segments(data_segments: pd.DataFrame):
    """Interpretasi tren keuangan (LLM)."""
    prompt = """
    Anda adalah seorang analis keuangan yang handal.
    Berdasarkan data segmen pendapatan dan biaya berikut (dalam miliar rupiah):

    {data}

    Analisis tren utama yang muncul dari data tersebut, meliputi:
    1. Apakah revenue tersebar (diversifikasi) atau terkonsentrasi pada 1–2 segmen? 
    2. Apakah ada gap besar antara revenue dan cost pada segmen tertentu?
    3. Apakah ada indikasi anomali pada segmen tertentu?
    Sajikan analisis dalam 3 poin utama. Tuliskan dalam bahasa yang singkat, padat, dan jelas.
    """
    interpret = run_llm(prompt, data_segments)

    with st.expander("🔎 Interpretasi Segmen"):
        st.markdown(interpret)

def risk_analysis(data_segments: pd.DataFrame):
    """Analisis risiko keuangan (LLM)."""
    prompt = """
    Anda adalah seorang analis risiko keuangan.
    Berdasarkan data pendapatan dan biaya berikut:

    {data}

    Identifikasi 2-3 potensi risiko utama yang muncul dari pola perbandingan rasio pendapatan dan biaya operasional.
    """
    risks = run_llm(prompt, data_segments)

    with st.expander("⚠️ Analisis Risiko Keuangan"):
        st.markdown(risks)

# ===================== MAIN APP ===================== #
def main():
    ticker = st.text_input("Masukkan Ticker (misalnya: BBRI, ASII, BBCA, HMSP)",  )
    if st.button("🔍 Lihat Insight"):
        data_segments = segments_summary(ticker)
        visualize_segments(ticker, data_segments)
        interpret_segments(data_segments)
        risk_analysis(data_segments)
       


if __name__ == "__main__":
    main()
