"""Streamlit app for generating LinkedIn posts using few-shot LLM prompting."""

import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_post

length_options = ["Short", "Medium", "Long"]
language_options = ["English", "Hinglish"]


@st.cache_resource
def load_few_shot():
    """Load and cache the FewShotPosts instance so it isn't reloaded on every rerun."""
    return FewShotPosts()


def main():
    """Render the LinkedIn Post Generator Streamlit app."""
    st.set_page_config(
        page_title="LinkedIn Post Generator",
        page_icon="💼",
        layout="centered",
    )

    st.markdown("""
        <style>
        /* ── Page background ── */
        .stApp { background-color: #f3f2ef; }

        /* ── Hero banner ── */
        .hero {
            background: linear-gradient(135deg, #0A66C2 0%, #0d8ecf 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.8rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(10,102,194,0.25);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255,255,255,0.85);
            font-size: 1rem;
            margin: 0;
        }

        /* ── Selector card: targets the native Streamlit block container ── */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(
            [data-testid="stSelectbox"]
        ) {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.2rem 1.4rem 1.4rem 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        }

        /* ── Generate button ── */
        .stButton > button {
            background: linear-gradient(135deg, #0A66C2, #0d8ecf);
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 1rem;
            font-weight: 700;
            padding: 0.65rem 0;
            width: 100%;
            letter-spacing: 0.3px;
            transition: opacity 0.2s;
        }
        .stButton > button:hover { opacity: 0.88; }

        /* ── Output card ── */
        .post-output {
            background: #ffffff;
            border-left: 5px solid #0A66C2;
            border-radius: 12px;
            padding: 1.5rem 1.8rem;
            margin-top: 1.2rem;
            white-space: pre-wrap;
            font-size: 0.97rem;
            line-height: 1.7;
            color: #1a1a1a;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        }

        /* ── Section labels ── */
        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #0A66C2;
            margin-bottom: 0.8rem;
        }

        /* ── Streamlit label overrides ── */
        label { font-weight: 600 !important; color: #333 !important; }
        .stSelectbox > div > div { border-radius: 8px !important; }
        textarea { border-radius: 8px !important; font-size: 0.93rem !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero">
            <h1>💼 LinkedIn Post Generator</h1>
            <p>Craft scroll-stopping LinkedIn posts in seconds — powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Selectors ──
    with st.container():
        st.markdown('<div class="section-label">Customize your post</div>',
                    unsafe_allow_html=True)

        fs = load_few_shot()
        tags = sorted(fs.get_tags())

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_tag = st.selectbox("🏷️ Topic", options=tags)
        with col2:
            selected_length = st.selectbox("📏 Length", options=length_options)
        with col3:
            selected_language = st.selectbox("🌐 Language", options=language_options)

    # ── Generate ──
    if st.button("⚡ Generate Post", use_container_width=True):
        with st.spinner("Crafting your post..."):
            post = generate_post(selected_length, selected_language, selected_tag)

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Generated Post</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="post-output">{post}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.text_area("📋 Copy your post", value=post, height=180, label_visibility="visible")


if __name__ == "__main__":
    main()
