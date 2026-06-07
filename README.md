# LinkedIn Post-Generator
An AI-powered tool that generates scroll-stopping LinkedIn posts tailored by topic, length, and language — using few-shot prompting with real examples from a curated post dataset.

# Features
- Topic-aware generation — choose from unified tags extracted from real LinkedIn posts
- Length control — Short (1–5 lines), Medium (6–10), or Long (11–15)
- Bilingual support — English or Hinglish (Hindi + English mix)
- Few-shot prompting — injects up to 2 real matching examples to guide the model's style
- Preprocessing pipeline — extracts metadata and unifies tags from raw post data via LLM
- Clean UI — minimal, intuitive Streamlit interface built for speed

## 🛠️ Built With

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM Orchestration | [LangChain](https://langchain.com) |
| LLM Inference | [Groq](https://groq.com) (Llama 3) |
| Data Processing | [Pandas](https://pandas.pydata.org) |
| Language | Python 3.10+ |


## Architecture

```
raw_posts.json
      │
      ▼
 preprocess.py          ← LLM extracts metadata (tags, language, line_count)
      │                    and unifies tags across all posts
      ▼
processed_posts.json
      │
      ▼
  few_shot.py           ← Filters posts by topic + length + language
      │
      ▼
post_generator.py       ← Builds prompt with instructions + few-shot examples
      │
      ▼
   call_llm.py          ← Groq LLM inference (Llama via LangChain)
      │
      ▼
    main.py             ← Streamlit UI
```