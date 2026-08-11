# AfriMedQA Llama Fine-Tuning Pipeline

This project turns the original Colab notebook export into a Colab-first QLoRA
fine-tuning pipeline and a Streamlit chatbot UI backed by the Gemini API.

## What This Contains

- `finetune_llm.py`: training and data-preview pipeline for AfriMedQA.
- `app.py`: Streamlit chatbot that calls Gemini API.
- `FInetune_LLm.ipynb`: cleaned Colab notebook version of the same workflow.
- `requirements.txt`: lightweight dependencies for Streamlit Cloud.
- `requirements-train.txt`: Colab/GPU dependencies for fine-tuning.
- `.streamlit/secrets.toml.example`: required Streamlit secrets.

## Training in Colab

1. Open the cleaned notebook in Colab.
2. Set runtime to GPU.
3. Install dependencies:

```bash
pip install -r requirements-train.txt
```

4. Authenticate with Hugging Face:

```python
from huggingface_hub import notebook_login
notebook_login()
```

5. Preview formatted data:

```bash
python finetune_llm.py preview --max-train-samples 5 --rows 2
```

6. Run a small smoke test before full training:

```bash
python finetune_llm.py train --max-train-samples 16 --num-train-epochs 0.01 --save-steps 10
```

7. Run full training:

```bash
python finetune_llm.py train
```

The final LoRA adapter is saved to:

```text
outputs/final_adapter
```

## Gemini API for Streamlit

The hosted Streamlit app uses Gemini directly, so no Hugging Face paid endpoint
is required for the chatbot UI.

Required secrets:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.0-flash"
```

## Run Streamlit Locally

```bash
streamlit run app.py
```

You can provide secrets through `.streamlit/secrets.toml` locally or environment
variables named `GEMINI_API_KEY` and `GEMINI_MODEL`.

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Add `GEMINI_API_KEY` and optional `GEMINI_MODEL` in app secrets.
4. Deploy.

## Medical Safety

The chatbot is for demonstration and decision-support testing only. It should
not be presented as a replacement for clinicians, emergency care, or local
treatment guidelines.
