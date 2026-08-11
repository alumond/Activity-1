# AfriMedQA Llama Fine-Tuning Pipeline

This project turns the original Colab notebook export into a Colab-first QLoRA
fine-tuning pipeline and a Streamlit chatbot UI.

## What This Contains

- `finetune_llm.py`: training and data-preview pipeline for AfriMedQA.
- `app.py`: Streamlit chatbot that calls an external Hugging Face inference endpoint.
- `FInetune_LLm.ipynb`: cleaned Colab notebook version of the same workflow.
- `requirements.txt`: dependencies for Colab training and Streamlit app execution.
- `.streamlit/secrets.toml.example`: required Streamlit secrets.

## Training in Colab

1. Open the cleaned notebook in Colab.
2. Set runtime to GPU.
3. Install dependencies:

```bash
pip install -r requirements.txt
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

## Hugging Face Endpoint

Streamlit Community Cloud should host the UI only. Deploy the fine-tuned model
or adapter through Hugging Face infrastructure, then copy the endpoint URL into
Streamlit secrets.

Required secrets:

```toml
HF_ENDPOINT_URL = "https://your-huggingface-endpoint-url"
HF_TOKEN = "hf_your_token_here"
```

## Run Streamlit Locally

```bash
streamlit run app.py
```

You can provide secrets through `.streamlit/secrets.toml` locally or environment
variables named `HF_ENDPOINT_URL` and `HF_TOKEN`.

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Add `HF_ENDPOINT_URL` and `HF_TOKEN` in app secrets.
4. Deploy.

## Medical Safety

The chatbot is for demonstration and decision-support testing only. It should
not be presented as a replacement for clinicians, emergency care, or local
treatment guidelines.
