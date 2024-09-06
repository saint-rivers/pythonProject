# Masked Language Model

Followed MLM tutorial at https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling

Flask tutorial at https://dev.to/victor_isaac_king/a-step-by-step-guide-to-deploying-a-machine-learning-model-in-a-docker-container-nkp

---

Run server with:
```bash
python3 main.py
```

---

Sample HTTPie Request

```bash
http POST http://localhost:9696/predict text='Mathematics is really <mask>.'

http POST http://localhost:9696/predict text='The milky way is a <mask> galaxy.'
```

---

Pre-trained Model Repo:

https://huggingface.co/saintrivers/test_eli5/tree/main