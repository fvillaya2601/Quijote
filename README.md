Don_Quijote_Streamlit_App_Qwen
=================================

Contenido:
- app.py : Streamlit app conectada directamente a **Qwen-7B** vía Hugging Face.
- requirements.txt : dependencias necesarias.
- labeled_sentences.csv : dataset extendido de frases etiquetadas (Narrador, Don Quijote, Sancho Panza, Otros).
- don_quijote.txt : texto completo (inclúyelo en la carpeta).

Instrucciones:
1. Coloca `don_quijote.txt` en la misma carpeta.
2. Exporta tu token de Hugging Face:
   ```bash
   export HF_TOKEN=tu_token
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta:
   ```bash
   streamlit run app.py
   ```

### Fine-tune Notebook (ejemplo en Colab)
```python
!pip install -q transformers datasets accelerate peft bitsandbytes

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

# Dataset desde don_quijote.txt
with open("don_quijote.txt","r",encoding="utf-8") as f:
    text = f.read()
samples = [{"text": s} for s in text.split(".") if len(s.split())>8]
dataset = Dataset.from_list(samples).train_test_split(test_size=0.1)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
tokenized = dataset.map(tokenize, batched=True)

config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="qwen-finetuned-quijote",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10
)

trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized["train"],
                  eval_dataset=tokenized["test"])
trainer.train()
```
