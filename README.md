# tech_challenge_fase3

# Fine-tuning de Modelo de Linguagem com Unsloth

Este projeto realiza o fine-tuning de um modelo de linguagem de grande porte (LLM) utilizando a biblioteca [Unsloth](https://github.com/unslothai/unsloth), focando na geração de descrições de produtos a partir de títulos.

## Objetivo

Ajustar um modelo Mistral 7B para gerar descrições criativas e atrativas de produtos com base apenas no título, visando aplicações como e-commerce, marketplaces ou catálogos automatizados.

---

## 1. Preparação do Ambiente

O ambiente foi configurado no Google Colab com as seguintes bibliotecas:

```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install triton
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install transformers datasets
```

## 2. Dataset

O dataset contém produtos com dois campos principais:
- `title`: título do produto
- `content`: descrição completa do produto

Foi aplicado um pré-processamento para transformar os dados no formato necessário:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

A instrução foi fixa e voltada para descrição de produtos de forma envolvente e criativa.

## 3. Modelo Utilizado

O modelo base selecionado foi:
```
unsloth/mistral-7b-v0.3-bnb-4bit
```

Este modelo foi carregado em modo 4-bit para redução de uso de memória (ideal para Colab).

## 4. Técnica de Fine-tuning

Utilizamos **LoRA (Low-Rank Adaptation)** para ajustar o modelo de forma leve e eficiente. Os parâmetros utilizados foram:

```python
FastLanguageModel.get_peft_model(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407
)
```

O template Alpaca foi usado para formatar os exemplos:
```
### Instrução:
...
### Título:
...
### Descrição:
...
```

## 5. Treinamento

A etapa de treinamento foi realizada com o `SFTTrainer` da TRL:

```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=60,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs"
)
```

> O número de etapas (`max_steps`) foi limitado para demonstração. Em produção, recomenda-se aumentar conforme o tamanho do dataset.

## 6. Geração de Texto

Após o treinamento, o modelo é capaz de gerar descrições realistas a partir de títulos simples:

**Exemplo de input:**
```
Smartphone Samsung Android 64GB
```
**Saída esperada:**
```
O Smartphone Samsung com 64GB oferece performance ágil, design elegante e uma experiência Android fluida, ideal para quem busca eficiência e estilo no dia a dia.
```

---

## 7. Salvamento

O modelo e o tokenizer foram salvos no Google Drive para uso posterior:
```python
model.save_pretrained("/content/drive/MyDrive/Notebooks/lora_model")
tokenizer.save_pretrained("/content/drive/MyDrive/Notebooks/lora_model")
```

---

## Conclusão

Este pipeline oferece uma base sólida para realizar fine-tuning de LLMs com recursos limitados, como no Google Colab. Pode ser expandido para tarefas como atendimento automatizado, geração de FAQs, resumos, etc.

---

**Autor:** Kauê  
**Tecnologias:** Unsloth · Hugging Face Transformers · LoRA · Colab

