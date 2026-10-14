# 🧠 OR-RISK-LLAMA v3.0: Financial Risk Intelligence & Compliance

<div align="center">

[![Llama 3.2](https://img.shields.io/badge/Model-Llama_3.2_1B_Instruct_4bit-blue.svg?logo=meta)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Apple MLX](https://img.shields.io/badge/Framework-Apple_MLX_Metal-black.svg?logo=apple)](https://github.com/ml-explore/mlx)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit_1.50-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Test Accuracy](https://img.shields.io/badge/Accuracy-98.9%25-success.svg)]()
[![F1-Score](https://img.shields.io/badge/Macro_F1-98.9%25-brightgreen.svg)]()
[![AML Miss Rate](https://img.shields.io/badge/ALTO_Miss_Rate-0.0%25-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Hardware-Apple_Silicon-orange.svg)](https://developer.apple.com/metal/)

**Plataforma Autônoma de Triagem, Auditoria de Risco Financeiro e Conformidade Bancária (AML / PLD-FT).**  
*Fine-Tuning com Llama 3.2 (1B) via QLoRA 4-bit no Apple Silicon com Apple MLX e AI Command Center Interativo em Streamlit & Plotly.*

---

</div>

## 📌 Visão Geral

O **OR-RISK-LLAMA** é uma solução de inteligência artificial de ponta desenvolvida para instituições financeiras, fintechs e equipes de conformidade (Compliance / PLD-FT). O sistema analisa transações financeiras em linguagem natural e gera relatórios técnicos estruturados em JSON, contendo score de risco de 0 a 100, identificação de tipologias de lavagem de dinheiro, fatores de risco e recomendações imediatas de governança (aprovação automática, diligência 2FA ou bloqueio com comunicado ao COAF/SISCOAF).

---

## 🚀 Principais Novidades da Versão 3.0

- **Dataset Bancário v3 Expandido**: 630 cenários financeiros balanceados (`train`, `valid`, `test`) cobrindo tipologias complexas de lavagem de dinheiro e fraude:
  - *Structuring / Smurfing* (depósitos fracionados abaixo do teto de comunicação obrigatória do COAF);
  - *Contas de Passagem / Mulas Financeiras* (dispersão instantânea de 99% do saldo em minutos);
  - *Invasão de Conta / Account Takeover* (troca cadastral noturna seguida de Pix imediato);
  - *Violação de Sanções Internacionais* (titular listado em bases OFAC / CSNU);
  - *Anomalias Temporais & Dispositivo* (transferências na madrugada com novos aparelhos/IMEI);
  - *Operações Legítimas de Rotina* (folha de pagamento homologada via eSocial, recolhimento tributário federal DARF/GPS/DAS e faturas habituais).
- **Saída Estruturada em JSON Padrão Produção**: O modelo gera nativamente JSON estrito com `risco`, `score`, `fatores_de_risco`, `tipologia`, `acao_recomendada` e `justificativa`.
- **QLoRA Otimizado no Apple Silicon (MLX)**: Treinado com **LoRA Rank 16**, scale 32 e max sequence length de 512 tokens. A loss de validação convergiu de **2.560** para impressionantes **0.022** com apenas **~1.5 GB de RAM unificada**.
- **Live Risk Auditor (Playground Interativo)**: Interface interativa com velocímetro de risco (Speedometer Plotly), badges de severidade, tags de fatores de risco e exportador JSON para APIs bancárias.
- **Matriz de Confusão & Avaliação Independente**: Acurácia de **98.9%**, Macro F1 de **98.9%** e **0.0% de Falsos Negativos em Risco ALTO** (segurança regulatória máxima).

---

## 🏗️ Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph DataEngine [Motor de Dados & Cenários v3]
        A1[Cenários de Lavagem de Dinheiro / AML] --> D[scripts/build_v3.py]
        A2[Fraudes Pix / Invasão de Conta] --> D
        A3[Rotinas Comerciais & Tributárias] --> D
        D --> B[Datasets JSONL v3\n450 Treino | 90 Validação | 90 Teste]
    end

    subgraph MLXTraining [Apple Silicon MLX QLoRA]
        B -->|training_config_v3.yaml| C[Base: Llama-3.2-1B-Instruct-4bit]
        C -->|QLoRA Rank 16 / Scale 32| T[Treinamento Nativo Metal\n200 Iters · Val Loss 0.022]
        T --> W[adapters_v3/adapters.safetensors\n22 MB LoRA Weights]
    end

    subgraph CommandCenter [OR-RISK AI Command Center 3.0]
        W --> AUD[Live Risk Auditor\nVelocímetro & Parecer Técnico]
        W --> EVAL[Model Evaluation & Confusion Matrix]
        B --> DS[Dataset Intelligence 3.0]
        T --> EXP[Loss Evolution & Telemetria]
    end
```

---

## 📊 Matriz de Desempenho e Comparativo de Versões

| Métrica / Dimensão | Versão 1.0 (Baseline) | Versão 2.0 (Balanced) | **Versão 3.0 (Enterprise Compliance)** |
|:---|:---:|:---:|:---:|
| **Amostras no Dataset** | 9 amostras | 240 amostras | **630 amostras balanceadas** |
| **LoRA Rank / Scale** | Rank 4 / Scale 10 | Rank 8 / Scale 20 | **Rank 16 / Scale 32** |
| **Formato de Resposta** | Texto livre | Texto simples rígido | **JSON Estruturado com Score 0-100** |
| **Training Loss Final** | 0.5340 | 0.0050 | **0.0440** |
| **Validation Loss Final** | 1.1040 | 1.0430 | **0.0220** |
| **Acurácia em Teste Independente** | N/D | 88.9% | **98.9%** |
| **Macro F1-Score** | N/D | 89.2% | **98.9%** |
| **Taxa de Falso Negativo (Risco ALTO)**| N/D | 7.9% | **0.0% (Zero Casos Críticos Perdidos)** |
| **Throughput no Apple Silicon** | ~40 tok/s | ~50 tok/s | **57.7 tokens/segundo** |
| **Consumo de Memória Unificada** | ~1.1 GB | ~1.1 GB | **~1.5 GB (Ideal para Macs de 8GB)** |

---

## 🎯 Exemplo de Saída Estruturada do Modelo v3

### Entrada (Cenário Financeiro Submetido):
> *"Conta bancária de titular individual recebe 18 depósitos sucessivos em dinheiro de R$ 9.800,00 totalizando R$ 176.400,00 logo abaixo do teto de comunicação obrigatória do COAF e transfere o valor integral para conta de terceiro no mesmo dia."*

### Saída Gerada pelo Llama 3.2 (adapters_v3):
```json
{
  "risco": "ALTO",
  "score": 95,
  "fatores_de_risco": [
    "Fracionamento deliberado de depósitos (Structuring/Smurfing)",
    "Valores logo abaixo do teto COAF",
    "Dispersão imediata de liquidez para terceiros"
  ],
  "tipologia": "Lavagem de Dinheiro / Structuring (Smurfing)",
  "acao_recomendada": "Bloqueio cautelar imediato e comunicação obrigatória ao COAF / SISCOAF.",
  "justificativa": "Padrão clássico de fracionamento de valores em dinheiro para burlar os mecanismos de controle e comunicação do COAF, seguido de rápida evasão dos recursos."
}
```

---

## 📂 Estrutura do Repositório

```text
OR-RISK-LLAMA/
├── adapters_v3/               # Pesos LoRA v3 finais treinados (Rank 16, Safetensors 22MB)
│   ├── adapter_config.json    # Parâmetros de arquitetura e hiperparâmetros
│   └── adapters.safetensors   # Pesos Safetensors finetunados
├── adapters_v2/               # Pesos LoRA da versão anterior (v2)
├── data/
│   └── v3/                    # Datasets de alta fidelidade para compliance
│       ├── train.jsonl        # 450 amostras balanceadas de treinamento
│       ├── valid.jsonl        # 90 amostras de validação
│       └── test.jsonl         # 90 amostras de teste independente
├── reports/
│   ├── evaluation_v3.json     # Relatório consolidado com matriz de confusão e latência
│   └── train_v3.log           # Log completo de treinamento MLX (200 iterações)
├── scripts/
│   ├── build_v3.py            # Gerador de cenários sintéticos e taxonomia bancária v3
│   └── evaluate_model.py      # Avaliador de teste, matriz de confusão e benchmarking
├── training_config_v3.yaml    # Configuração de treinamento QLoRA no Apple MLX
├── training_dashboard.py      # AI Command Center 3.0 (Streamlit + Plotly)
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação completa
```

---

## 🛠️ Guia de Instalação e Execução

### 1. Clonar o Repositório

```bash
git clone https://github.com/maykonlincolnusa/OR-RISK-LLAMA.git
cd OR-RISK-LLAMA
```

### 2. Configurar o Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Executar o AI Command Center (Dashboard v3)

```bash
streamlit run training_dashboard.py
```
O dashboard será aberto no navegador em `http://localhost:8501`.

---

## ⚡ Como Treinar Novos Adaptadores no Apple Silicon

Para executar um novo ciclo de fine-tuning QLoRA via terminal com o framework Apple MLX:

```bash
python -m mlx_lm.lora --config training_config_v3.yaml
```

Ou diretamente via comando inline:

```bash
python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data ./data/v3 \
  --adapter-path ./adapters_v3 \
  --iters 200 \
  --batch-size 1 \
  --learning-rate 1.5e-5 \
  --num-layers 8 \
  --lora-parameters '{"rank": 16, "scale": 32.0, "dropout": 0.0}' \
  --max-seq-length 512 \
  --grad-checkpoint \
  --save-every 50 \
  --steps-per-eval 25 \
  --steps-per-report 5
```

---

## 🔍 Execução de Inferência via CLI

Para testar o modelo diretamente no terminal:

```bash
python -m mlx_lm.generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path ./adapters_v3 \
  --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nVocê é o OR-RISK, um sistema de inteligência artificial de triagem e auditoria de risco financeiro e conformidade bancária (AML/PLD-FT). Analise a transação ou cenário informado e retorne estritamente um objeto JSON com as seguintes chaves: 'risco' ('BAIXO', 'MÉDIO' ou 'ALTO'), 'score' (número inteiro de 0 a 100), 'fatores_de_risco' (lista de strings), 'tipologia' (descrição da categoria da operação), 'acao_recomendada' (recomendação de governança/compliance) e 'justificativa' (parecer técnico fundamentado).<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConta recebe 15 depósitos sucessivos em dinheiro de R$ 9.800 logo abaixo do teto COAF e dispersa em minutos.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
  --max-tokens 150
```

---

## 💻 Compatibilidade de Hardware

- **Sistema Operacional**: macOS 14 (Sonoma) ou macOS 15+ (Sequoia)
- **Processador**: Apple Silicon (M1, M2, M3, M4 — Base, Pro, Max ou Ultra)
- **Memória Unificada Mínima**: 8 GB RAM (consumo real em pico de treino: **~1.5 GB**, permitindo uso simultâneo de outras tarefas sem lentidão)

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais informações.
