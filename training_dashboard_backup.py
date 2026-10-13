import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
ADAPTER = ROOT / "adapters" / "adapters.safetensors"

st.set_page_config(
    page_title="OR-RISK-LLAMA | Training Monitor",
    page_icon="🧠",
    layout="wide",
)

def read_dataset(filename):
    path = DATA / filename
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

def extract_risk(record):
    messages = record.get("messages", [])
    answers = [
        m.get("content", "")
        for m in messages
        if m.get("role") == "assistant"
    ]
    text = " ".join(answers).upper()

    match = re.search(
        r"RISCO:\s*(BAIXO|MÉDIO|MEDIO|ALTO)", text
    )
    if not match:
        return "Não identificado"

    label = match.group(1)
    return "MÉDIO" if label == "MEDIO" else label

def read_losses():
    results = {}
    pattern = re.compile(
        r"Iter\s+(\d+):\s*"
        r"(Train|Val|Validation)\s+loss\s+"
        r"([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    for logfile in sorted(REPORTS.glob("*.log")):
        for match in pattern.finditer(
            logfile.read_text(encoding="utf-8", errors="replace")
        ):
            iteration = int(match.group(1))
            kind = match.group(2).lower()
            value = float(match.group(3))

            entry = results.setdefault(
                (logfile.name, iteration),
                {
                    "Execução": logfile.name,
                    "Iteração": iteration,
                    "Training Loss": None,
                    "Validation Loss": None,
                },
            )

            column = (
                "Training Loss"
                if kind == "train"
                else "Validation Loss"
            )
            entry[column] = value

    return pd.DataFrame(list(results.values()))

def last_metric(df, column):
    if df.empty or column not in df:
        return "N/D"
    values = df[column].dropna()
    return f"{values.iloc[-1]:.4f}" if not values.empty else "N/D"

st.title("OR-RISK-LLAMA")
st.caption(
    "Financial Risk Intelligence | "
    "Llama 3.2 1B | Apple MLX | QLoRA"
)

st.divider()

train = read_dataset("train.jsonl")
valid = read_dataset("valid.jsonl")
test = read_dataset("test.jsonl")

adapter_exists = ADAPTER.exists()
adapter_mb = (
    ADAPTER.stat().st_size / (1024 * 1024)
    if adapter_exists else 0
)

losses = read_losses()

if not losses.empty:
    runs = sorted(losses["Execução"].unique())
    selected_run = st.selectbox(
        "Execução de treinamento",
        runs,
        index=len(runs) - 1,
    )
    run_losses = losses[
        losses["Execução"] == selected_run
    ].sort_values("Iteração")
else:
    run_losses = pd.DataFrame()

st.subheader("Indicadores do projeto")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Training Samples", len(train))
c2.metric("Validation Samples", len(valid))
c3.metric("Adapter", f"{adapter_mb:.2f} MB" if adapter_exists else "N/D")
c4.metric("Test Samples", len(test))

c5, c6, c7 = st.columns(3)

c5.metric(
    "Training Loss",
    last_metric(run_losses, "Training Loss")
)
c6.metric(
    "Validation Loss",
    last_metric(run_losses, "Validation Loss")
)
c7.metric(
    "Última iteração registrada",
    int(run_losses["Iteração"].max())
    if not run_losses.empty else "N/D"
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "Desempenho",
    "Datasets",
    "Modelo"
])

with tab1:
    st.subheader("Evolução do treinamento")

    if run_losses.empty:
        st.warning(
            "Nenhum histórico de treinamento encontrado. "
            "Salve os logs em reports/*.log para visualizar "
            "as curvas de treinamento e validação."
        )
    else:
        chart = run_losses.set_index("Iteração")[
            ["Training Loss", "Validation Loss"]
        ]
        st.line_chart(chart)
        st.dataframe(
            run_losses,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "As curvas mostram perdas registradas pelo MLX. "
        "Loss não equivale à acurácia de detecção de fraude."
    )

with tab2:
    st.subheader("Distribuição dos exemplos")

    counts = Counter(
        extract_risk(record)
        for record in train
    )

    if counts:
        distribution = pd.DataFrame(
            [
                {"Classificação": label, "Quantidade": count}
                for label, count in counts.items()
            ]
        )

        st.bar_chart(
            distribution.set_index("Classificação")
        )
        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Nenhum dado de treinamento encontrado.")

    st.subheader("Conjuntos disponíveis")
    st.dataframe(
        pd.DataFrame([
            {"Conjunto": "Treinamento", "Exemplos": len(train)},
            {"Conjunto": "Validação", "Exemplos": len(valid)},
            {"Conjunto": "Teste", "Exemplos": len(test)},
        ]),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("Modelo ajustado")

    config_path = (
        ROOT / "adapters" / "adapter_config.json"
    )

    st.write(
        "**Modelo base:** "
        "mlx-community/Llama-3.2-1B-Instruct-4bit"
    )
    st.write(
        "**Técnica:** QLoRA / adaptação de baixa dimensão"
    )
    st.write(
        "**Pesos ajustados:**",
        "Disponíveis" if adapter_exists else "Não encontrados"
    )

    if config_path.exists():
        config = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        st.json(config)
    else:
        st.warning("Configuração do adapter não encontrada.")

    st.subheader("Avaliação independente")

    if test:
        st.info(
            "Dataset de teste encontrado. "
            "A avaliação de inferência ainda precisa ser executada."
        )
    else:
        st.warning(
            "Ainda não existe test.jsonl. "
            "Crie um conjunto de teste independente para "
            "medir a capacidade de generalização."
        )

    st.write(
        "Próxima etapa: comparar as respostas do modelo "
        "original e do modelo ajustado."
    )

st.divider()
st.caption(
    "OR System | AI Engineering Research | "
    "Métricas extraídas exclusivamente dos arquivos locais"
)
