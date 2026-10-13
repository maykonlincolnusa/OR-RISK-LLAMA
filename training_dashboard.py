import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
ADAPTERS = ROOT / "adapters"

BASE_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

st.set_page_config(
    page_title="OR RISK | AI Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# DESIGN SYSTEM
# -------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: #080F1C;
        color: #E8EEF8;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #0C1626;
        border-right: 1px solid #1E3048;
    }

    h1, h2, h3 {
        color: #F1F5FB !important;
        letter-spacing: -0.7px;
    }

    .hero {
        background: linear-gradient(
            115deg,
            #122840 0%,
            #102135 55%,
            #102C32 100%
        );
        border: 1px solid #24465D;
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #F1F7FF;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #9CB5CC;
        line-height: 1.7;
    }

    .eyebrow {
        color: #39D8BD;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #101D30;
        border: 1px solid #24354B;
        border-radius: 15px;
        padding: 23px;
        margin-bottom: 12px;
        min-height: 142px;
    }

    .metric-label {
        font-size: 12px;
        color: #96A9C0;
        letter-spacing: 0.4px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        color: #F4F8FF;
        margin-top: 10px;
        margin-bottom: 7px;
    }

    .metric-note {
        font-size: 11px;
        color: #36D2B7;
    }

    .section-label {
        font-size: 12px;
        color: #39D8BD;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #223750;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button {
        background: #102E39;
        border: 1px solid #287D83;
        color: #D8FFF6;
        border-radius: 10px;
    }

    .footnote {
        color: #8198B0;
        font-size: 12px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# DATA LOADERS
# -------------------------------------------------

def load_jsonl(filename):
    path = DATA / filename

    if not path.exists():
        return [], 0

    records = []
    errors = 0

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            errors += 1

    return records, errors


def get_messages(record):
    messages = record.get("messages", [])
    return messages if isinstance(messages, list) else []


def get_content(record, role):
    return " ".join(
        str(message.get("content", ""))
        for message in get_messages(record)
        if message.get("role") == role
    )


def get_risk(record):
    answer = get_content(record, "assistant").upper()

    match = re.search(
        r"RISCO\s*:\s*(BAIXO|M[EÉ]DIO|ALTO)",
        answer
    )

    if match:
        value = match.group(1)
        return "MÉDIO" if value == "MEDIO" else value

    return "Não identificado"


def load_training_logs():
    rows = []

    pattern = re.compile(
        r"\bIter(?:ation)?\s*(\d+)\s*:\s*"
        r"(Train|Val(?:idation)?)\s+loss\s*"
        r"[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    if not REPORTS.exists():
        return pd.DataFrame()

    for file in sorted(REPORTS.glob("*.log")):
        records = {}

        lines = file.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        for line in lines:
            match = pattern.search(line)

            if not match:
                continue

            iteration = int(match.group(1))
            kind = match.group(2).lower()
            value = float(match.group(3))

            if iteration not in records:
                records[iteration] = {
                    "Run": file.name,
                    "Iteration": iteration,
                    "Training Loss": None,
                    "Validation Loss": None,
                }

            key = (
                "Training Loss"
                if kind == "train"
                else "Validation Loss"
            )

            records[iteration][key] = value

        rows.extend(records.values())

    return pd.DataFrame(rows)


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101D30",
        plot_bgcolor="#101D30",
        font_color="#AEC0D5",
        height=height,
        margin=dict(l=25, r=25, t=45, b=25),
        legend=dict(
            orientation="h",
            y=-0.23,
        ),
        xaxis=dict(
            gridcolor="#26374D",
            zerolinecolor="#26374D"
        ),
        yaxis=dict(
            gridcolor="#26374D",
            zerolinecolor="#26374D"
        ),
    )

    return fig


def latest_value(df, column):
    if df.empty or column not in df:
        return None

    values = df[column].dropna()

    if values.empty:
        return None

    return float(values.iloc[-1])


# -------------------------------------------------
# LOAD PROJECT STATE
# -------------------------------------------------

train, train_errors = load_jsonl("train.jsonl")
valid, valid_errors = load_jsonl("valid.jsonl")
test, test_errors = load_jsonl("test.jsonl")

logs = load_training_logs()

adapter_file = ADAPTERS / "adapters.safetensors"
config_file = ADAPTERS / "adapter_config.json"

adapter_exists = adapter_file.exists()

adapter_mb = (
    adapter_file.stat().st_size / 1024**2
    if adapter_exists else 0
)

risk_counts = Counter(get_risk(r) for r in train)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.markdown("## ◈ OR SYSTEM")
    st.caption("AI ENGINEERING PLATFORM")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Training Analytics",
            "Dataset Intelligence",
            "Model Registry",
            "Experiments",
        ],
    )

    st.divider()

    st.markdown("**MODEL CONFIGURATION**")

    st.caption("Llama 3.2 1B")
    st.caption("Apple MLX")
    st.caption("QLoRA | 4-bit")

    if adapter_exists:
        st.success("Adapter disponível")
    else:
        st.warning("Adapter não encontrado")

    st.divider()

    st.caption("OR-RISK-LLAMA")
    st.caption("Local AI Research Environment")


# -------------------------------------------------
# HERO
# -------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">OR SYSTEM / AI RESEARCH</div>
        <div class="hero-title">
            Financial Risk Intelligence
        </div>
        <div class="hero-subtitle">
            AI Training Command Center
            <br>
            Llama 3.2 · Apple Silicon · MLX · QLoRA
            <br>
            Experiment tracking, dataset analytics
            and model evaluation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# TRAINING STATE
# -------------------------------------------------

run_data = pd.DataFrame()
selected_run = None

if not logs.empty:
    available_runs = sorted(logs["Run"].unique())

    selected_run = st.selectbox(
        "Training experiment",
        available_runs,
        index=len(available_runs) - 1,
    )

    run_data = (
        logs[logs["Run"] == selected_run]
        .sort_values("Iteration")
    )

train_loss = latest_value(run_data, "Training Loss")
val_loss = latest_value(run_data, "Validation Loss")

train_loss_text = (
    f"{train_loss:.4f}"
    if train_loss is not None else "N/D"
)

val_loss_text = (
    f"{val_loss:.4f}"
    if val_loss is not None else "N/D"
)


# -------------------------------------------------
# KPI COMPONENTS
# -------------------------------------------------

st.markdown(
    '<div class="section-label">MODEL PERFORMANCE</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card(
        "TRAINING SAMPLES",
        str(len(train)),
        "Exemplos para treinamento",
    )

with k2:
    metric_card(
        "VALIDATION SAMPLES",
        str(len(valid)),
        "Conjunto de validação",
    )

with k3:
    metric_card(
        "TRAINING LOSS",
        train_loss_text,
        "Última métrica registrada",
    )

with k4:
    metric_card(
        "VALIDATION LOSS",
        val_loss_text,
        "Última avaliação registrada",
    )


# -------------------------------------------------
# OVERVIEW
# -------------------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="section-label">TRAINING OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.7, 1])

    with left:

        st.subheader("Loss Evolution")

        if run_data.empty:
            st.info(
                "Nenhum log de treinamento foi encontrado. "
                "Os gráficos serão exibidos quando houver "
                "métricas em reports/*.log."
            )
        else:
            fig = go.Figure()

            for column, color in [
                ("Training Loss", "#34D5BD"),
                ("Validation Loss", "#7FA7FF"),
            ]:
                values = run_data.dropna(
                    subset=[column]
                )

                if not values.empty:
                    fig.add_trace(go.Scatter(
                        x=values["Iteration"],
                        y=values[column],
                        mode="lines+markers",
                        name=column,
                        line=dict(
                            color=color,
                            width=3
                        ),
                    ))

            fig.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Loss",
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )

    with right:

        st.subheader("Risk Distribution")

        if risk_counts:
            labels = list(risk_counts.keys())
            values = list(risk_counts.values())

            color_map = {
                "BAIXO": "#34D5BD",
                "MÉDIO": "#FFCB70",
                "ALTO": "#FF758F",
                "Não identificado": "#748399",
            }

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.68,
                        marker=dict(
                            colors=[
                                color_map.get(
                                    label, "#748399"
                                )
                                for label in labels
                            ]
                        ),
                        textinfo="label+percent",
                    )
                ]
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )
        else:
            st.info("Nenhum dataset encontrado.")

    st.markdown(
        '<div class="section-label">SYSTEM STATUS</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "ADAPTER SIZE",
            f"{adapter_mb:.2f} MB" if adapter_exists else "N/D",
            "Parâmetros ajustados",
        )

    with c2:
        metric_card(
            "RECORDED RUNS",
            str(logs["Run"].nunique())
            if not logs.empty else "0",
            "Experimentos com métricas",
        )

    with c3:
        metric_card(
            "TEST ACCURACY",
            "N/D",
            "Avaliação independente pendente",
        )


# -------------------------------------------------
# TRAINING ANALYTICS
# -------------------------------------------------

elif page == "Training Analytics":

    st.subheader("Training Analytics")

    if run_data.empty:
        st.warning(
            "Não há métricas registradas para exibir."
        )
    else:
        fig = go.Figure()

        for column, color in [
            ("Training Loss", "#34D5BD"),
            ("Validation Loss", "#7FA7FF"),
        ]:

            values = run_data.dropna(
                subset=[column]
            )

            if not values.empty:
                fig.add_trace(go.Scatter(
                    x=values["Iteration"],
                    y=values[column],
                    name=column,
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                ))

        fig.update_layout(
            title="Training and Validation",
            xaxis_title="Iteration",
            yaxis_title="Loss",
        )

        st.plotly_chart(
            chart_layout(fig, 470),
            use_container_width=True,
        )

        st.subheader("Training Metrics")

        st.dataframe(
            run_data,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "As métricas são extraídas dos logs reais "
            "produzidos pelo MLX-LM."
        )


# -------------------------------------------------
# DATASET INTELLIGENCE
# -------------------------------------------------

elif page == "Dataset Intelligence":

    st.subheader("Dataset Intelligence")

    counts = pd.DataFrame([
        {"Dataset": "Training", "Samples": len(train)},
        {"Dataset": "Validation", "Samples": len(valid)},
        {"Dataset": "Test", "Samples": len(test)},
    ])

    fig = go.Figure(
        go.Bar(
            x=counts["Dataset"],
            y=counts["Samples"],
            marker_color=[
                "#34D5BD",
                "#7FA7FF",
                "#FFCB70",
            ],
            text=counts["Samples"],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Dataset Composition",
        yaxis_title="Number of samples",
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True,
    )

    st.subheader("Dataset Explorer")

    dataset_name = st.selectbox(
        "Select dataset",
        ["Training", "Validation", "Test"]
    )

    selected = {
        "Training": train,
        "Validation": valid,
        "Test": test,
    }[dataset_name]

    rows = []

    for index, record in enumerate(selected, start=1):
        rows.append({
            "ID": index,
            "Input": get_content(record, "user"),
            "Expected Response": get_content(
                record, "assistant"
            ),
            "Risk": get_risk(record),
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Dataset vazio ou inexistente.")

    st.subheader("Data Quality")

    all_errors = (
        train_errors +
        valid_errors +
        test_errors
    )

    duplicate_inputs = len(train) - len(
        set(get_content(r, "user") for r in train)
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        metric_card(
            "JSON ERRORS",
            str(all_errors),
            "Registros JSON inválidos",
        )

    with q2:
        metric_card(
            "DUPLICATE INPUTS",
            str(duplicate_inputs),
            "Duplicações no treinamento",
        )

    with q3:
        metric_card(
            "UNLABELED RISK",
            str(risk_counts.get("Não identificado", 0)),
            "Exemplos sem classe reconhecida",
        )

    if len(train) < 100:
        st.warning(
            "O conjunto de treinamento ainda é pequeno. "
            "Os resultados não demonstram generalização "
            "para operações financeiras reais."
        )

    if not test:
        st.warning(
            "Ainda não existe um conjunto de teste independente."
        )


# -------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------

elif page == "Model Registry":

    st.subheader("Model Registry")

    st.markdown("### Base Model")

    st.code(BASE_MODEL)

    st.markdown("### Fine-Tuning Configuration")

    st.write("Framework: Apple MLX")
    st.write("Method: QLoRA")
    st.write("Quantization: 4-bit")
    st.write(
        "Adapter status:",
        "Available" if adapter_exists else "Missing"
    )

    if config_file.exists():

        try:
            config = json.loads(
                config_file.read_text(
                    encoding="utf-8"
                )
            )

            st.subheader("Adapter Configuration")
            st.json(config)

        except json.JSONDecodeError:
            st.error("Invalid adapter configuration.")

    if adapter_exists:
        st.success(
            f"Adapter encontrado: {adapter_mb:.2f} MB"
        )

    st.info(
        "O adapter contém os parâmetros ajustados "
        "do modelo. Seu tamanho não representa "
        "uma métrica de qualidade preditiva."
    )


# -------------------------------------------------
# EXPERIMENT TRACKING
# -------------------------------------------------

elif page == "Experiments":

    st.subheader("Experiment Tracking")

    if logs.empty:
        st.warning(
            "Nenhum experimento com logs foi encontrado."
        )

    else:
        summary = []

        for name, group in logs.groupby("Run"):
            group = group.sort_values("Iteration")

            summary.append({
                "Experiment": name,
                "Last Iteration": int(
                    group["Iteration"].max()
                ),
                "Training Loss": latest_value(
                    group, "Training Loss"
                ),
                "Validation Loss": latest_value(
                    group, "Validation Loss"
                ),
            })

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure()

        for name, group in logs.groupby("Run"):

            values = group.dropna(
                subset=["Training Loss"]
            ).sort_values("Iteration")

            if not values.empty:

                fig.add_trace(go.Scatter(
                    x=values["Iteration"],
                    y=values["Training Loss"],
                    mode="lines+markers",
                    name=name,
                ))

        fig.update_layout(
            title="Training Loss by Experiment",
            xaxis_title="Iteration",
            yaxis_title="Training Loss",
        )

        st.plotly_chart(
            chart_layout(fig, 450),
            use_container_width=True,
        )

        st.caption(
            "A comparação usa apenas os logs disponíveis. "
            "Para comparar modelos rigorosamente, "
            "mantenha datasets e protocolos consistentes."
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="footnote">
        OR SYSTEM · FINANCIAL RISK INTELLIGENCE
        <br>
        AI Engineering Research / Local ML Infrastructure
        <br>
        Experimental system — not validated for
        production financial decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
