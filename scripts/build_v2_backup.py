import json
import random
from pathlib import Path

SEED = 42
rng = random.Random(SEED)

OUTPUT = Path("data/v2")
OUTPUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "BAIXO": {
        "train": [
            (
                "Cliente paga R$ {amount} mensalmente ao mesmo fornecedor, "
                "mantendo o histórico de pagamentos.",
                "Pagamento recorrente e compatível com o histórico."
            ),
            (
                "Transferência de R$ {amount} entre contas verificadas "
                "do mesmo titular.",
                "Transferência entre contas do mesmo titular, sem anomalia descrita."
            ),
            (
                "Pagamento de R$ {amount} de uma fatura habitual "
                "a um prestador conhecido.",
                "Operação habitual com beneficiário conhecido."
            ),
        ],
        "valid": [
            (
                "Empresa realiza pagamento recorrente de R$ {amount} "
                "de tributo previsto em seu calendário.",
                "Pagamento previsto e compatível com a atividade informada."
            ),
        ],
        "test": [
            (
                "Cliente compra R$ {amount} em estabelecimento habitual, "
                "dentro de seu padrão histórico.",
                "Operação compatível com o comportamento informado."
            ),
        ],
    },
    "MÉDIO": {
        "train": [
            (
                "Cliente transfere R$ {amount} para um beneficiário "
                "novo, acima de sua média habitual.",
                "Novo beneficiário e desvio do valor habitual."
            ),
            (
                "Conta movimenta R$ {amount} em horário incomum, "
                "sem histórico semelhante.",
                "Horário e valor fora do padrão informado."
            ),
            (
                "Cliente realiza remessa internacional de R$ {amount} "
                "a um destinatário sem histórico de relacionamento.",
                "Ausência de histórico suficiente para avaliar a operação."
            ),
        ],
        "valid": [
            (
                "Empresa paga R$ {amount} por equipamento novo, "
                "valor acima de suas compras habituais.",
                "Compra incomum que requer verificação contextual."
            ),
        ],
        "test": [
            (
                "Cliente cadastra um novo favorecido e realiza "
                "{count} transferências de R$ {amount}, "
                "acima de seu padrão usual.",
                "Novo favorecido e frequência atípica das transferências."
            ),
        ],
    },
    "ALTO": {
        "train": [
            (
                "Conta recém-aberta recebe R$ {amount} e distribui "
                "rapidamente os recursos entre {count} destinatários "
                "sem relacionamento conhecido.",
                "Conta recente, entrada expressiva e dispersão rápida."
            ),
            (
                "Após receber R$ {amount}, uma conta realiza "
                "{count} transferências sucessivas para terceiros "
                "sem justificativa econômica apresentada.",
                "Saídas sucessivas para terceiros após recebimento."
            ),
            (
                "Conta com pouca atividade recebe {count} depósitos "
                "de diferentes origens e encaminha quase todo o valor "
                "para outra conta no mesmo dia.",
                "Concentração e rápida movimentação de recursos."
            ),
        ],
        "valid": [
            (
                "Conta aberta recentemente recebe R$ {amount} de "
                "múltiplas origens e transfere quase todo o saldo "
                "para novos destinatários em menos de uma hora.",
                "Movimentação acelerada e incompatibilidade com o perfil descrito."
            ),
        ],
        "test": [
            (
                "Conta sem histórico comercial relevante recebe "
                "R$ {amount} e faz {count} repasses rápidos "
                "para diferentes terceiros.",
                "Volume elevado e dispersão rápida sem contexto econômico suficiente."
            ),
        ],
    },
}

SIZES = {
    "train": 80,
    "valid": 15,
    "test": 15,
}

RANGES = {
    "BAIXO": (100, 3500),
    "MÉDIO": (2000, 18000),
    "ALTO": (20000, 120000),
}

SYSTEM = (
    "Você é um assistente experimental de triagem de risco financeiro. "
    "Classifique somente o cenário informado em BAIXO, MÉDIO ou ALTO "
    "e apresente uma justificativa breve. "
    "Indicadores de risco não comprovam fraude ou lavagem de dinheiro."
)

def money(value):
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

all_inputs = []

for split, per_class in SIZES.items():
    rows = []

    for label, sections in SCENARIOS.items():
        scenarios = sections[split]
        lower, upper = RANGES[label]

        for i in range(per_class):
            template, reason = scenarios[i % len(scenarios)]

            amount = money(rng.randint(lower, upper))
            count = rng.randint(4, 12)

            question = template.format(
                amount=amount,
                count=count,
            )

            assert question not in all_inputs, "Exemplo duplicado"
            all_inputs.append(question)

            answer = (
                f"Risco: {label}. "
                f"Justificativa: {reason} "
                "Classificação preliminar, sujeita à análise adicional."
            )

            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            })

    rng.shuffle(rows)

    path = OUTPUT / f"{split}.jsonl"

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print(f"{split}: {len(rows)} exemplos -> {path}")

print("Dataset v2 criado com sucesso.")
