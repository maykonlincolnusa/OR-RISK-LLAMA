"""
Dataset Generator v3 for OR-RISK-LLAMA
Generates realistic financial compliance and risk intelligence datasets (AML/PLD-FT, Fraud, Routine).
Outputs structured JSON responses for Llama 3.2 Instruct fine-tuning with Apple MLX.
"""

import json
import random
from pathlib import Path

SEED = 42
rng = random.Random(SEED)

OUTPUT_DIR = Path("data/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "Você é o OR-RISK, um sistema de inteligência artificial de triagem e auditoria de risco financeiro "
    "e conformidade bancária (AML/PLD-FT). Analise a transação ou cenário informado e retorne estritamente "
    "um objeto JSON com as seguintes chaves: 'risco' ('BAIXO', 'MÉDIO' ou 'ALTO'), 'score' (número inteiro de 0 a 100), "
    "'fatores_de_risco' (lista de strings), 'tipologia' (descrição da categoria da operação), "
    "'acao_recomendada' (recomendação de governança/compliance) e 'justificativa' (parecer técnico fundamentado)."
)

# Template generators for BAIXO (Low Risk)
LOW_TEMPLATES = [
    (
        "Empresa {empresa} realiza pagamento de folha de salários via TED no valor de R$ {valor:,.2f} referente a {qtd_func} colaboradores homologados.",
        {
            "risco": "BAIXO",
            "score_range": (5, 18),
            "fatores": ["Operação em lote homologada", "Folha de pagamento periódica", "Compatibilidade com porte da empresa"],
            "tipologia": "Folha de Pagamento Corporativa",
            "acao": "Aprovação automática e registro contábil de rotina.",
            "justificativa": "Operação habitual de folha salarial com colaboradores cadastrados e histórico contábil convergente."
        }
    ),
    (
        "Recolhimento de tributo federal {tributo} no valor de R$ {valor:,.2f} com código de barras autenticado pela {empresa}.",
        {
            "risco": "BAIXO",
            "score_range": (2, 12),
            "fatores": ["Guia tributária oficial", "Beneficiário governamental (Receita Federal)", "Recorrência fiscal"],
            "tipologia": "Recolhimento Tributário Oficial",
            "acao": "Liquidação imediata sem necessidade de revisão manual.",
            "justificativa": "Pagamento de obrigação tributária legal com documento oficial e sem desvio econômico."
        }
    ),
    (
        "Transferência de R$ {valor:,.2f} entre contas bancárias de mesma titularidade (mesmo CPF/CNPJ) no banco {banco}.",
        {
            "risco": "BAIXO",
            "score_range": (3, 15),
            "fatores": ["Mesma titularidade", "Origem e destino verificados", "Inexistência de intermediários anônimos"],
            "tipologia": "Transferência Interbancária Mesma Titularidade",
            "acao": "Processamento direto com monitoramento passivo.",
            "justificativa": "Movimentação entre contas do próprio titular com lastro e documentação convergentes."
        }
    ),
    (
        "Pagamento recorrente de fatura de fornecedor homologado {fornecedor} no valor de R$ {valor:,.2f} com nota fiscal emitida há {dias} dias.",
        {
            "risco": "BAIXO",
            "score_range": (8, 20),
            "fatores": ["Fornecedor com histórico ativo", "Documento fiscal verificado", "Valor compatível com contrato"],
            "tipologia": "Pagamento a Fornecedor Regular",
            "acao": "Autorização pelo módulo de contas a pagar.",
            "justificativa": "Fatura amparada por nota fiscal eletrônica em conformidade com o histórico operacional da entidade."
        }
    ),
    (
        "Cliente adquire produtos de consumo pessoal em {estabelecimento} no valor de R$ {valor:,.2f} através de cartão de débito habitual.",
        {
            "risco": "BAIXO",
            "score_range": (4, 16),
            "fatores": ["Padrão de consumo pessoal compatível", "Uso de cartão físico e chip com senha", "Localidade habitual"],
            "tipologia": "Consumo Pessoal Habitual",
            "acao": "Liberação direta pelo emissor do cartão.",
            "justificativa": "Despesa cotidiana compatível com o perfil sócio-ocupacional e canal seguro de autenticação."
        }
    ),
    (
        "Aporte de R$ {valor:,.2f} em fundo de investimento de renda fixa na corretora {corretora}, devidamente registrada na CVM.",
        {
            "risco": "BAIXO",
            "score_range": (5, 19),
            "fatores": ["Entidade regulada pelo Bacen/CVM", "Perfil de investidor compatível", "Origem bancária rastreável"],
            "tipologia": "Aplicação Financeira em Instituição Autorizada",
            "acao": "Efetivação de custódia com reporte regulatório padrão.",
            "justificativa": "Aplicação de capital em instituição financeira autorizada, sem indicativos de ocultação de patrimônio."
        }
    ),
    (
        "Liquidação de fatura de energia elétrica da concessionária {concessionaria} no montante de R$ {valor:,.2f} via débito automático.",
        {
            "risco": "BAIXO",
            "score_range": (1, 10),
            "fatores": ["Concessionária de serviço público", "Débito automático programado", "Valor dentro da média histórica"],
            "tipologia": "Serviços Públicos Essenciais",
            "acao": "Liquidação automática programada.",
            "justificativa": "Despesa operacional essencial de utilidade pública sem qualquer anomalia de transação."
        }
    ),
]

# Template generators for MÉDIO (Moderate Risk)
MEDIUM_TEMPLATES = [
    (
        "Cliente realiza transferência Pix de R$ {valor:,.2f} para novo beneficiário ({beneficiario}) sem histórico prévio de transações.",
        {
            "risco": "MÉDIO",
            "score_range": (45, 62),
            "fatores": ["Novo beneficiário", "Ausência de histórico de relacionamento", "Valor superior ao desvio padrão semanal"],
            "tipologia": "Transferência Atípica para Novo Favorecido",
            "acao": "Retenção preventiva para análise contextual ou confirmação biométrica no app.",
            "justificativa": "Inexistência de histórico prévio entre as partes combinada com valor fora da média dos últimos 30 dias."
        }
    ),
    (
        "Conta corrente registra movimentação de R$ {valor:,.2f} às {hora} horas da madrugada em canal móvel, sem registros históricos nesse intervalo.",
        {
            "risco": "MÉDIO",
            "score_range": (48, 65),
            "fatores": ["Horário não convencional (madrugada)", "Comportamento discrepante da rotina temporal", "Canal remoto"],
            "tipologia": "Anomalia Temporal de Movimentação",
            "acao": "Solicitação de confirmação adicional (2FA / Step-up authentication).",
            "justificativa": "A transação ocorreu em faixa horária com alta taxa de risco sem correspondência nos hábitos usuais do cliente."
        }
    ),
    (
        "Remessa internacional de R$ {valor:,.2f} para empresa {empresa_ext} no país {pais}, com fatura comercial em fase de validação cambial.",
        {
            "risco": "MÉDIO",
            "score_range": (50, 68),
            "fatores": ["Operação de câmbio internacional", "Documentação comprobatória pendente de conferência", "Risco cambial"],
            "tipologia": "Operação de Câmbio em Análise",
            "acao": "Diligência documental prévia antes da liquidação cambial definitiva.",
            "justificativa": "Transferência internacional que exige checagem de fatura comercial e vínculo do contrato de câmbio."
        }
    ),
    (
        "Sequência de {qtd_saques} saques em caixas eletrônicos no total de R$ {valor:,.2f} em agências distintas na mesma região metropolitana.",
        {
            "risco": "MÉDIO",
            "score_range": (52, 69),
            "fatores": ["Múltiplos saques em espécie", "Proximidade do teto diário", "Dispersão física entre caixas"],
            "tipologia": "Fracionamento Preliminar de Saques em Espécie",
            "acao": "Alerta aos analistas de prevenção à fraude e monitoramento de saldo residual.",
            "justificativa": "Volume acumulado de saques em dinheiro vivo próximo ao limite de monitoramento automatizado."
        }
    ),
    (
        "Acesso à conta bancária e tentativa de transação de R$ {valor:,.2f} originada de novo dispositivo celular (IMEI não reconhecido) em cidade distante.",
        {
            "risco": "MÉDIO",
            "score_range": (55, 70),
            "fatores": ["Dispositivo móvel não reconhecido", "Geolocalização atípica", "Tentativa de transferência imediata"],
            "tipologia": "Acesso Atípico por Novo Dispositivo",
            "acao": "Bloqueio preventivo da operação e envio de notificação push de validação de segurança.",
            "justificativa": "Indício de troca de dispositivo combinada com deslocamento físico incompatível com o intervalo de tempo."
        }
    ),
    (
        "Aumento expressivo de 280% no volume financeiro mensal de microempresa {empresa}, totalizando R$ {valor:,.2f} sem atualização no cadastro de faturamento.",
        {
            "risco": "MÉDIO",
            "score_range": (46, 64),
            "fatores": ["Crescimento abrupto de receita", "Defasagem cadastral", "Incompatibilidade provisória de porte fiscal"],
            "tipologia": "Descompasso de Faturamento Declarado",
            "acao": "Notificação ao cliente para atualização de demonstrativo de faturamento (DRE/DEFIS).",
            "justificativa": "Salto operacional substancial sem declaração comprobatória de receita atualizada junto à instituição."
        }
    ),
    (
        "Transação de R$ {valor:,.2f} envolvendo parte relacionada listada como Pessoa Politicamente Exposta (PEP) para prestação de serviços de assessoria.",
        {
            "risco": "MÉDIO",
            "score_range": (58, 72),
            "fatores": ["Envolvimento de Pessoa Politicamente Exposta (PEP)", "Contrato de consultoria", "Escrutínio regulatório reforçado"],
            "tipologia": "Operação com Pessoa Politicamente Exposta (PEP)",
            "acao": "Submissão da operação para aprovação de compliance e comitê de risco.",
            "justificativa": "Operações com partes PEP requerem diligência contínua reforçada segundo circulares do Banco Central."
        }
    ),
]

# Template generators for ALTO (High Risk - AML/Fraud/Smurfing)
HIGH_TEMPLATES = [
    (
        "Conta recebe {qtd_dep} depósitos sucessivos em espécie de R$ 9.800,00 totalizando R$ {valor:,.2f} logo abaixo do teto de comunicação obrigatória e transfere o total em seguida.",
        {
            "risco": "ALTO",
            "score_range": (85, 98),
            "fatores": ["Fracionamento deliberado de depósitos (Structuring/Smurfing)", "Valores logo abaixo do limite de notificação", "Rápida evasão dos recursos"],
            "tipologia": "Lavagem de Dinheiro / Structuring (Smurfing)",
            "acao": "Bloqueio cautelar de segurança e encaminhamento imediato de comunicação ao COAF/SISCOAF.",
            "justificativa": "Padrão clássico de fracionamento de valores em dinheiro para burlar os mecanismos de controle e comunicação do COAF."
        }
    ),
    (
        "Conta recém-criada (menos de 7 dias) recebe R$ {valor:,.2f} em {qtd_entradas} transferências de origens não identificadas e pulveriza 99% do saldo em menos de {minutos} minutos.",
        {
            "risco": "ALTO",
            "score_range": (88, 99),
            "fatores": ["Conta recente sem histórico", "Comportamento de conta de passagem (Money Mule)", "Dispersão instantânea de liquidez"],
            "tipologia": "Conta de Passagem / Mula Financeira (Pass-Through)",
            "acao": "Bloqueio judicial/preventivo das contas de destino e encerramento unilateral de conta.",
            "justificativa": "Características explícitas de conta de passagem ('mula de dinheiro') para lavar recursos ilícitos com pulverização rápida."
        }
    ),
    (
        "CPF registrado como desempregado e sem renda comprovada movimenta R$ {valor:,.2f} no mês, adquirindo bens de luxo e realizando remessas atípicas.",
        {
            "risco": "ALTO",
            "score_range": (82, 96),
            "fatores": ["Grave descompasso patrimonial e de renda", "Ausência de justificativa de origem de fundos", "Aquisição atípica de bens de alto valor"],
            "tipologia": "Grave Descompasso Patrimonial / Ocultação de Bens",
            "acao": "Congelamento temporário para prestação de contas e notificação formal às autoridades de fiscalização.",
            "justificativa": "Disparidade extrema entre a capacidade financeira cadastrada e o fluxo de capital movimentado, sem lastro demonstrável."
        }
    ),
    (
        "Troca de credenciais (senha e telefone) seguida imediatamente de tentativa de transferência via Pix de R$ {valor:,.2f} para conta de terceiro com histórico de fraude.",
        {
            "risco": "ALTO",
            "score_range": (90, 100),
            "fatores": ["Invasão de conta (Account Takeover)", "Alteração prévia de dados cadastrais críticos", "Destinatário com histórico negativo"],
            "tipologia": "Fraude de Invasão de Conta (Account Takeover)",
            "acao": "Interrupção imediata da transação, bloqueio preventivo do acesso e contato de segurança com titular via canal seguro.",
            "justificativa": "Indício contundente de tomada de controle indevido da conta por agentes fraudadores externos."
        }
    ),
    (
        "Remessa não justificada de R$ {valor:,.2f} para entidade sediada em jurisdição considerada paraíso fiscal ou não cooperante segundo o GAFI/FATF.",
        {
            "risco": "ALTO",
            "score_range": (84, 97),
            "fatores": ["Jurisdição de alto risco / Paraíso fiscal (GAFI/FATF)", "Inexistência de lastro comercial ou invoice verossímil", "Ocultação de beneficiário final"],
            "tipologia": "Evasão de Divisas e Ocultação Internacional de Ativos",
            "acao": "Recusa formal da ordem de remessa e reporte urgente ao setor de Prevenção à Lavagem de Dinheiro.",
            "justificativa": "Envio de capital para território com deficiências estratégicas de combate à lavagem de dinheiro sem comprovação fidedigna."
        }
    ),
    (
        "Gateway de pagamentos detecta {qtd_transacoes} tentativas sucessivas de transações de R$ {valor:,.2f} com cartões de crédito diversos em menos de {segundos} segundos.",
        {
            "risco": "ALTO",
            "score_range": (92, 100),
            "fatores": ["Ataque de força bruta / Teste de cartões (Carding)", "Alta velocidade de submissão de requisições", "Origem de IP malicioso"],
            "tipologia": "Ataque Automatizado de Teste de Cartões (Carding)",
            "acao": "Bloqueio do IP no firewall do gateway, invalidação das transações e reporte aos adquirentes envolvidos.",
            "justificativa": "Ataque cibernético automatizado visando testar números de cartões furtados contra a base da credenciadora."
        }
    ),
    (
        "Titular da conta foi listado em base internacional de sanções financeiras (OFAC/CSNU), e tenta liquidar ativos no total de R$ {valor:,.2f}.",
        {
            "risco": "ALTO",
            "score_range": (95, 100),
            "fatores": ["Inclusão em listas restritivas globais (OFAC/CSNU)", "Tentativa de esvaziamento patrimonial", "Risco legal sancionatório imediato"],
            "tipologia": "Violação de Sanções Financeiras Internacionais",
            "acao": "Indisponibilidade imediata de ativos conforme Lei Federal 13.810/2019 e comunicação direta ao Ministério da Justiça.",
            "justificativa": "Pessoa física ou jurídica formalmente sancionada com dever legal de bloqueio cautelar imediato sem aviso prévio."
        }
    ),
]

COMPANIES = [
    "TechLog Soluções Ltda.", "Vanguard Distribuidora S.A.", "AgroBrasil Exportadora", 
    "Alpha Comércio de Eletrônicos", "Horizon Consultoria Financeira", "Delta Logística Integrada", 
    "Minas Metais & Engenharia", "Orbital Serviços Médicos", "Lumina Telecomunicações"
]

ESTABLISHMENTS = [
    "Supermercado Pão de Açúcar", "Drogaria São Paulo", "Posto Ipiranga Centro", 
    "Livraria Cultura", "Restaurante Fogo de Chão", "Carrefour Express", "Kalunga Papelaria"
]

BANKS = ["Itaú Unibanco", "Bradesco", "Banco do Brasil", "Santander", "BTG Pactual", "Nubank", "Inter"]
BROKERS = ["XP Investimentos", "BTG Digital", "Clear Corretora", "Rico Investimentos", "Ágora Investimentos"]
CONCESSIONAIRES = ["Enel Distribuição", "Sabesp", "CPFL Energia", "Comgás", "Claro Telecom", "Vivo Telefônica"]
COUNTRIES = ["Panamá", "Ilhas Virgens Britânicas", "Chipre", "Seychelles", "Belize", "Suíça", "Emirados Árabes Unidos"]
NAMES = [
    "Carlos Eduardo Silva", "Mariana Costa Oliveira", "Roberto Mendonça Filho", 
    "Fernanda Ribeiro Santos", "Lucas Almeida Prado", "Beatriz Nogueira Duarte",
    "Juliana Vasconcelos", "Marcelo Toledo Ramos", "Camila Fagundes Leite"
]
TAXES = ["DARF IRPJ (Código 2088)", "DAS Simples Nacional", "GPS Previdência Social", "DARF CSLL", "FGTS Digital"]

def generate_sample(risk_level: str, rng_gen: random.Random) -> dict:
    if risk_level == "BAIXO":
        tmpl, meta = rng_gen.choice(LOW_TEMPLATES)
        valor = round(rng_gen.uniform(80.0, 150000.0), 2)
        text = tmpl.format(
            empresa=rng_gen.choice(COMPANIES),
            valor=valor,
            qtd_func=rng_gen.randint(12, 350),
            tributo=rng_gen.choice(TAXES),
            banco=rng_gen.choice(BANKS),
            fornecedor=rng_gen.choice(COMPANIES),
            dias=rng_gen.randint(1, 15),
            estabelecimento=rng_gen.choice(ESTABLISHMENTS),
            corretora=rng_gen.choice(BROKERS),
            concessionaria=rng_gen.choice(CONCESSIONAIRES)
        )
    elif risk_level == "MÉDIO":
        tmpl, meta = rng_gen.choice(MEDIUM_TEMPLATES)
        valor = round(rng_gen.uniform(2500.0, 220000.0), 2)
        text = tmpl.format(
            valor=valor,
            beneficiario=rng_gen.choice(NAMES),
            hora=rng_gen.choice(["02:15", "03:40", "04:10", "01:50"]),
            empresa_ext=rng_gen.choice(COMPANIES),
            pais=rng_gen.choice(COUNTRIES),
            qtd_saques=rng_gen.randint(4, 9),
            empresa=rng_gen.choice(COMPANIES)
        )
    else: # ALTO
        tmpl, meta = rng_gen.choice(HIGH_TEMPLATES)
        valor = round(rng_gen.uniform(35000.0, 1850000.0), 2)
        text = tmpl.format(
            qtd_dep=rng_gen.randint(8, 25),
            valor=valor,
            qtd_entradas=rng_gen.randint(6, 30),
            minutos=rng_gen.randint(3, 18),
            qtd_transacoes=rng_gen.randint(15, 60),
            segundos=rng_gen.randint(10, 45)
        )

    score = rng_gen.randint(meta["score_range"][0], meta["score_range"][1])
    
    assistant_json = {
        "risco": meta["risco"],
        "score": score,
        "fatores_de_risco": meta["fatores"],
        "tipologia": meta["tipologia"],
        "acao_recomendada": meta["acao"],
        "justificativa": meta["justificativa"]
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    }

def main():
    print("=" * 60)
    print("OR-RISK-LLAMA: Gerador de Datasets Financeiros v3")
    print("=" * 60)

    # Balanced split sizes
    # Train: 150 per class = 450 total
    # Valid: 30 per class = 90 total
    # Test: 30 per class = 90 total
    splits = {
        "train": {"BAIXO": 150, "MÉDIO": 150, "ALTO": 150},
        "valid": {"BAIXO": 30, "MÉDIO": 30, "ALTO": 30},
        "test": {"BAIXO": 30, "MÉDIO": 30, "ALTO": 30},
    }

    for split_name, counts in splits.items():
        records = []
        # Separate random generator per split to avoid leakage
        split_rng = random.Random(SEED + (100 if split_name == "valid" else 200 if split_name == "test" else 0))
        
        for risk, count in counts.items():
            for _ in range(count):
                sample = generate_sample(risk, split_rng)
                records.append(sample)

        # Shuffle split records
        split_rng.shuffle(records)

        target_file = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(target_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"✓ {split_name}.jsonl gerado: {len(records)} exemplos ({target_file})")

    print("\nDataset v3 gerado com sucesso em data/v3/!")

if __name__ == "__main__":
    main()
