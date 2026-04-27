import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

DATASET_PATH = "/smartcity_bigdata_dataset_50000.csv"

print("=" * 65)
print("  ETAPA 1 — EXTRAÇÃO DOS DADOS")
print("=" * 65)

df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])

print(f"\n✔ Dataset carregado com sucesso!")
print(f"  → Registros : {len(df):,}")
print(f"  → Colunas   : {df.shape[1]}")
print(f"\nPrimeiras linhas:")
print(df.head())

print("\n" + "=" * 65)
print("  ETAPA 2 — TRANSFORMAÇÃO DOS DADOS")
print("=" * 65)

print("\n[2.1] Valores nulos por coluna:")
nulos = df.isnull().sum()
print(nulos)
if nulos.sum() == 0:
    print("  ✔ Nenhum valor nulo encontrado.")
else:
    df.dropna(inplace=True)
    print(f"  ⚠ Removidas {nulos.sum()} linhas com valores nulos.")

duplicatas = df.duplicated().sum()
print(f"\n[2.2] Registros duplicados: {duplicatas}")
if duplicatas > 0:
    df.drop_duplicates(inplace=True)
    print(f"  ⚠ {duplicatas} duplicatas removidas.")
else:
    print("  ✔ Nenhuma duplicata encontrada.")

print("\n[2.3] Remoção de valores fisicamente inconsistentes:")
antes = len(df)
df = df[
    (df["vehicle_count"] >= 0) &
    (df["avg_speed_kmh"] >= 0) &
    (df["bus_delay_minutes"] >= 0) &
    (df["temperature_c"].between(-10, 60)) &
    (df["rain"].isin([0, 1])) &
    (df["social_event"].isin([0, 1]))
]
removidos = antes - len(df)
print(f"  Registros removidos por inconsistência: {removidos}")
print(f"  ✔ Registros válidos restantes: {len(df):,}")

df["hora"]        = df["timestamp"].dt.hour
df["dia_semana"]  = df["timestamp"].dt.day_name()
df["mes"]         = df["timestamp"].dt.month
df["periodo_dia"] = pd.cut(
    df["hora"],
    bins=[-1, 5, 11, 17, 20, 23],
    labels=["Madrugada", "Manhã", "Tarde", "Noite", "Noite Alta"]
)

print("\n[2.4] Atributos derivados criados: hora, dia_semana, mes, periodo_dia")

print("\n" + "=" * 65)
print("  ETAPA 3 — ESTATÍSTICAS DESCRITIVAS")
print("=" * 65)

cols_num = ["vehicle_count", "temperature_c", "bus_delay_minutes", "avg_speed_kmh"]
desc = df[cols_num].describe().round(2)
print(desc.to_string())

print("\n[Distribuição de Chuva]")
print(df["rain"].value_counts().rename({0: "Sem Chuva", 1: "Com Chuva"}))

print("\n[Distribuição de Eventos Sociais]")
print(df["social_event"].value_counts().rename({0: "Sem Evento", 1: "Com Evento"}))

print("\n[Regiões presentes no dataset]")
print(df["region"].value_counts())

print("\n" + "=" * 65)
print("  ETAPA 4 — QUESTÕES ANALÍTICAS")
print("=" * 65)

q1 = df.groupby("region")["vehicle_count"].mean().sort_values(ascending=False).round(2)
print("\n[Q1] Fluxo médio de veículos por região:")
print(q1.to_string())

q2 = df.groupby("hora").agg(
    veiculos_medios=("vehicle_count", "mean"),
    velocidade_media=("avg_speed_kmh", "mean")
).round(2)
hora_pico_veiculos   = q2["veiculos_medios"].idxmax()
hora_pico_velocidade = q2["velocidade_media"].idxmin()
print(f"\n[Q2] Hora de pico (mais veículos)      : {hora_pico_veiculos}h")
print(f"     Hora mais lenta (menor velocidade): {hora_pico_velocidade}h")

q3 = df.groupby("rain")[["vehicle_count", "avg_speed_kmh"]].mean().round(2)
q3.index = ["Sem Chuva", "Com Chuva"]
print("\n[Q3] Impacto da chuva:")
print(q3.to_string())

q4 = df.groupby(["region", "social_event"])["vehicle_count"].mean().unstack().round(2)
q4.columns = ["Sem Evento", "Com Evento"]
q4["Variação (%)"] = ((q4["Com Evento"] - q4["Sem Evento"]) / q4["Sem Evento"] * 100).round(1)
print("\n[Q4] Impacto de eventos sociais por região:")
print(q4.sort_values("Variação (%)", ascending=False).to_string())

corr_q5 = df[["vehicle_count", "bus_delay_minutes"]].corr().iloc[0, 1].round(4)
print(f"\n[Q5] Correlação (vehicle_count x bus_delay_minutes): {corr_q5}")
if abs(corr_q5) >= 0.7:
    forca = "forte"
elif abs(corr_q5) >= 0.4:
    forca = "moderada"
else:
    forca = "fraca"
print(f"     → Correlação {forca} {'positiva' if corr_q5 > 0 else 'negativa'}")

q6 = df.groupby("region")["avg_speed_kmh"].mean().sort_values().round(2)
print("\n[Q6] Velocidade média por região (crescente):")
print(q6.to_string())

print("\n" + "=" * 65)
print("  ETAPA 6 — CARGA (Simulação)")
print("=" * 65)

agg_regiao = df.groupby("region").agg(
    total_registros=("sensor_id", "count"),
    media_veiculos=("vehicle_count", "mean"),
    media_velocidade=("avg_speed_kmh", "mean"),
    media_atraso_onibus=("bus_delay_minutes", "mean"),
    proporcao_chuva=("rain", "mean"),
    proporcao_evento=("social_event", "mean")
).round(2)

agg_regiao.to_csv("output_agregado_regiao.csv")
print("  ✔ output_agregado_regiao.csv exportado (simulação Sistema Analítico).")

import json
amostra_mongo = df.copy()
amostra_mongo["timestamp"] = amostra_mongo["timestamp"].astype(str)
registros_mongo = amostra_mongo.to_dict(orient="records")
with open("output_mongodb.json", "w", encoding="utf-8") as f:
    json.dump(registros_mongo, f, ensure_ascii=False, indent=2)
print("  ✔ output_mongodb.json exportado (simulação MongoDB - NoSQL).")