import pandas as pd
from prophet import Prophet
import os

# --- CONFIGURAÇÃO ---
# Caminho para o teu ficheiro de dados limpo (do projeto anterior)
INPUT_FILE = '../data/input/ecommerce_clean.csv' 
OUTPUT_FILE = '../data/output/forecast_results.csv'
PERIODOS_FUTURO = 26  # Prever 26 semanas (aprox. 6 meses)

def run_forecasting():
    print("🚀 A iniciar Motor de Previsão de Vendas...")

    # 1. CARREGAR DADOS
    # Precisamos de Data (Order Date) e Valor (Price)
    try:
        # Ajusta o nome das colunas conforme o teu CSV original!
        # Assumindo que tens 'order_purchase_timestamp' e 'price'
        df = pd.read_csv(INPUT_FILE)
        print(f"✅ Dados carregados: {len(df)} linhas.")
    except FileNotFoundError:
        print("❌ Erro: Ficheiro de dados não encontrado. Verifica o caminho.")
        return

    # 2. PRÉ-PROCESSAMENTO (Agregação Temporal)
    print("⚙️ A preparar dados para a IA...")
    
    # Converter para Data
    df['date'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # O Prophet EXIGE duas colunas com nomes específicos:
    # 'ds' = Datestamp
    # 'y'  = Valor a prever
    
    # Vamos agrupar as vendas por SEMANA ('W'). 
    # Mês ('M') é bom, mas Semana dá mais detalhe para o gráfico.
    df_grouped = df.set_index('date').resample('W')['price'].sum().reset_index()
    
    # Renomear para o padrão do Prophet
    df_grouped.columns = ['ds', 'y']
    
    # Remover semanas sem vendas ou com valores estranhos (opcional, mas bom para limpeza)
    df_grouped = df_grouped[df_grouped['y'] > 0]

    print(f"   -> Dados agrupados em {len(df_grouped)} semanas de histórico.")

    # 3. TREINAR O MODELO (A Magia do Facebook Prophet)
    print("🧠 A treinar o modelo (isto pode demorar uns segundos)...")
    
    # daily_seasonality=False porque e-commerce foca-se mais em sazonalidade semanal/anual
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_grouped)

    # 4. PREVER O FUTURO
    print(f"🔮 A projetar os próximos {PERIODOS_FUTURO} semanas...")
    
    # Cria um dataframe vazio com as datas futuras
    future = model.make_future_dataframe(periods=PERIODOS_FUTURO, freq='W')
    
    # O modelo preenche as datas com previsões
    forecast = model.predict(future)

    # 5. PREPARAR PARA POWER BI
    # O 'forecast' tem muitas colunas técnicas. Vamos ficar só com as essenciais.
    # yhat = A previsão (linha central)
    # yhat_lower = Cenário Pessimista
    # yhat_upper = Cenário Otimista
    
    final_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Juntar os dados reais originais para termos "Histórico vs Previsão" no mesmo ficheiro
    # Fazemos um merge pela data
    final_df = pd.merge(final_df, df_grouped, on='ds', how='left')
    
    # Renomear para ficar bonito no Power BI
    final_df.columns = ['Data', 'Previsao_Base', 'Cenario_Pessimista', 'Cenario_Otimista', 'Vendas_Reais']
    
    # Pequeno truque: Se houver Venda Real, a Previsão deve ser igual à Real (para o gráfico não ter duas linhas sobrepostas no passado)
    # Ou deixamos as duas para comparar "Real vs Modelo" (Backtesting). Vamos deixar as duas.

    # 6. EXPORTAR
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Sucesso! Ficheiro gerado em: {OUTPUT_FILE}")
    print("   Agora abre o Power BI e carrega este ficheiro.")

if __name__ == "__main__":
    run_forecasting()