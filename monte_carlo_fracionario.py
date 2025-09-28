"""
Monte Carlo com Cálculo Fracionário - Código Completo Integrado
Combina análise tradicional com detecção de memória longa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Importações essenciais
try:
    import yfinance as yf
except ImportError:
    print("Instalando yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf

try:
    from hurst import compute_Hc
except ImportError:
    print("Instalando hurst...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'hurst'])
    from hurst import compute_Hc

from numpy import linalg as LA
from scipy import stats

# ============================================================================
# CONFIGURAÇÕES PRINCIPAIS
# ============================================================================

# Parâmetros da simulação
lista_acao = ['VALE3.SA', 'ITUB4.SA', 'BBDC3.SA', 'WEGE3.SA', 'PETR4.SA']
data_final = dt.datetime.now()
data_inicial = data_final - dt.timedelta(days=300)

numero_simulacoes = 10000
dias_projetados = 252 * 3  # 3 anos
capital_inicial = 1000

# Seed para reprodutibilidade
np.random.seed(42)

# ============================================================================
# PARTE 1: COLETA DE DADOS ROBUSTA
# ============================================================================

print("="*60)
print("COLETANDO DADOS DO MERCADO")
print("="*60)

def baixar_dados_yahoo(tickers, inicio, fim):
    """
    Baixa dados do Yahoo Finance de forma robusta
    """
    try:
        # Baixa dados
        dados = yf.download(
            tickers,
            start=inicio,
            end=fim,
            auto_adjust=True,
            progress=False
        )
        
        # Se for apenas um ticker, reorganiza
        if len(tickers) == 1:
            precos = dados['Close'].to_frame()
            precos.columns = tickers
        else:
            # Para múltiplos tickers
            if 'Close' in dados.columns.get_level_values(0):
                precos = dados['Close']
            else:
                precos = dados
                
        return precos.dropna()
        
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        # Gera dados simulados para teste
        print("Gerando dados simulados para demonstração...")
        dates = pd.date_range(start=inicio, end=fim, freq='D')
        data = {}
        for ticker in tickers:
            # Simula preços com tendência e volatilidade
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            data[ticker] = prices
        return pd.DataFrame(data, index=dates)

# Baixa os dados
precos = baixar_dados_yahoo(lista_acao, data_inicial, data_final)
print(f"Dados coletados: {len(precos)} dias para {len(lista_acao)} ações")

# Calcula retornos
retornos = precos.pct_change().dropna()

# ============================================================================
# PARTE 2: ANÁLISE DE HURST (MEMÓRIA LONGA)
# ============================================================================

print("\n" + "="*60)
print("ANÁLISE DE MEMÓRIA DAS AÇÕES (HURST)")
print("="*60)

def calcular_hurst_robusto(serie, nome="Série"):
    """
    Calcula Hurst de forma robusta
    """
    try:
        # Remove NaN e garante array numpy
        serie_limpa = pd.Series(serie).dropna().values
        
        if len(serie_limpa) < 50:
            print(f"⚠️ {nome}: Poucos dados para Hurst confiável")
            return 0.5
            
        # Calcula Hurst
        H, c, data = compute_Hc(serie_limpa, kind='price')
        
        return float(H)
        
    except Exception as e:
        print(f"Erro calculando Hurst para {nome}: {e}")
        return 0.5

# Calcula Hurst para cada ação
hurst_values = {}
usar_fracionario = False

for acao in lista_acao:
    if acao in precos.columns:
        H = calcular_hurst_robusto(precos[acao], acao)
        hurst_values[acao] = H
        
        # Interpretação
        if H > 0.55:
            interpretacao = "Persistência (tendência)"
            usar_fracionario = True
        elif H < 0.45:
            interpretacao = "Anti-persistência (reversão)"
            usar_fracionario = True
        else:
            interpretacao = "Aleatório"
        
        print(f"{acao}: H = {H:.3f} - {interpretacao}")

# Média de Hurst
media_hurst = np.mean(list(hurst_values.values()))
print(f"\nHurst médio da carteira: {media_hurst:.3f}")

if usar_fracionario:
    print("\n⚠️ DETECTADA MEMÓRIA LONGA - Usando simulação fracionária!")
    print("Projeções serão mais realistas considerando persistência/reversão")
else:
    print("\n✓ Carteira próxima ao aleatório - Método tradicional adequado")

# ============================================================================
# PARTE 3: FUNÇÃO DE MOVIMENTO BROWNIANO FRACIONÁRIO
# ============================================================================

def gerar_movimento_browniano_fracionario(H, n, M=256):
    """
    Gera movimento browniano fracionário (fBm)
    H: expoente de Hurst (0.5 = browniano padrão)
    n: número de pontos
    M: pontos para FFT (potência de 2)
    """
    try:
        # Garante M como potência de 2
        M = 2**int(np.ceil(np.log2(max(M, 2*n))))
        
        # Função de autocovariância para fBm
        def autocovariance(H, k):
            k = np.abs(k)
            return 0.5 * (np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))
        
        # Gera vetor de autocovariâncias
        autocov = np.array([autocovariance(H, i) for i in range(M)])
        
        # Eigenvalues via FFT
        eigenvalues = np.real(np.fft.fft(autocov))
        eigenvalues = np.maximum(eigenvalues, 0)  # Garante não-negativos
        
        # Gera ruído branco complexo
        white_noise = np.random.randn(M//2 + 1) + 1j * np.random.randn(M//2 + 1)
        white_noise[0] = white_noise[0].real
        if M % 2 == 0:
            white_noise[M//2] = white_noise[M//2].real
        
        # Constrói espectro completo
        white_noise_full = np.concatenate([white_noise, np.conj(white_noise[-2:0:-1])])
        
        # Gera fBm via FFT inversa
        fBm = np.real(np.fft.ifft(np.sqrt(eigenvalues) * white_noise_full))
        
        # Retorna apenas os n pontos necessários
        return fBm[:n]
        
    except Exception as e:
        print(f"Erro gerando fBm: {e}. Usando browniano padrão.")
        return np.cumsum(np.random.randn(n))

# ============================================================================
# PARTE 4: SIMULAÇÃO MONTE CARLO
# ============================================================================

print("\n" + "="*60)
print(f"SIMULAÇÃO MONTE CARLO {'FRACIONÁRIA' if usar_fracionario else 'TRADICIONAL'}")
print("="*60)

# Prepara parâmetros
numero_acoes = len(lista_acao)
pesos_carteira = np.full(numero_acoes, 1/numero_acoes)
matriz_covariancia = retornos.cov()
retorno_medio = retornos.mean().values

print(f"Simulações: {numero_simulacoes:,}")
print(f"Dias projetados: {dias_projetados}")
print(f"Capital inicial: R$ {capital_inicial:,.2f}")

# Arrays para armazenar resultados
retornos_carteira = np.zeros([dias_projetados, numero_simulacoes])
montante_final = np.zeros(numero_simulacoes)

# Decomposição de Cholesky para correlação
try:
    L = LA.cholesky(matriz_covariancia)
except:
    print("Matriz não positiva definida, usando SVD")
    U, s, Vt = LA.svd(matriz_covariancia)
    L = U @ np.diag(np.sqrt(s))

# ============================================================================
# LOOP DE SIMULAÇÃO
# ============================================================================

print("\nRodando simulações...")

for s in range(numero_simulacoes):
    if s % 1000 == 0:
        print(f"  Progresso: {s}/{numero_simulacoes}")
    
    if usar_fracionario:
        # MÉTODO FRACIONÁRIO - com memória
        incrementos = np.zeros((dias_projetados, numero_acoes))
        
        for i, acao in enumerate(lista_acao):
            H = hurst_values.get(acao, 0.5)
            
            # Gera movimento browniano fracionário
            fBm = gerar_movimento_browniano_fracionario(H, dias_projetados + 1)
            
            # Calcula incrementos (diferenças)
            incrementos[:, i] = np.diff(fBm)
            
            # Normaliza variância
            if np.std(incrementos[:, i]) > 0:
                incrementos[:, i] = incrementos[:, i] / np.std(incrementos[:, i])
        
        # Aplica correlação
        incrementos_correlacionados = incrementos @ L.T
        
        # Adiciona drift e escala
        matriz_retorno_medio = retorno_medio * np.ones((dias_projetados, numero_acoes))
        opcoes_sinteticas = matriz_retorno_medio + incrementos_correlacionados * np.sqrt(matriz_covariancia.values.diagonal())
        
    else:
        # MÉTODO TRADICIONAL - sem memória
        rpdf = np.random.normal(size=(dias_projetados, numero_acoes))
        matriz_retorno_medio = retorno_medio * np.ones((dias_projetados, numero_acoes))
        opcoes_sinteticas = matriz_retorno_medio + np.inner(rpdf, L)
    
    # Calcula retorno da carteira ponderada
    retorno_diario = opcoes_sinteticas @ pesos_carteira
    
    # Acumula retornos
    retornos_carteira[:, s] = np.cumprod(1 + retorno_diario) * capital_inicial
    montante_final[s] = retornos_carteira[-1, s]

print("Simulações concluídas!")

# ============================================================================
# PARTE 5: ANÁLISE DE RESULTADOS
# ============================================================================

print("\n" + "="*60)
print("ANÁLISE DE RESULTADOS")
print("="*60)

# Calcula percentis
montante_01 = np.percentile(montante_final, 1)
montante_05 = np.percentile(montante_final, 5)
montante_25 = np.percentile(montante_final, 25)
montante_50 = np.percentile(montante_final, 50)
montante_75 = np.percentile(montante_final, 75)
montante_95 = np.percentile(montante_final, 95)
montante_99 = np.percentile(montante_final, 99)

# Métricas
cenarios_com_lucro = (montante_final > capital_inicial).sum() / len(montante_final) * 100
retorno_esperado = (montante_50 / capital_inicial - 1) * 100
desvio_padrao = np.std(montante_final)
var_95 = capital_inicial - montante_05
cvar_95 = capital_inicial - np.mean(montante_final[montante_final <= montante_05])
sharpe = retorno_esperado / (desvio_padrao / capital_inicial * 100) if desvio_padrao > 0 else 0

# ============================================================================
# PARTE 6: VISUALIZAÇÃO
# ============================================================================

# Configura visualização
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Gráfico 1: Trajetórias simuladas
ax1 = fig.add_subplot(gs[0:2, 0:2])
# Plota amostra de 100 simulações
indices_amostra = np.random.choice(numero_simulacoes, min(100, numero_simulacoes), replace=False)
for idx in indices_amostra:
    ax1.plot(retornos_carteira[:, idx], alpha=0.1, color='blue', linewidth=0.5)

# Adiciona percentis principais
for p, cor, estilo in zip([5, 50, 95], ['red', 'green', 'red'], ['--', '-', '--']):
    traj_p = np.percentile(retornos_carteira, p, axis=1)
    ax1.plot(traj_p, color=cor, linewidth=2, linestyle=estilo, label=f'P{p}')

ax1.axhline(y=capital_inicial, color='black', linestyle=':', label='Capital inicial')
ax1.set_xlabel('Dias')
ax1.set_ylabel('Valor da Carteira (R$)')
ax1.set_title(f'Evolução da Carteira - {"Modelo Fracionário" if usar_fracionario else "Modelo Tradicional"}')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Distribuição dos montantes finais
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(montante_final, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(capital_inicial, color='black', linestyle=':', linewidth=2, label='Inicial')
ax2.axvline(montante_50, color='green', linestyle='-', linewidth=2, label='Mediana')
ax2.axvline(montante_05, color='red', linestyle='--', linewidth=2, label='VaR 95%')
ax2.set_xlabel('Montante Final (R$)')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição Final')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Análise de Hurst
ax3 = fig.add_subplot(gs[1, 2])
acoes_nomes = [a.replace('.SA', '') for a in lista_acao]
h_values = list(hurst_values.values())
cores_h = ['green' if h > 0.55 else 'red' if h < 0.45 else 'yellow' for h in h_values]
bars = ax3.barh(acoes_nomes, h_values, color=cores_h)
ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Aleatório')
ax3.set_xlabel('Expoente de Hurst')
ax3.set_title('Memória de Longo Prazo')
ax3.set_xlim(0, 1)

# Adiciona valores nas barras
for bar, h in zip(bars, h_values):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             f'{h:.3f}', ha='left', va='center')

# Gráfico 4: Métricas resumidas
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

# Cria texto com resultados
texto_resultados = f"""
{'='*100}
RELATÓRIO FINAL - SIMULAÇÃO MONTE CARLO {'FRACIONÁRIA' if usar_fracionario else 'TRADICIONAL'}
{'='*100}

CARTEIRA: {', '.join([a.replace('.SA', '') for a in lista_acao])}
Capital Inicial: R$ {capital_inicial:,.2f} | Período: {dias_projetados//252} anos | Simulações: {numero_simulacoes:,}

CARACTERÍSTICAS DE MEMÓRIA:
  • Hurst Médio: {media_hurst:.3f} - {"Persistência detectada" if media_hurst > 0.55 else "Reversão detectada" if media_hurst < 0.45 else "Próximo ao aleatório"}
  • Modelo usado: {"Browniano Fracionário (com memória)" if usar_fracionario else "Browniano Padrão (sem memória)"}

PROJEÇÕES (Montantes em R$):
  • Pessimista (P1):  R$ {montante_01:>10,.2f}    • Mediano (P50):   R$ {montante_50:>10,.2f}
  • Ruim (P5):        R$ {montante_05:>10,.2f}    • Bom (P75):       R$ {montante_75:>10,.2f}
  • Baixo (P25):      R$ {montante_25:>10,.2f}    • Ótimo (P95):     R$ {montante_95:>10,.2f}
                                                    • Excelente (P99): R$ {montante_99:>10,.2f}

MÉTRICAS DE RISCO E RETORNO:
  • Probabilidade de Lucro: {cenarios_com_lucro:.1f}%
  • Retorno Esperado: {retorno_esperado:.1f}%
  • VaR 95%: R$ {var_95:.2f} (perda máxima com 95% de confiança)
  • CVaR 95%: R$ {cvar_95:.2f} (perda média nos 5% piores casos)
  • Índice Sharpe: {sharpe:.2f}

{'='*100}
"""

ax4.text(0.5, 0.5, texto_resultados, fontsize=9, family='monospace',
         ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Análise Monte Carlo - {"Cálculo Fracionário" if usar_fracionario else "Método Tradicional"}',
             fontsize=14, fontweight='bold')
plt.show()

# ============================================================================
# PARTE 7: RELATÓRIO DETALHADO
# ============================================================================

print("\n" + "="*60)
print("RELATÓRIO EXECUTIVO")
print("="*60)

print(f"""
📊 ANÁLISE DA CARTEIRA
   Ativos: {', '.join([a.replace('.SA','') for a in lista_acao])}
   Período histórico: {len(precos)} dias
   Projeção: {dias_projetados//252} anos

📈 MEMÓRIA DE LONGO PRAZO
   Hurst médio: {media_hurst:.3f}
   Interpretação: {"Tendências persistem" if media_hurst > 0.55 else 
                  "Mercado reversor" if media_hurst < 0.45 else
                  "Comportamento aleatório"}

💰 CENÁRIOS PROJETADOS (R$ {capital_inicial:,.2f} inicial)
   Pessimista (1%):  R$ {montante_01:>10,.2f} ({(montante_01/capital_inicial-1)*100:+.1f}%)
   Conservador (5%): R$ {montante_05:>10,.2f} ({(montante_05/capital_inicial-1)*100:+.1f}%)
   Mediano (50%):    R$ {montante_50:>10,.2f} ({(montante_50/capital_inicial-1)*100:+.1f}%)
   Otimista (95%):   R$ {montante_95:>10,.2f} ({(montante_95/capital_inicial-1)*100:+.1f}%)
   Excelente (99%):  R$ {montante_99:>10,.2f} ({(montante_99/capital_inicial-1)*100:+.1f}%)

📊 MÉTRICAS DE RISCO
   Probabilidade de lucro: {cenarios_com_lucro:.1f}%
   VaR 95%: R$ {var_95:.2f}
   CVaR 95%: R$ {cvar_95:.2f}
   Sharpe Ratio: {sharpe:.2f}
""")

# ============================================================================
# PARTE 8: RECOMENDAÇÕES BASEADAS NA ANÁLISE
# ============================================================================

print("\n" + "="*60)
print("RECOMENDAÇÕES ESTRATÉGICAS")
print("="*60)

if media_hurst > 0.55:
    print("""
📈 ESTRATÉGIA PARA MERCADO COM PERSISTÊNCIA:
   
   1. GESTÃO DE POSIÇÕES
      • Mantenha posições vencedoras por mais tempo
      • Use stop-loss mais distante (tendências duram)
      • Pyramiding pode ser efetivo em tendências fortes
   
   2. TIMING DE ENTRADA
      • Aguarde confirmação de tendência antes de entrar
      • Breakouts têm maior probabilidade de sucesso
      • Evite tentar pegar topos/fundos
   
   3. GESTÃO DE RISCO
      • Quando a reversão vier, será forte - esteja preparado
      • Considere trailing stops para proteger lucros
      • Diversifique entre ativos com diferentes H
""")

elif media_hurst < 0.45:
    print("""
📉 ESTRATÉGIA PARA MERCADO REVERSOR:
   
   1. GESTÃO DE POSIÇÕES
      • Realize lucros rapidamente
      • Use stop-loss mais apertado
      • Evite manter posições por muito tempo
   
   2. TIMING DE ENTRADA
      • Compre em suportes, venda em resistências
      • Use indicadores de sobrecompra/sobrevenda
      • Estratégias contrarian são favorecidas
   
   3. GESTÃO DE RISCO
      • Mercado "nervoso" - reduza tamanho das posições
      • Faça rebalanceamento frequente
      • Considere estratégias de arbitragem
""")

else:
    print("""
🎯 ESTRATÉGIA PARA MERCADO EFICIENTE:
   
   1. GESTÃO DE POSIÇÕES
      • Foque em diversificação ampla
      • Buy and hold pode ser adequado
      • Minimize custos operacionais
   
   2. TIMING DE ENTRADA
      • Dollar-cost averaging é recomendado
      • Evite market timing
      • Rebalanceamento periódico (trimestral/semestral)
   
   3. GESTÃO DE RISCO
      • Use alocação estratégica de ativos
      • Considere indexação passiva
      • Foque no longo prazo
""")

# ============================================================================
# PARTE 9: EXPORTAR RESULTADOS (OPCIONAL)
# ============================================================================

# Salva resultados em DataFrame
resultados = pd.DataFrame({
    'Métrica': [
        'Capital Inicial', 'Retorno Esperado (%)', 'Desvio Padrão',
        'VaR 95%', 'CVaR 95%', 'Sharpe Ratio', 'Prob. Lucro (%)',
        'P1', 'P5', 'P25', 'P50', 'P75', 'P95', 'P99',
        'Hurst Médio', 'Modelo'
    ],
    'Valor': [
        capital_inicial, retorno_esperado, desvio_padrao,
        var_95, cvar_95, sharpe, cenarios_com_lucro,
        montante_01, montante_05, montante_25, montante_50,
        montante_75, montante_95, montante_99,
        media_hurst, 'Fracionário' if usar_fracionario else 'Tradicional'
    ]
})

# Salva em CSV
nome_arquivo = f'monte_carlo_resultados_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
resultados.to_csv(nome_arquivo, index=False)
print(f"\n✅ Resultados salvos em: {nome_arquivo}")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*60)
