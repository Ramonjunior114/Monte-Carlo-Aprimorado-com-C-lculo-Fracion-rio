# Monte Carlo Aprimorado com Cálculo Fracionário
# Melhora a simulação original incorporando memória longa e persistência

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
from numpy import linalg as LA
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para cálculo fracionário
from hurst import compute_Hc
from fracdiff import fdiff

# ============================================================================
# PARTE 1: COLETA DE DADOS (igual ao original)
# ============================================================================

lista_acao = ['VALE3.SA','ITUB4.SA','BBDC3.SA','WEGE3.SA','PETR4.SA']

data_final = dt.datetime.now()
data_inicial = data_final - dt.timedelta(days=300)

precos = pdr.get_data_yahoo(lista_acao, data_inicial, data_final)['Adj Close']
retornos = precos.pct_change().dropna()

# ============================================================================
# PARTE 2: ANÁLISE DE MEMÓRIA LONGA (NOVO!)
# ============================================================================

print("="*60)
print("ANÁLISE DE MEMÓRIA DAS AÇÕES")
print("="*60)

# Calcula expoente de Hurst para cada ação
hurst_values = {}
usar_fracionario = False

for acao in lista_acao:
    # Calcula H para os preços
    H, c, data = compute_Hc(precos[acao].dropna().values, kind='price')
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

media_hurst = np.mean(list(hurst_values.values()))
print(f"\nMédia Hurst da carteira: {media_hurst:.3f}")

if usar_fracionario:
    print("\n⚠️ DETECTADA MEMÓRIA LONGA - Usando simulação fracionária!")
    print("Isso tornará as projeções mais realistas.")
else:
    print("\n✓ Carteira próxima ao aleatório - Método tradicional OK")

# ============================================================================
# PARTE 3: MOVIMENTO BROWNIANO FRACIONÁRIO (NOVO!)
# ============================================================================

def gerar_browniano_fracionario(H, n, M=256):
    """
    Gera movimento browniano fracionário usando método de Hosking.
    
    H: Expoente de Hurst
    n: Número de pontos
    M: Número de pontos para FFT (potência de 2)
    """
    # Garante que M seja potência de 2 e maior que n
    M = 2**int(np.ceil(np.log2(max(M, 2*n))))
    
    # Função de autocovariância para fBm
    def autocovariance(H, k):
        return 0.5 * (abs(k-1)**(2*H) - 2*abs(k)**(2*H) + abs(k+1)**(2*H))
    
    # Gera autocovariâncias
    autocov = np.array([autocovariance(H, i) for i in range(M)])
    
    # Método circulante usando FFT
    eigenvalues = np.real(np.fft.fft(autocov))
    
    # Garante valores não-negativos
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Gera ruído branco
    white_noise = np.random.randn(M//2 + 1) + 1j * np.random.randn(M//2 + 1)
    white_noise[0] = white_noise[0].real
    if M % 2 == 0:
        white_noise[M//2] = white_noise[M//2].real
    
    # Constrói espectro completo
    white_noise_full = np.concatenate([white_noise, np.conj(white_noise[-2:0:-1])])
    
    # Gera fBm via FFT
    fBm = np.real(np.fft.ifft(np.sqrt(eigenvalues) * white_noise_full))
    
    return fBm[:n]

# ============================================================================
# PARTE 4: SIMULAÇÃO MONTE CARLO APRIMORADA
# ============================================================================

numero_simulacoes = 10000
dias_projetados = 252 * 3
capital_inicial = 1000

# Parâmetros da carteira
numero_acoes = len(lista_acao)
pesos_carteira = np.full(numero_acoes, 1/numero_acoes)
matriz_covariancia = retornos.cov()
retorno_medio = retornos.mean(axis=0).to_numpy()

# Decide método baseado na análise de Hurst
if usar_fracionario:
    print("\n" + "="*60)
    print("SIMULAÇÃO MONTE CARLO FRACIONÁRIA")
    print("="*60)
    metodo = "Browniano Fracionário"
else:
    print("\n" + "="*60)
    print("SIMULAÇÃO MONTE CARLO TRADICIONAL")
    print("="*60)
    metodo = "Browniano Padrão"

print(f"Método: {metodo}")
print(f"Simulações: {numero_simulacoes:,}")
print(f"Dias projetados: {dias_projetados}")

# Arrays para resultados
retornos_carteira = np.zeros([dias_projetados, numero_simulacoes])
montante_final = np.zeros(numero_simulacoes)

# Decomposição de Cholesky
L = LA.cholesky(matriz_covariancia)

# ============================================================================
# SIMULAÇÃO PRINCIPAL
# ============================================================================

for s in range(numero_simulacoes):
    if usar_fracionario:
        # MÉTODO FRACIONÁRIO (NOVO!)
        # Gera movimento browniano fracionário para cada ação
        incrementos = np.zeros((dias_projetados, numero_acoes))
        
        for i, acao in enumerate(lista_acao):
            H = hurst_values[acao]
            
            # Gera fBm
            fBm = gerar_browniano_fracionario(H, dias_projetados + 1)
            
            # Calcula incrementos
            incrementos[:, i] = np.diff(fBm)
            
            # Normaliza para ter variância unitária
            incrementos[:, i] = incrementos[:, i] / np.std(incrementos[:, i])
        
        # Aplica correlação via Cholesky
        incrementos_correlacionados = np.dot(incrementos, L.T)
        
        # Adiciona drift (retorno médio)
        retornos_simulados = retorno_medio + incrementos_correlacionados * np.sqrt(1/252)
        
    else:
        # MÉTODO TRADICIONAL (original)
        rpdf = np.random.normal(size=(dias_projetados, numero_acoes))
        matriz_retorno_medio = retorno_medio * np.ones(shape=(dias_projetados, numero_acoes))
        opcoes_sinteticas = matriz_retorno_medio + np.inner(rpdf, L)
        retornos_simulados = opcoes_sinteticas
    
    # Calcula evolução da carteira
    retorno_diario = np.dot(retornos_simulados, pesos_carteira)
    retornos_carteira[:, s] = np.cumprod(1 + retorno_diario) * capital_inicial
    montante_final[s] = retornos_carteira[-1, s]

# ============================================================================
# PARTE 5: ANÁLISE DE RESULTADOS COMPARATIVA
# ============================================================================

# Calcula percentis
montante_99 = np.percentile(montante_final, 1)
montante_95 = np.percentile(montante_final, 5)
montante_50 = np.percentile(montante_final, 50)
montante_05 = np.percentile(montante_final, 95)
montante_01 = np.percentile(montante_final, 99)

cenarios_com_lucro = (len(montante_final[montante_final > capital_inicial]) / len(montante_final)) * 100

# Calcula métricas de risco
desvio_padrao = np.std(montante_final)
var_95 = capital_inicial - montante_95
cvar_95 = capital_inicial - np.mean(montante_final[montante_final <= montante_95])

# ============================================================================
# PARTE 6: VISUALIZAÇÃO APRIMORADA
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Trajetórias simuladas
ax1 = axes[0, 0]
# Plota amostra das simulações (100 para não poluir)
sample_idx = np.random.choice(numero_simulacoes, 100, replace=False)
for idx in sample_idx:
    ax1.plot(retornos_carteira[:, idx], alpha=0.3, linewidth=0.5)

# Adiciona percentis
percentis = [5, 25, 50, 75, 95]
cores = ['red', 'orange', 'green', 'orange', 'red']
for p, cor in zip(percentis, cores):
    traj_percentil = np.percentile(retornos_carteira, p, axis=1)
    ax1.plot(traj_percentil, color=cor, linewidth=2, label=f'P{p}')

ax1.axhline(y=capital_inicial, color='black', linestyle='--', label='Capital inicial')
ax1.set_xlabel('Dias')
ax1.set_ylabel('Valor da Carteira (R$)')
ax1.set_title(f'Simulações Monte Carlo - {metodo}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Distribuição dos montantes finais
ax2 = axes[0, 1]
ax2.hist(montante_final, bins=100, alpha=0.7, color='blue', edgecolor='black')
ax2.axvline(capital_inicial, color='red', linestyle='--', linewidth=2, label='Capital inicial')
ax2.axvline(montante_50, color='green', linestyle='-', linewidth=2, label='Mediana')
ax2.axvline(montante_95, color='orange', linestyle='--', linewidth=2, label='VaR 95%')
ax2.set_xlabel('Montante Final (R$)')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição dos Montantes Finais')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Análise de Hurst por ação
ax3 = axes[1, 0]
acoes_nomes = [acao.replace('.SA', '') for acao in lista_acao]
h_values = list(hurst_values.values())
cores_barras = ['green' if h > 0.55 else 'red' if h < 0.45 else 'yellow' for h in h_values]
bars = ax3.bar(acoes_nomes, h_values, color=cores_barras)
ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax3.set_ylabel('Expoente de Hurst')
ax3.set_title('Memória de Longo Prazo por Ação')
ax3.set_ylim(0, 1)

# Adiciona valores nas barras
for bar, h in zip(bars, h_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{h:.3f}', ha='center', va='bottom')

# Gráfico 4: Comparação de risco-retorno
ax4 = axes[1, 1]
retorno_esperado = (montante_50 / capital_inicial - 1) * 100
risco = (desvio_padrao / capital_inicial) * 100
sharpe = retorno_esperado / risco if risco > 0 else 0

# Texto com métricas
texto_metricas = f"""
MÉTRICAS DA CARTEIRA ({metodo})

Retorno esperado: {retorno_esperado:.1f}%
Risco (desvio): {risco:.1f}%
Sharpe Ratio: {sharpe:.2f}

VaR 95%: R$ {var_95:.2f}
CVaR 95%: R$ {cvar_95:.2f}

Prob. de lucro: {cenarios_com_lucro:.1f}%

Montantes (percentis):
  P1:  R$ {montante_99:,.2f}
  P5:  R$ {montante_95:,.2f}
  P50: R$ {montante_50:,.2f}
  P95: R$ {montante_05:,.2f}
  P99: R$ {montante_01:,.2f}

Hurst médio: {media_hurst:.3f}
"""

ax4.text(0.1, 0.5, texto_metricas, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
ax4.axis('off')

plt.suptitle(f'Análise Monte Carlo com {"Cálculo Fracionário" if usar_fracionario else "Método Tradicional"}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# PARTE 7: RELATÓRIO EXECUTIVO
# ============================================================================

print("\n" + "="*60)
print("RELATÓRIO EXECUTIVO")
print("="*60)

print(f"""
ANÁLISE DA CARTEIRA: {', '.join([a.replace('.SA','') for a in lista_acao])}

📊 MÉTODO UTILIZADO:
   {metodo} {"(Recomendado pelos dados)" if usar_fracionario else "(Padrão)"}
   
📈 CARACTERÍSTICAS DE MEMÓRIA:
   Hurst médio: {media_hurst:.3f}
   {"⚠️ Carteira tem MEMÓRIA LONGA - tendências persistem" if media_hurst > 0.55 else 
    "⚠️ Carteira tem REVERSÃO À MÉDIA - tendências revertem" if media_hurst < 0.45 else
    "✓ Carteira próxima ao aleatório"}

💰 PROJEÇÃO PARA 3 ANOS (R$ {capital_inicial} inicial):

   CENÁRIO PESSIMISTA (1% prob.): R$ {montante_99:,.2f}
   CENÁRIO RUIM (5% prob.):       R$ {montante_95:,.2f}
   CENÁRIO MEDIANO (50% prob.):   R$ {montante_50:,.2f}
   CENÁRIO BOM (95% prob.):       R$ {montante_05:,.2f}
   CENÁRIO ÓTIMO (99% prob.):     R$ {montante_01:,.2f}

📊 MÉTRICAS DE RISCO:
   Probabilidade de lucro: {cenarios_com_lucro:.1f}%
   VaR 95% (perda máx. 95% confiança): R$ {var_95:.2f}
   CVaR 95% (perda média cauda): R$ {cvar_95:.2f}
   Sharpe Ratio: {sharpe:.2f}

🎯 RECOMENDAÇÕES BASEADAS NA ANÁLISE:
""")

if media_hurst > 0.55:
    print("""
   1. TENDÊNCIAS PERSISTEM - Use stop-loss mais distante
   2. Mantenha posições vencedoras por mais tempo
   3. Considere aumentar exposição em tendências confirmadas
   4. Cuidado: reversões serão mais bruscas quando ocorrerem
   """)
elif media_hurst < 0.45:
    print("""
   1. MERCADO REVERSOR - Use estratégias de mean-reversion
   2. Realize lucros mais rapidamente
   3. Compre nas quedas, venda nas altas
   4. Stop-loss mais apertado é recomendado
   """)
else:
    print("""
   1. MERCADO EFICIENTE - Diversificação é crucial
   2. Rebalanceamento regular recomendado
   3. Foque em redução de custos operacionais
   4. Estratégias passivas podem ser adequadas
   """)

print("\n" + "="*60)
print("FIM DA ANÁLISE")
print("="*60)

# ============================================================================
# PARTE 8: EXPORTAR RESULTADOS (NOVO!)
# ============================================================================

# Cria DataFrame com resultados
resultados_df = pd.DataFrame({
    'Métrica': ['Retorno Esperado (%)', 'Risco (%)', 'Sharpe', 'VaR 95%', 'CVaR 95%', 
                'Prob. Lucro (%)', 'P1', 'P5', 'P50', 'P95', 'P99', 'Hurst Médio'],
    'Valor': [retorno_esperado, risco, sharpe, var_95, cvar_95, cenarios_com_lucro,
              montante_99, montante_95, montante_50, montante_05, montante_01, media_hurst]
})

# Salva em Excel
resultados_df.to_excel('resultados_monte_carlo_fracionario.xlsx', index=False)
print("\n✓ Resultados salvos em 'resultados_monte_carlo_fracionario.xlsx'")
