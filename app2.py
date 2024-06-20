import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.tsa.stattools as ts
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic


#st.set_page_config(layout='wide')

# Criar um combobox no sidebar para selecionar o idioma
language = st.sidebar.selectbox('Select language / Selecione o idioma / Seleccione el idioma', ('English', 'Español', 'Português'))

if language == 'English':
    tit1 = 'Euro/Swiss Franc (EUR/CHF) Price Relationship: Cointegration Analysis and Time Series Models'
    text101 = 'EUR/CHF Weekly Close Prices with SMA'
    text102 = 'Close Price'
    text103 = 'Date'
    text104 = 'EUR/CHF Close Price'
    text105 = ' weeks'
    text106 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Analysis</span>
  
The chart shows the weekly closing prices of EUR/CHF from June 2019 to June 2024. There is a stabilization in prices during the initial period of analysis until March 2021, followed by a slight downward trend in the Euro price against the Swiss Franc until September 2022, and then stabilization.

Given the observed behavior in EUR/CHF prices, the next step would be to perform a cointegration analysis. This analysis will allow us to quantify the long-term relationship between the Euro and the Swiss Franc, identifying whether there is a stable relationship that can be exploited for modeling and investment strategies. It is a crucial step to deepen our understanding of market dynamics and potential trading opportunities.
"""
    tit2 = 'Cointegration Analysis'
    text201 = "Engle-Granger Test"
    text202 = "ADF Statistic"
    text203 = "P-value"
    text204 = "Critical Value 1%"
    text205 = "Critical Value 5%"
    text206 = "Critical Value 10%"
    text207 = "► The ADF test results indicate that the residuals of the Engle-Granger regression <font color='#2ECC71'>are stationary</font>, suggesting the <font color='#2ECC71'>presence of cointegration</font> between EUR/CHF."
    text208 = "► The ADF test results indicate that the spread <span style='color: #FF4500;'>is not stationary</span>. Applying differencing, the differenced spread is used."
    text209 = "Johansen Test"
    text210 = "Trace Statistic"
    text211 = "Critical Value 1%"
    text212 = "Max Eigenvalue Statistic"
    text213 = "Critical Value 1%"
    text214 = "► The Johansen test results indicate that <font color='#2ECC71'>there is evidence of cointegration between EUR/CHF</font>, by both the trace test and the maximum eigenvalue test."
    text215 = "► The Johansen test results <font color='#E74C3C'>do not provide sufficient evidence of cointegration between EUR/CHF.</font>"
    text216 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Analysis</span>

The results of the Engle-Granger tests indicate that there is not enough evidence of cointegration between the closing prices of EUR and CHF, suggesting that the series do not share a stable long-term relationship that can be exploited for future predictions. This implies that variations in EUR/CHF prices may be more random or influenced by factors not captured by cointegration.

The next crucial step will be the analysis of the spread between EUR and CHF, which is the difference between their closing prices. The goal will be to investigate if there is a stable long-term relationship between these spreads, which can provide valuable insights for arbitrage-based trading strategies or other investment opportunities.
"""
    tit3 = 'Spread Modeling'
    text301 = "Spread between EUR and CHF over time"
    text302 = "Spread"
    text303 = "Original Spread"
    text304 = "Differenced Spread"
    text305 = "ADF Test for Spread Stationarity"
    text306 = "► The ADF test results indicate that the spread <span style='color: #32CD32;'>is stationary</span>, suggesting no need for additional transformation."
    text307 = "► The ADF test results indicate that the spread <span style='color: #FF4500;'>is not stationary</span>. Applying differencing, the differenced spread is used."
    text308 = "ADF Test for Stationarity of Differenced Spread"
    text309 = "► The ADF test results indicate that the differenced spread <span style='color: #32CD32;'>is stationary</span>."
    text310 = "► The ADF test results indicate that the differenced spread <span style='color: #FF4500;'>is not stationary</span>."
    text311 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Analysis</span>

The chart of the original spread between EUR and CHF reveals variations over the analyzed period, showing periods of high and low spreads. In 2024, it is observed that the spread is stabilizing close to zero. On the other hand, the differenced spread appears stationary, with values near zero after differentiation.

The results of the Engle-Granger test indicate that the original spread between EUR and CHF is not stationary, highlighting the need for differentiation to make the series stationary. After differentiation, the ADF test confirms that the differenced spread is stationary, implying that its movements over time are more predictable and less subject to non-stationary trends.

Now it is crucial to perform the Time Series Model for the Spread between EUR and CHF to explore these dynamics more deeply. The main objective of this model is to capture and model the historical patterns and behaviors of the spread, allowing for the prediction of its future movements. This is essential for investors and analysts who wish to better understand the relationships between the two currencies and identify trading opportunities based on robust quantitative analysis.
"""
    tit4 = 'Time Series Model for Spread'
    text401 = "ARMA Model for Spread"
    text402 = "Best parameters: "
    text403 = "ARMA Model Residuals"
    text404 = "Residuals"
    text405 = "Date"
    text406 = "The ARMA model fitted to the spread suggests the following interpretation:"
    text4071 = "AR Parameter (p):"
    text4072 = "indicates that the spread is influenced by"
    text4073 = "lagged period."
    text4081 = "Parameter MA (q):"
    text4082 = "indicates that the spread is influenced by"
    text4083 = "periods of past errors."
    text409 = "The residuals of the ARMA model show the goodness of fit and potential predictability of the spread."
    text410 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Analysis</span>

The ARMA model is a statistical tool used to analyze and forecast time series, such as the spread between two currencies. Here are the key practical points from the results obtained:

**Influence of Previous Periods:** The AR (Autoregressive) parameter of the ARMA model for EUR/CHF spread was estimated to be 1. This means that the current value of the spread is significantly influenced by its own previous value. In practical terms, if the spread between EUR and CHF increased or decreased in the previous period, this will have a direct impact on the current spread.

**Stationarity of Residuals:** The residuals of the ARMA model, which represent the differences between observed values and values predicted by the model, were found to be stationary around zero. This is important because it indicates that the model captures the variability of the data well without leaving unexplained patterns in the residuals.

**Predictability of Spread:** Based on the estimated coefficients and analysis of residuals, the ARMA model suggests that the past behavior of the spread can be used to predict its future behavior with some accuracy. This is useful for traders and financial analysts looking to anticipate movements in the foreign exchange market between EUR and CHF.

**Interpretation of Information Criteria:** The AIC, BIC, and HQIC criteria provide measures for selecting between alternative models. In the presented case, lower values of these criteria indicate that the chosen ARMA model is most suitable for the observed data of EUR/CHF spread, considering both the goodness of fit and model complexity.

In summary, the results of the ARMA model for EUR/CHF spread provide a solid framework for understanding how past variations influence the current spread and for making predictions about its future behavior. This practical analysis is essential for making informed decisions in the financial market, from investment strategies to foreign exchange risk management.
"""
    tit5 = 'Considerations for a Trading Strategy'
    text51 = """
► Spread-Based Trading Strategy

**Interpretation of Cointegration Results:**

**Stationary Spread:** After analyzing the Engle-Granger and Johansen tests, we conclude that there is not enough evidence of cointegration between EUR/CHF. This suggests that there is no stable long-term relationship between these currencies.

**ARMA Model for Trading Strategy:**

**Best Parameters:** Based on the ARMA (1,0) model, we observe that the model suggests a dynamic where the spread between EUR/CHF is mainly influenced by the value of the previous period, without a moving average component.

**Practical Implementation:** The ARMA model provides insights into how the spread between EUR/CHF may behave based on historical data. For example, if we observe an increase in the spread beyond what is expected by the model, this may indicate a trading opportunity.

**Trading Strategy Based on the ARMA Model:**

**Deviation Analysis:** To identify arbitrage opportunities, we may consider selling EUR and buying CHF when the current spread exceeds a significant positive deviation from the ARMA model forecasts.

**Risk Management:** Establishing stop-loss and take-profit limits based on the ARMA model projections helps manage risks and capture potential gains during spread fluctuations.

**Continuous Monitoring and Adjustments:**

**Regular Review:** Maintaining constant vigilance over spread changes and adjusting the strategy based on new ARMA model forecasts is crucial for maximizing profit opportunities and minimizing losses.

**Flexibility:** Quickly adjusting the strategy based on new data and market conditions helps adapt to changing EUR/CHF market dynamics.

**Long-Term Strategy and Education:**

**Diversification and Knowledge:** Combining strategies based on statistical models like ARMA with a deep understanding of economic and geopolitical fundamentals strengthens the ability to make informed decisions in the forex market.

**Investment in Knowledge:** Continuing education in technical, economic, and macroeconomic analysis sustains the ability to interpret data and trends accurately, essential for effective trading in EUR/CHF.

These practical insights help traders and investors strategically apply the ARMA model in the EUR/CHF market, enhancing profit opportunities and effectively managing associated risks.
"""
elif language == 'Español':
    tit1 = 'Relación de Precios entre Euro/Franco Suizo (EUR/CHF): Análisis de Cointegración y Modelos Temporales'
    text101 = 'Precios de Cierre Semanal de EUR/CHF con SMA'
    text102 = 'Precio de Cierre'
    text103 = 'Fecha'
    text104 = 'Precio de Cierre EUR/CHF'
    text105 = ' semanas'
    text106 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Análisis</span>
  
El gráfico muestra los precios de cierre semanales de EUR/CHF desde junio de 2019 hasta junio de 2024. Se observa una estabilización en los precios durante el período inicial del análisis hasta marzo de 2021, seguida de una ligera tendencia a la baja en el precio del Euro frente al Franco Suizo hasta septiembre de 2022, y luego estabilización.

Dado el comportamiento observado en los precios de EUR/CHF, el próximo paso sería realizar un análisis de cointegración. Este análisis nos permitirá cuantificar la relación de largo plazo entre el Euro y el Franco Suizo, identificando si existe una relación estable que se pueda explotar para modelado y estrategias de inversión. Es un paso crucial para profundizar nuestra comprensión de las dinámicas del mercado y las oportunidades potenciales de negociación.
"""
    tit2 = 'Análisis de Cointegración'
    text201 = "Prueba de Engle-Granger"
    text202 = "Estadística ADF"
    text203 = "Valor-p"
    text204 = "Valor Crítico 1%"
    text205 = "Valor Crítico 5%"
    text206 = "Valor Crítico 10%"
    text207 = "► Los resultados de la prueba ADF indican que los residuos de la regresión de Engle-Granger <font color='#2ECC71'>son estacionarios</font>, sugiriendo la <font color='#2ECC71'>presencia de cointegración</font> entre EUR/CHF."
    text208 = "► Los resultados de la prueba ADF indican que el spread <span style='color: #FF4500;'>no es estacionario</span>. Aplicando la diferenciación, se utiliza el spread diferenciado."
    text209 = "Prueba de Johansen"
    text210 = "Estadística de Traza"
    text211 = "Valor Crítico 1%"
    text212 = "Estadística de Máximo Autovalor"
    text213 = "Valor Crítico 1%"
    text214 = "► Los resultados de la prueba de Johansen indican que <font color='#2ECC71'>hay evidencia de cointegración entre EUR/CHF</font>, tanto por la prueba de traza como por la prueba de máximo autovalor."
    text215 = "► Los resultados de la prueba de Johansen <font color='#E74C3C'>no proporcionan evidencia suficiente de cointegración entre EUR/CHF.</font>"
    text216 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Análisis</span>

Los resultados de los tests de Engle-Granger indican que no hay suficiente evidencia de cointegración entre los precios de cierre de EUR y CHF, sugiriendo que las series no comparten una relación estable a largo plazo que pueda ser explotada para predicciones futuras. Esto implica que las variaciones en los precios de EUR/CHF pueden ser más aleatorias o influenciadas por factores no capturados por la cointegración.

El próximo paso crucial será el análisis del spread entre EUR y CHF, que es la diferencia entre sus precios de cierre. El objetivo será investigar si existe una relación estable a largo plazo entre estos spreads, lo cual puede proporcionar ideas valiosas para estrategias de negociación basadas en arbitraje u otras oportunidades de inversión.
"""
    tit3 = 'Modelado del Spread'
    text301 = "Spread entre EUR y CHF a lo largo del tiempo"
    text302 = "Spread"
    text303 = "Spread Original"
    text304 = "Spread Diferenciado"
    text305 = "Prueba ADF para la Estacionariedad del Spread"
    text306 = "► Los resultados de la prueba ADF indican que el spread <span style='color: #32CD32;'>es estacionario</span>, sugiriendo que no hay necesidad de transformación adicional."
    text307 = "► Los resultados de la prueba ADF indican que el spread <span style='color: #FF4500;'>no es estacionario</span>. Aplicando la diferenciación, se utiliza el spread diferenciado."
    text308 = "Prueba ADF para la Estacionariedad del Spread Diferenciado"
    text309 = "► Los resultados de la prueba ADF indican que el spread diferenciado <span style='color: #32CD32;'>es estacionario</span>."
    text310 = "► Los resultados de la prueba ADF indican que el spread diferenciado <span style='color: #FF4500;'>no es estacionario</span>."
    text311 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Análisis</span>

El gráfico del spread original entre EUR y CHF revela variaciones a lo largo del período analizado, mostrando períodos de spreads altos y bajos. En 2024, se observa que el spread se está estabilizando cerca de cero. Por otro lado, el spread diferenciado parece estacionario, con valores cercanos a cero después de la diferenciación.

Los resultados del test de Engle-Granger indican que el spread original entre EUR y CHF no es estacionario, resaltando la necesidad de diferenciación para hacer que la serie sea estacionaria. Después de la diferenciación, el test ADF confirma que el spread diferenciado es estacionario, lo que implica que sus movimientos a lo largo del tiempo son más predecibles y menos sujetos a tendencias no estacionarias.

Ahora es crucial realizar el Modelo de Series Temporales para el Spread entre EUR y CHF para explorar estas dinámicas de manera más profunda. El objetivo principal de este modelo es capturar y modelar los patrones y comportamientos históricos del spread, permitiendo la predicción de sus movimientos futuros. Esto es fundamental para inversores y analistas que desean entender mejor las relaciones entre las dos monedas e identificar oportunidades de negociación basadas en análisis cuantitativos robustos.
"""
    tit4 = 'Modelo de Series Temporales para el Spread'
    text401 = "Modelo ARMA para el Spread"
    text402 = "Mejores parámetros:"
    text403 = "Residuos del Modelo ARMA"
    text404 = "Residuos"
    text405 = "Fecha"
    text406 = "El modelo ARMA ajustado al spread sugiere la siguiente interpretación:"
    text4071 = "Parámetro AR (p):"
    text4072 = "indica que el spread está influenciado por"
    text4073 = "período pasado."
    text4081 = "Parámetro MA (q):"
    text4082 = "indica que el spread no está influenciado por"
    text4083 = "períodos de error pasados."
    text409 = "Los residuos del modelo ARMA muestran la calidad del ajuste y la potencial predictibilidad del spread."
    text410 = """
<span style=\"color: #0068c9; font-weight: bold; font-size: 18px;\">Análisis</span>

El modelo ARMA es una herramienta estadística utilizada para analizar y prever series temporales, como el spread entre dos monedas. Aquí están los puntos prácticos clave de los resultados obtenidos:

**Influencia de los Períodos Anteriores:** El parámetro AR (Autorregresivo) del modelo ARMA para el spread EUR/CHF se estimó en 1. Esto significa que el valor actual del spread está influenciado significativamente por su propio valor anterior. En términos prácticos, si el spread entre EUR y CHF aumentó o disminuyó en el período anterior, esto tendrá un impacto directo en el spread actual.

**Estacionariedad de los Residuos:** Los residuos del modelo ARMA, que representan las diferencias entre los valores observados y los valores predichos por el modelo, se mostraron estacionarios alrededor de cero. Esto es importante porque indica que el modelo captura bien la variabilidad de los datos sin dejar patrones inexplicados en los residuos.

**Previsibilidad del Spread:** Basado en los coeficientes estimados y el análisis de los residuos, el modelo ARMA sugiere que el comportamiento pasado del spread puede ser utilizado para predecir su comportamiento futuro con cierta precisión. Esto es útil para traders y analistas financieros que buscan anticipar movimientos en el mercado de divisas entre EUR y CHF.

**Interpretación de los Criterios de Información:** Los criterios AIC, BIC y HQIC proporcionan medidas para seleccionar entre modelos alternativos. En el caso presentado, valores más bajos de estos criterios indican que el modelo ARMA elegido es el más adecuado para los datos observados del spread EUR/CHF, considerando tanto la bondad del ajuste como la complejidad del modelo.

En resumen, los resultados del modelo ARMA para el spread EUR/CHF proporcionan un marco sólido para entender cómo las variaciones pasadas influyen en el spread actual y para hacer predicciones sobre su comportamiento futuro. Este análisis práctico es esencial para tomar decisiones informadas en el mercado financiero, desde estrategias de inversión hasta gestión del riesgo cambiario.
"""
    tit5 = 'Consideraciones para una Estrategia de Negociación'
    text51 = """
► Estrategia de Negociación Basada en el Spread

**Interpretación de los Resultados de Cointegración:**

**Spread Estacionario:** Después de analizar las pruebas de Engle-Granger y Johansen, concluimos que no hay suficiente evidencia de cointegración entre EUR/CHF. Esto sugiere que no existe una relación estable a largo plazo entre estas monedas.

**Modelo ARMA para Estrategia de Negociación:**

**Mejores Parámetros:** Según el modelo ARMA (1,0), observamos que el modelo sugiere una dinámica donde el spread entre EUR/CHF está influenciado principalmente por el valor del período anterior, sin componente de media móvil.

**Implementación Práctica:** El modelo ARMA proporciona ideas sobre cómo podría comportarse el spread entre EUR/CHF basado en datos históricos. Por ejemplo, si observamos un aumento en el spread más allá de lo esperado por el modelo, esto podría indicar una oportunidad de negociación.

**Estrategia de Negociación Basada en el Modelo ARMA:**

**Análisis de Desviación:** Para identificar oportunidades de arbitraje, podríamos considerar vender EUR y comprar CHF cuando el spread actual supere una desviación positiva significativa con respecto a las previsiones del modelo ARMA.

**Gestión de Riesgos:** Establecer límites de stop-loss y take-profit basados en las proyecciones del modelo ARMA ayuda a gestionar los riesgos y capturar ganancias potenciales durante las fluctuaciones del spread.

**Monitoreo Continuo y Ajustes:**

**Revisión Regular:** Mantener una vigilancia constante sobre los cambios en el spread y ajustar la estrategia en función de las nuevas previsiones del modelo ARMA es crucial para maximizar las oportunidades de beneficio y minimizar las pérdidas.

**Flexibilidad:** Ajustar rápidamente la estrategia en función de nuevos datos y condiciones del mercado ayuda a adaptarse a las cambiantes dinámicas del mercado EUR/CHF.

**Estrategia a Largo Plazo y Educación:**

**Diversificación y Conocimiento:** Combinar estrategias basadas en modelos estadísticos como ARMA con un profundo conocimiento de los fundamentos económicos y geopolíticos fortalece la capacidad para tomar decisiones informadas en el mercado de divisas.

**Inversión en Conocimiento:** Continuar la educación en análisis técnico, económico y macroeconómico sostiene la habilidad para interpretar datos y tendencias con precisión, esencial para una negociación efectiva en EUR/CHF.

Estos consejos prácticos ayudan a los traders e inversores a aplicar estratégicamente el modelo ARMA en el mercado EUR/CHF, potenciando las oportunidades de beneficio y gestionando eficazmente los riesgos asociados.
"""
elif language == 'Português':
    tit1 = 'Relação de Preços entre Euro/Franco Suíço (EUR/CHF): Análise de Cointegração e Modelos Temporais'
    text101 = 'Preços de Fechamento Semanal do EUR/CHF com SMA'
    text102 = 'Preço de Fechamento'
    text103 = 'Data'
    text104 = 'Preço de Fechamento EUR/CHF'
    text105 = ' semanas'
    text106 = """
<span style="color: #0068c9; font-weight: bold; font-size: 18px;">Análise</span>
  
O gráfico mostra os preços de fechamento semanais do EUR/CHF de junho de 2019 a junho de 2024. Observa-se uma estabilização nos preços durante o período inicial da análise até março de 2021, seguida por uma leve tendência de queda no preço do Euro em relação ao Franco Suíço até setembro de 2022, seguida por uma estabilização.
    
Dado o comportamento observado nos preços do EUR/CHF, o próximo passo seria realizar uma análise de cointegração. Esta análise nos permitirá quantificar a relação de longo prazo entre o Euro e o Franco Suíço, identificando se há uma relação estável que pode ser explorada para fins de modelagem e estratégias de investimento. É um passo crucial para aprofundar nossa compreensão das dinâmicas de mercado e potenciais oportunidades de negociação.
"""
    tit2 = 'Análise de Cointegração'
    text201 = "Teste de Engle-Granger"
    text202 = "Estatística ADF"
    text203 = "Valor-p"
    text204 = "Valor Crítico 1%"
    text205 = "Valor Crítico 5%"
    text206 = "Valor Crítico 10%"
    text207 = "► Os resultados do teste ADF indicam que os resíduos da regressão de Engle-Granger <font color='#2ECC71'>são estacionários</font>, sugerindo a <font color='#2ECC71'>presença de cointegração</font> entre EUR/CHF."
    text208 = "► Os resultados do teste ADF indicam que o spread <span style='color: #FF4500;'>não é estacionário</span>. Aplicando a diferenciação, o spread diferenciado é utilizado."
    text209 = "Teste de Johansen"
    text210 = "Estatística de Traço"
    text211 = "Valor Crítico 1%"
    text212 = "Estatística de Máximo Autovalor"
    text213 = "Valor Crítico 1%"
    text214 = "► Os resultados do teste de Johansen indicam que <font color='#2ECC71'>há evidências de cointegração entre EUR/CHF</font>, tanto pelo teste de traço quanto pelo teste de máximo autovalor."
    text215 = "► Os resultados do teste de Johansen <font color='#E74C3C'>não fornecem evidências suficientes de cointegração entre EUR/CHF.</font>"
    text216 = """
<span style="color: #0068c9; font-weight: bold; font-size: 18px;">Análise</span>

Os resultados dos testes de Engle-Granger indicam que não há evidências suficientes de cointegração entre os preços de fechamento do EUR e CHF, sugerindo que as séries não compartilham um relacionamento estável de longo prazo que possa ser explorado para previsões futuras. Isso implica que as variações nos preços do EUR/CHF podem ser mais aleatórias ou influenciadas por fatores não capturados pela cointegração.

A próxima etapa crucial será a análise do spread entre EUR e CHF, que é a diferença entre seus preços de fechamento. O objetivo será investigar se há um relacionamento de longo prazo estável entre esses spreads, o que pode fornecer insights valiosos para estratégias de negociação baseadas em arbitragem ou outras oportunidades de investimento.
"""
    tit3 = 'Modelagem do Spread'
    text301 = "Spread entre EUR e CHF ao longo do tempo"
    text302 = "Spread"
    text303 = "Spread Original"
    text304 = "Spread Diferenciado"
    text305 = "Teste ADF para Estacionariedade do Spread"
    text306 = "► Os resultados do teste ADF indicam que o spread <span style='color: #32CD32;'>é estacionário</span>, sugerindo que não há necessidade de transformação adicional."
    text307 = "► Os resultados do teste ADF indicam que o spread <span style='color: #FF4500;'>não é estacionário</span>. Aplicando a diferenciação, o spread diferenciado é utilizado."
    text308 = "Teste ADF para Estacionariedade do Spread Diferenciado"
    text309 = "► Os resultados do teste ADF indicam que o spread diferenciado <span style='color: #32CD32;'>é estacionário</span>."
    text310 = "► Os resultados do teste ADF indicam que o spread diferenciado <span style='color: #FF4500;'>não é estacionário</span>."
    text311 = """
<span style="color: #0068c9; font-weight: bold; font-size: 18px;">Análise</span>

O gráfico do spread original entre EUR e CHF revela variações ao longo do período analisado, exibindo períodos de alta e baixa. Em 2024, observa-se que o spread está se estabilizando próximo a zero. Por outro lado, o spread diferenciado mostra-se estacionário, com valores próximos a zero após a aplicação da diferenciação.

Os resultados do teste de Engle-Granger indicam que o spread original entre EUR e CHF não é estacionário, destacando a necessidade de diferenciação para tornar a série estacionária. Após a diferenciação, o teste ADF confirma que o spread diferenciado é estacionário, o que implica que seus movimentos ao longo do tempo são mais previsíveis e menos sujeitos a tendências não estacionárias.

Agora é crucial realizar o Modelo de Séries Temporais para o Spread entre EUR e CHF para explorar essas dinâmicas de forma mais profunda. O objetivo principal deste modelo é capturar e modelar os padrões e comportamentos históricos do spread, permitindo a previsão de seus futuros movimentos. Isso é fundamental para investidores e analistas que desejam entender melhor as relações entre as duas moedas e identificar oportunidades de negociação baseadas em análises quantitativas robustas.

"""
    tit4 = 'Modelo de Séries Temporais para o Spread'
    text401 = "Modelo ARMA para o Spread"
    text402 = "Melhores parâmetros:"
    text403 = "Resíduos do Modelo ARMA"
    text404 = "Resíduos"
    text405 = "Data"
    text406 = "O modelo ARMA ajustado ao spread sugere a seguinte interpretação:"
    text4071 = "Parâmetro AR (p):"
    text4072 = "indica que o spread é influenciado por"
    text4073 = "períodos passados."
    text4081 = "Parâmetro MA (q):"
    text4082 = "indica que o spread é influenciado por"
    text4083 = "períodos de erro passados."
    text409 = "Os resíduos do modelo ARMA mostram a qualidade do ajuste e a potencial previsibilidade do spread."
    text410 = """
<span style="color: #0068c9; font-weight: bold; font-size: 18px;">Análise</span>

O modelo ARMA é uma ferramenta estatística utilizada para analisar e prever séries temporais, como o spread entre duas moedas. Aqui estão os principais pontos práticos dos resultados obtidos:

**Influência dos Períodos Anteriores:** O parâmetro AR (Autoregressive) do modelo ARMA para o spread EUR/CHF foi estimado em 1. Isso significa que o valor atual do spread é significativamente influenciado pelo seu próprio valor anterior. Em termos práticos, se o spread entre EUR e CHF aumentou ou diminuiu no período anterior, isso terá um impacto direto no spread atual.

**Estacionariedade dos Resíduos:** Os resíduos do modelo ARMA, que representam as diferenças entre os valores observados e os valores previstos pelo modelo, mostraram-se estacionários em torno de zero. Isso é importante porque indica que o modelo captura bem a variabilidade dos dados sem deixar padrões não explicados nos resíduos.

**Previsibilidade do Spread:** Com base nos coeficientes estimados e na análise dos resíduos, o modelo ARMA sugere que o comportamento passado do spread pode ser usado para prever seu comportamento futuro com certa precisão. Isso é útil para traders e analistas financeiros que buscam antecipar movimentos no mercado de câmbio entre EUR e CHF.

**Interpretação dos Critérios de Informação:** Os critérios AIC, BIC e HQIC fornecem medidas para selecionar entre modelos alternativos. No caso apresentado, valores mais baixos desses critérios indicam que o modelo ARMA escolhido é o mais apropriado para os dados observados do spread EUR/CHF, considerando tanto a adequação do ajuste quanto a complexidade do modelo.

Em resumo, os resultados do modelo ARMA para o spread EUR/CHF fornecem uma estrutura sólida para entender como variações passadas influenciam o spread atual e para fazer previsões sobre seu comportamento futuro. Essa análise prática é essencial para tomar decisões informadas no mercado financeiro, desde estratégias de investimento até gestão de riscos cambiais.
"""
    tit5 = 'Considerações para uma Estratégia de Negociação'
    text51 = """
► Estratégia de Negociação Baseada no Spread

**Interpretação dos Resultados da Cointegração:**

**Spread Estacionário:** Após análise dos testes de Engle-Granger e Johansen, concluímos que não há evidências suficientes de cointegração entre EUR/CHF. Isso sugere que não existe uma relação de longo prazo estável entre essas moedas.

**Modelo ARMA para Estratégia de Negociação:**

**Melhores Parâmetros:** Com base no modelo ARMA (1,0), observamos que o modelo sugere uma dinâmica onde o spread entre EUR/CHF é influenciado principalmente pelo valor do período anterior, sem componente de média móvel.

**Implementação Prática:** O modelo ARMA nos fornece insights sobre como o spread entre EUR/CHF pode se comportar com base em dados históricos. Por exemplo, se observarmos um aumento no spread além do esperado pelo modelo, isso pode indicar uma oportunidade de negociação.

**Estratégia de Negociação Baseada no Modelo ARMA:**

**Análise de Desvios:** Para identificar oportunidades de arbitragem, podemos considerar vender EUR e comprar CHF quando o spread atual exceder um desvio positivo significativo em relação às previsões do modelo ARMA.

**Gestão de Risco:** Estabelecer limites de stop-loss e take-profit com base nas projeções do modelo ARMA ajuda a gerenciar riscos e capturar potenciais ganhos durante flutuações do spread.

**Monitoramento Contínuo e Ajustes:**

**Revisão Regular:** Manter uma vigilância constante sobre as mudanças no spread e ajustar a estratégia com base nas novas previsões do modelo ARMA é crucial para maximizar oportunidades de lucro e minimizar perdas.

**Flexibilidade:** Ajustar rapidamente a estratégia com base em novos dados e condições de mercado ajuda a adaptar-se às dinâmicas cambiantes do mercado EUR/CHF.

**Estratégia de Longo Prazo e Educação:**

**Diversificação e Conhecimento:** Combinação de estratégias baseadas em modelos estatísticos como ARMA com uma compreensão profunda dos fundamentos econômicos e geopolíticos fortalece a capacidade de tomar decisões informadas no mercado cambial.

**Investimento em Conhecimento:** Continuar a se educar em análise técnica, econômica e macroeconômica sustenta a habilidade de interpretar dados e tendências com precisão, essencial para uma negociação eficaz em EUR/CHF.

Esses conselhos práticos ajudam traders e investidores a aplicar o modelo ARMA de forma estratégica no mercado EUR/CHF, potencializando oportunidades de lucro e gerenciando eficazmente os riscos associados.
"""


# Configurar a interface do Streamlit
st.title(tit1)

# 1. Recolección de Datos----------------------------------------------------
# Obtén datos históricos de precios de cierre semanales para el par de divisas EUR/CHF.
# Puedes acceder a estos datos a través de fuentes financieras en línea o utilizando bibliotecas de Python
# como yfinance, pandas_datareader, etc.

# Instalar as bibliotecas necessárias
# !pip install yfinance pandas statsmodels streamlit

# Definir o intervalo de datas para os últimos 5 anos
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=5)

# Obter os dados históricos de preços de fechamento semanais para EUR/CHF
eur_chf_data = yf.download('EURCHF=X', start=start_date, end=end_date, interval='1wk')

# Selecionar apenas a coluna de preços de fechamento (Close)
eur_chf_close = eur_chf_data['Close']

# Calcular a média móvel simples (SMA) para suavizar os dados
sma_window = 10  # Janela da média móvel simples
sma = eur_chf_close.rolling(window=sma_window).mean()

# Criar figura Plotly
fig = go.Figure()

# Adicionar o gráfico de linhas para preços de fechamento
fig.add_trace(go.Scatter(x=eur_chf_close.index, y=eur_chf_close, mode='lines', name=text104))

# Adicionar a média móvel simples (SMA)
fig.add_trace(go.Scatter(x=eur_chf_close.index, y=sma, mode='lines', name=f'SMA {sma_window}'+ text105, line=dict(dash='dash')))

# Adicionar título e rótulos dos eixos
fig.update_layout(
    title=text101,
    xaxis_title=text103,
    yaxis_title=text102
)

# Adicionar modo hover para interatividade
fig.update_layout(hovermode='x unified')

# Exibir o gráfico Plotly no Streamlit
st.plotly_chart(fig)

# Aplicando o estilo CSS dentro do componente HTML do Streamlit
html_code = f"""
<style>
    .left-bar {{
        border-left: 6px solid #0068c9; /* Cor da barra lateral */
        padding-left: 15px; /* Espaçamento entre a barra e o texto */
        white-space: pre-wrap; /* Para manter as quebras de linha */
        width: 100%; /* Define a largura do conteúdo */
        text-align: justify; /* Alinhamento justificado */
    }}
    .left-bar span {{
        font-size: 18px; /* Tamanho da fonte da palavra "Análise" em pixels */
    }}
</style>
<div class="left-bar">
    {text106}
</div>
"""

# Exibindo o HTML no Streamlit
st.markdown(html_code, unsafe_allow_html=True)

# 2. Análisis de Cointegración-------------------------------------------------
# Realiza una prueba de cointegración de Engle-Granger
X = sm.add_constant(eur_chf_close)
result_engle = sm.OLS(eur_chf_close, X).fit()
residuals = result_engle.resid

# Realizar o teste ADF nos resíduos para verificar a estacionariedade
adf_result = sm.tsa.adfuller(residuals)


# Resultados do teste de Engle-Granger em colunas
st.write("")
st.write("")
st.write("")
st.markdown(f"<h2 style='text-align: center'>{tit2}</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader(text201)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(label=text202, value=f'{adf_result[0]:.4f}')
col2.metric(label=text203, value=f'{adf_result[1]:.4f}')
col3.metric(label=text204, value=f'{adf_result[4]["1%"]:.4f}')
col4.metric(label=text205, value=f'{adf_result[4]["5%"]:.4f}')
col5.metric(label=text206, value=f'{adf_result[4]["10%"]:.4f}')

# Análise e Conclusão do Teste de Engle-Granger
if adf_result[1] < 0.05:
    texto_eagle = text207

else:
    texto_eagle = text208
    
st.markdown(texto_eagle, unsafe_allow_html=True)

# Realizar o teste de cointegração de Johansen
data_matrix = eur_chf_close.to_numpy().reshape(-1, 1)
result_johansen = coint_johansen(data_matrix, det_order=-1, k_ar_diff=1)

# Resultados do teste de Johansen em colunas
st.subheader(text209)
col1, col2, col3, col4 = st.columns(4)
col1.metric(label=text210, value=f'{result_johansen.lr1[0]:.4f}')
col2.metric(label=text211, value=f'{result_johansen.cvt[0, 1]:.4f}')
col3.metric(label=text212, value=f'{result_johansen.lr2[0]:.4f}')
col4.metric(label=text213, value=f'{result_johansen.cvm[0, 1]:.4f}')

# Análise e Conclusão do Teste de Johansen
trace_statistic = result_johansen.lr1[0]
trace_crit_value = result_johansen.cvt[0, 1]
max_eigen_statistic = result_johansen.lr2[0]
max_eigen_crit_value = result_johansen.cvm[0, 1]

if trace_statistic > trace_crit_value and max_eigen_statistic > max_eigen_crit_value:
    texto_johansen = text214

else:
    texto_johansen = text215
    
st.markdown(texto_johansen, unsafe_allow_html=True)

st.write("")
st.write("")
# Aplicando o estilo CSS dentro do componente HTML do Streamlit
html_code = f"""
<style>
    .left-bar {{
        border-left: 6px solid #0068c9; /* Cor da barra lateral */
        padding-left: 15px; /* Espaçamento entre a barra e o texto */
        white-space: pre-wrap; /* Para manter as quebras de linha */
        width: 100%; /* Define a largura do conteúdo */
        text-align: justify; /* Alinhamento justificado */
    }}
    .left-bar span {{
        font-size: 18px; /* Tamanho da fonte da palavra "Análise" em pixels */
    }}
</style>
<div class="left-bar">
    {text216}
</div>
"""

# Exibindo o HTML no Streamlit
st.markdown(html_code, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.markdown(f"<h2 style='text-align: center'>{tit3}</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Obter os dados históricos de preços de fechamento semanais para EUR/CHF e CHF/EUR
eur_data = yf.download('EUR=X', start=start_date, end=end_date, interval='1wk')
chf_data = yf.download('CHF=X', start=start_date, end=end_date, interval='1wk')

# Extrair os preços de fechamento para EUR/CHF e CHF/EUR
eur_close = eur_data['Close']
chf_close = chf_data['Close']

# Calcular o spread (diferença entre os preços de fechamento do Euro e do Franco Suíço)
spread = eur_close - chf_close

# Realizar o teste ADF para verificar a estacionariedade do spread
adf_result = ts.adfuller(spread.dropna())

# Adicionar uma linha para diferenciar a série temporal
if adf_result[1] > 0.05:
    spread_diff = spread.diff().dropna()
    adf_result_diff = ts.adfuller(spread_diff)
else:
    spread_diff = spread
    adf_result_diff = adf_result

# Criar figura Plotly para o spread original e diferenciado
fig = go.Figure()

# Adicionar o gráfico de linha para o spread original
fig.add_trace(go.Scatter(x=spread.index, y=spread, mode='lines', name=text303))

# Adicionar o gráfico de linha para o spread diferenciado (se necessário)
if adf_result[1] > 0.05:
    fig.add_trace(go.Scatter(x=spread_diff.index, y=spread_diff, mode='lines', name=text304))

# Adicionar título e rótulos dos eixos
fig.update_layout(
    title=text301,
    xaxis_title= text103,
    yaxis_title=text302,
    hovermode='x unified'  # Modo hover para interatividade
)

# Exibir o gráfico Plotly no Streamlit
st.plotly_chart(fig)

# Mostrar os resultados do teste ADF em cards
st.subheader(text305)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(label=text202, value=f'{adf_result[0]:.4f}')
col2.metric(label=text203, value=f'{adf_result[1]:.4f}')
col3.metric(label=text204, value=f'{adf_result[4]["1%"]:.4f}')
col4.metric(label=text205, value=f'{adf_result[4]["5%"]:.4f}')
col5.metric(label=text206, value=f'{adf_result[4]["10%"]:.4f}')

# Mostrar a interpretação dos resultados do teste ADF
if adf_result[1] < 0.05:
    st.markdown(text306, unsafe_allow_html=True)
else:
    st.markdown(text307, unsafe_allow_html=True)

# Mostrar os resultados do teste ADF para o spread diferenciado (se necessário)
if adf_result[1] > 0.05:
    st.subheader(text308)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=text202, value=f'{adf_result_diff[0]:.4f}')
    col2.metric(label=text203, value=f'{adf_result_diff[1]:.4f}')
    col3.metric(label=text204, value=f'{adf_result_diff[4]["1%"]:.4f}')
    col4.metric(label=text205, value=f'{adf_result_diff[4]["5%"]:.4f}')
    col5.metric(label=text206, value=f'{adf_result_diff[4]["10%"]:.4f}')

    if adf_result_diff[1] < 0.05:
        st.markdown(text309, unsafe_allow_html=True)
    else:
        st.markdown(text310, unsafe_allow_html=True)
st.write("")
st.write("")
# Aplicando o estilo CSS dentro do componente HTML do Streamlit
html_code = f"""
<style>
    .left-bar {{
        border-left: 6px solid #0068c9; /* Cor da barra lateral */
        padding-left: 15px; /* Espaçamento entre a barra e o texto */
        white-space: pre-wrap; /* Para manter as quebras de linha */
        width: 100%; /* Define a largura do conteúdo */
        text-align: justify; /* Alinhamento justificado */
    }}
    .left-bar span {{
        font-size: 18px; /* Tamanho da fonte da palavra "Análise" em pixels */
    }}
</style>
<div class="left-bar">
    {text311}
</div>
"""

# Exibindo o HTML no Streamlit
st.markdown(html_code, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.markdown(f"<h2 style='text-align: center'>{tit4}</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Ajustar um modelo ARMA para o spread
# Selecionar os melhores parâmetros p e q utilizando AIC
arma_order = arma_order_select_ic(spread.dropna(), ic='aic', max_ar=5, max_ma=5)

# Extrair os melhores parâmetros p e q
p = arma_order.aic_min_order[0]
q = arma_order.aic_min_order[1]

# Ajustar o modelo ARMA com os melhores parâmetros
arma_model = ARIMA(spread.dropna(), order=(p, 0, q)).fit()

# Mostrar o resumo do modelo ARMA
st.subheader(text401)
st.write(f'{text402} p = {p}, q = {q}')
st.write(arma_model.summary())

# Plotar os resíduos do modelo ARMA
fig_resid = go.Figure()
fig_resid.add_trace(go.Scatter(x=spread.dropna().index, y=arma_model.resid, mode='lines', name='Resíduos'))

# Adicionar título e rótulos dos eixos
fig_resid.update_layout(
    title=text403,
    xaxis_title=text405,
    yaxis_title=text404
)

# Exibir o gráfico Plotly no Streamlit
st.plotly_chart(fig_resid)

st.write(text406)
st.write(f"1. **{text4071}** {p} {text4072} {p} {text4073}")
st.write(f"2. **{text4081}** {q} {text4082} {q} {text4083}")
st.write(text409)

st.write("")
st.write("")
# Aplicando o estilo CSS dentro do componente HTML do Streamlit
html_code = f"""
<style>
    .left-bar {{
        border-left: 6px solid #0068c9; /* Cor da barra lateral */
        padding-left: 15px; /* Espaçamento entre a barra e o texto */
        white-space: pre-wrap; /* Para manter as quebras de linha */
        width: 100%; /* Define a largura do conteúdo */
        text-align: justify; /* Alinhamento justificado */
    }}
    .left-bar span {{
        font-size: 18px; /* Tamanho da fonte da palavra "Análise" em pixels */
    }}
</style>
<div class="left-bar">
    {text410}
</div>
"""

# Exibindo o HTML no Streamlit
st.markdown(html_code, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.markdown(f"<h2 style='text-align: center'>{tit5}</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.write(text51)

# Espaçamento final
st.write("\n")
