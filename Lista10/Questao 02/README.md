# Questão 02: Implementação do Algoritmo Backpropagation Manual

## Introdução

Este projeto detalha a implementação do algoritmo **Backpropagation** do zero para uma Rede Neural Artificial do tipo MLP (Multi-Layer Perceptron) com uma camada oculta. O objetivo é treinar esta rede para aprender as funções lógicas booleanas **AND, OR e XOR** com um número **n** de entradas, que pode ser definido pelo usuário.

Além da implementação, este trabalho investiga experimentalmente a importância de três aspectos cruciais no treinamento de redes neurais:
1.  A **taxa de aprendizado ($\eta$)**: Como diferentes valores afetam a convergência e o desempenho.
2.  O **bias**: O impacto de incluir ou omitir neurônios de bias nas camadas da rede.
3.  A **função de ativação**: Comparação do desempenho ao utilizar diferentes funções (pelo menos duas entre Sigmoide, Tangente Hiperbólica e ReLU) na camada oculta.

Este documento apresentará a teoria do Backpropagation, a arquitetura da rede implementada, os detalhes da implementação, os resultados dos experimentos e as conclusões obtidas. O código-fonte desenvolvido no Jupyter Notebook `Backpropagation_manual.ipynb` acompanha este relatório.

---

## 1. O Algoritmo Backpropagation e a Rede MLP

A Rede Neural Artificial implementada é uma **MLP (Multi-Layer Perceptron)** com uma camada de entrada, uma camada oculta e uma camada de saída. O aprendizado é realizado pelo algoritmo Backpropagation, que ajusta os pesos e biases da rede para minimizar uma função de custo (erro).

### 1.1. Arquitetura da Rede
* **Camada de Entrada**: Recebe $n$ características booleanas (0 ou 1).
* **Camada Oculta**: Número configurável de neurônios (ex: $N_h$), cada um aplicando uma função de ativação não-linear (Sigmoide, Tanh ou ReLU).
* **Camada de Saída**: Um único neurônio (para classificação binária) que aplica uma função de ativação (ex: Sigmoide) para produzir uma probabilidade entre 0 e 1.
* **Pesos e Biases**:
    * $W_1$: Matriz de pesos conectando a camada de entrada à camada oculta.
    * $b_1$: Vetor de biases para os neurônios da camada oculta (se `usar_bias=True`).
    * $W_2$: Matriz de pesos conectando a camada oculta à camada de saída.
    * $b_2$: Bias para o neurônio da camada de saída (se `usar_bias=True`).
    Os biases são implementados adicionando uma entrada fixa de valor 1 a cada camada e um peso correspondente.

### 1.2. Funções de Ativação Implementadas
* **Sigmoide**: $\sigma(x) = \frac{1}{1 + e^{-x}}$. Saída: (0, 1). Derivada: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.
* **Tangente Hiperbólica (Tanh)**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Saída: (-1, 1). Derivada: $\tanh'(x) = 1 - \tanh^2(x)$.
* **ReLU (Rectified Linear Unit)**: $ReLU(x) = \max(0, x)$. Saída: $[0, \infty)$. Derivada: $ReLU'(x) = 1$ se $x > 0$, $0$ caso contrário.
* **Linear**: $f(x) = x$. Saída: $(-\infty, \infty)$. Derivada: $f'(x) = 1$.

### 1.3. Processo de Aprendizagem (Feedforward e Backpropagation)

**a) Feedforward (Propagação Direta):**
Para cada amostra de entrada $X$:
1.  Entrada da camada oculta: $Z_1 = X \cdot W_1 (+ b_1)$
2.  Saída da camada oculta: $A_1 = \text{ativacao_oculta}(Z_1)$
3.  Entrada da camada de saída: $Z_2 = A_1 \cdot W_2 (+ b_2)$
4.  Saída final (predição): $\hat{y} = A_2 = \text{ativacao_saida}(Z_2)$

**b) Função de Custo (Erro):**
Utilizamos a **Entropia Cruzada Binária** para problemas de classificação binária:
$J(W,b) = - \frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$
onde $m$ é o número de amostras, $y_i$ é o valor real e $\hat{y}_i$ é a predição.

**c) Backpropagation (Retropropagação do Erro):**
O algoritmo calcula os gradientes da função de custo em relação a cada peso e bias, propagando o erro da camada de saída para as camadas anteriores.
1.  **Erro na Camada de Saída ($\delta_2$)**:
    Se a ativação da saída é Sigmoide e o custo é Entropia Cruzada, $\delta_2 = (\hat{y} - y)$.
    De forma geral: $\delta_2 = (\hat{y} - y) \cdot \text{derivada_ativacao_saida}(Z_2)$.
2.  **Gradientes para Pesos e Bias da Camada de Saída ($\Delta W_2, \Delta b_2$)**:
    $\Delta W_2 = \frac{1}{m} A_1^T \cdot \delta_2$
    (O gradiente do bias está incluído em $\Delta W_2$ se $A_1$ contiver a entrada de bias).
3.  **Erro na Camada Oculta ($\delta_1$)**:
    $\delta_1 = (\delta_2 \cdot W_2^T_{\text{sem_bias_saida}}) \cdot \text{derivada_ativacao_oculta}(Z_1)$.
4.  **Gradientes para Pesos e Bias da Camada Oculta ($\Delta W_1, \Delta b_1$)**:
    $\Delta W_1 = \frac{1}{m} X^T \cdot \delta_1$
    (O gradiente do bias está incluído em $\Delta W_1$ se $X$ contiver a entrada de bias).

**d) Atualização dos Pesos e Biases:**
Os pesos e biases são ajustados na direção oposta ao gradiente (Descida do Gradiente):
$W = W - \eta \Delta W$
$b = b - \eta \Delta b$
(onde $\eta$ é a taxa de aprendizado).

Este ciclo de feedforward, cálculo de custo, backpropagation e atualização é repetido por um número definido de épocas.

---

## 2. Estrutura da Implementação (Notebook `Backpropagation_manual.ipynb`)

O código é implementado em um Jupyter Notebook e inclui:

### 2.1. Classe `RedeNeuralMLP`
Encapsula toda a lógica da rede neural:
* **`__init__(...)`**: Inicializa a arquitetura, pesos, biases e define as funções de ativação.
* **Funções de Ativação Privadas**: `_sigmoide`, `_tanh`, `_relu`, `_linear` e suas respectivas derivadas.
* **`_adicionar_bias_a_entrada(...)`**: Método auxiliar para incluir o termo de bias.
* **`feedforward(X_entrada_batch)`**: Realiza a propagação direta.
* **`_calcular_custo(y_predito, y_real)`**: Calcula o erro de entropia cruzada.
* **`backward(X_entrada_batch, y_real, y_predito, taxa_aprendizado)`**: Implementa o backpropagation e a atualização dos pesos.
* **`train(X_treino, y_treino, taxa_aprendizado, n_epocas_treino)`**: Orquestra o processo de treinamento.
* **`predict_proba(X_teste)`** e **`predict_classes(X_teste, limiar)`**: Para fazer predições.

### 2.2. Funções Auxiliares
* **`gerar_dados(n_entradas, tipo_funcao)`**: Gera os conjuntos de dados para AND, OR e XOR com `n` entradas.
* **`plotar_curva_erro(erros_por_epoca, titulo_plot)`**: Plota o custo da rede ao longo das épocas de treinamento.
* **`executar_experimento_mlp(...)`**: Função modular para executar um ciclo completo de configuração, treinamento e avaliação da rede para um experimento específico, facilitando as investigações.

---

## 3. Resultados dos Experimentos e Análise

*(Nesta seção, você deverá detalhar os resultados e as análises dos experimentos realizados com o `Backpropagation_manual.ipynb`. Para cada investigação, apresente os resultados (ex: gráficos de curva de erro, acurácias finais) e discuta as observações.)*

### 3.1. Investigação da Taxa de Aprendizado ($\eta$)
* **Objetivo**: Observar como diferentes taxas de aprendizado afetam a velocidade de convergência, a estabilidade do treinamento e o custo final para a função XOR.
* **Metodologia**: Treinou-se a rede para a função XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, com bias) utilizando as seguintes taxas de aprendizado: `[0.001, 0.01, 0.1, 0.5, 1.0, 2.0]`. O número de épocas foi fixado em 3000.
* **Resultados Esperados (Preencha com seus resultados)**:
    * **Taxas muito baixas (ex: 0.001)**: Convergência lenta, custo final pode ser alto dentro do número de épocas.
    * **Taxas adequadas (ex: 0.1, 0.5)**: Boa convergência, atingindo um baixo custo.
    * **Taxas muito altas (ex: 1.0, 2.0)**: O treinamento pode se tornar instável (custo oscilando muito) ou divergir (custo aumentando), impedindo a rede de aprender.
* **Gráficos**: *(Insira aqui o gráfico comparativo das curvas de custo para diferentes taxas de aprendizado gerado pelo seu notebook, ou um resumo)*
* **Análise**: *(Discuta suas observações sobre o impacto da taxa de aprendizado)*

### 3.2. Investigação da Importância do Bias
* **Objetivo**: Verificar o impacto da inclusão ou omissão dos neurônios de bias no aprendizado da função XOR.
* **Metodologia**: Treinou-se a rede para XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, taxa de aprendizado de 0.1, 3000 épocas) em duas configurações: (1) com bias e (2) sem bias.
* **Resultados Esperados (Preencha com seus resultados)**:
    * **Com Bias**: A rede deve ser capaz de aprender a função XOR com alta acurácia.
    * **Sem Bias**: A rede provavelmente terá dificuldade em aprender XOR, resultando em baixa acurácia ou não convergência. Isso ocorre porque o bias permite que a função de ativação seja deslocada, o que é crucial para separar dados que não são linearmente separáveis pela origem.
* **Dados/Gráficos**: *(Apresente a acurácia final e as curvas de custo para os dois casos. Ex:)*
    * XOR com Bias: Acurácia = XX.X%, Custo Final = Y.YYY
    * XOR sem Bias: Acurácia = AA.A%, Custo Final = B.BBB
* **Análise**: *(Discuta por que o bias é importante, especialmente para problemas como o XOR)*

### 3.3. Investigação da Importância da Função de Ativação (Camada Oculta)
* **Objetivo**: Comparar o desempenho da rede ao usar diferentes funções de ativação na camada oculta (Sigmoide, Tanh, ReLU) para aprender a função XOR.
* **Metodologia**: Treinou-se a rede para XOR (2 entradas, 4 neurônios ocultos, taxa de aprendizado de 0.1, 3000 épocas, com bias, saída sigmoide) variando a função de ativação da camada oculta.
* **Resultados Esperados (Preencha com seus resultados)**:
    * **Sigmoide**: Deve aprender XOR, mas pode sofrer com gradientes evanescentes se a rede for mais profunda (não é o nosso caso aqui) ou se os valores de entrada para a função forem muito grandes/pequenos.
    * **Tanh**: Frequentemente converge mais rápido que a sigmoide por ser centrada em zero. Também pode aprender XOR.
    * **ReLU**: Pode aprender XOR. É computacionalmente eficiente, mas pode sofrer com o problema do "dying ReLU" se os neurônios pararem de ativar (saída sempre zero).
* **Dados/Gráficos**: *(Apresente a acurácia final e as curvas de custo para cada função de ativação testada. Ex:)*
    * XOR com Sigmoide Oculta: Acurácia = XX.X%, Custo Final = Y.YYY
    * XOR com Tanh Oculta: Acurácia = AA.A%, Custo Final = B.BBB
    * XOR com ReLU Oculta: Acurácia = CC.C%, Custo Final = D.DDD
* **Análise**: *(Discuta as diferenças observadas no treinamento e desempenho com cada função de ativação. Mencione prós e contras de cada uma com base nos seus resultados.)*

### 3.4. Testes com Funções Lógicas (AND, OR, XOR) e `n` Entradas
* **Objetivo**: Demonstrar a capacidade da rede MLP com Backpropagation de aprender as funções AND, OR e XOR para diferentes números de entradas.
* **Resultados Esperados (Preencha com seus resultados)**:
    * A rede deve aprender AND e OR com alta acurácia para `n=2, 3, 4` (ou mais).
    * A rede deve aprender XOR com `n=2` e `n=3` (XOR generalizado como paridade) com alta acurácia, embora possa exigir mais épocas ou uma arquitetura/hiperparâmetros mais ajustados em comparação com AND/OR.
* **Tabela de Resultados (Exemplo - Preencha com seus dados)**:

| Função | N Entradas | N Oculta | Tx. Apr. | Ativ. Oculta | Bias | Épocas | Acurácia (%) | Custo Final |
|--------|------------|----------|----------|--------------|------|--------|--------------|-------------|
| AND    | 2          | 4        | 0.1      | sigmoide     | Sim  | 2000   | (seu valor)  | (seu valor) |
| OR     | 2          | 4        | 0.1      | sigmoide     | Sim  | 2000   | (seu valor)  | (seu valor) |
| XOR    | 2          | 4        | 0.1      | sigmoide     | Sim  | 5000   | (seu valor)  | (seu valor) |
| AND    | 3          | 4        | 0.1      | sigmoide     | Sim  | 3000   | (seu valor)  | (seu valor) |
| OR     | 3          | 4        | 0.1      | sigmoide     | Sim  | 3000   | (seu valor)  | (seu valor) |
| XOR    | 3          | 6        | 0.1      | tanh         | Sim  | 7000   | (seu valor)  | (seu valor) |
| AND    | 4          | 8        | 0.1      | sigmoide     | Sim  | 4000   | (seu valor)  | (seu valor) |
| OR     | 4          | 8        | 0.1      | sigmoide     | Sim  | 4000   | (seu valor)  | (seu valor) |

*(Adapte a tabela acima com as configurações e resultados que você obteve no seu notebook `Backpropagation_manual.ipynb`)*

---

## 4. Considerações Finais

*(Resuma as principais aprendizagens do desenvolvimento da Questão 02. Destaque:*
* *A capacidade do Backpropagation com uma MLP de resolver problemas não linearmente separáveis como o XOR.*
* *A sensibilidade do processo de treinamento à taxa de aprendizado.*
* *A função crítica do bias para permitir que a rede aprenda uma maior variedade de mapeamentos.*
* *Como a escolha da função de ativação pode impactar a eficiência e o sucesso do treinamento.*
* *Observações sobre a generalização da rede para diferentes números de entradas `n`.)*

---

## 5. Código Desenvolvido

O código implementado para a Questão 02 está disponível no seguinte Jupyter Notebook:
`Backpropagation_manual.ipynb` *(https://github.com/LuanCarrieiros/IA/blob/main/Lista10/Questao%2002/Backpropagation_manual.ipynb).*