# Questão 02: Implementação do Algoritmo Backpropagation Manual

## Introdução

Este projeto detalha a implementação do algoritmo **Backpropagation** do zero para uma Rede Neural Artificial do tipo MLP (Multi-Layer Perceptron) com uma camada oculta. O objetivo é treinar esta rede para aprender as funções lógicas booleanas **AND, OR e XOR** com um número **n** de entradas, que pode ser definido pelo usuário.

Além da implementação, este trabalho investiga experimentalmente a importância de três aspectos cruciais no treinamento de redes neurais:
1.  A **taxa de aprendizado (<span class="math-inline">\\eta</span>)**: Como diferentes valores afetam a convergência e o desempenho.
2.  O **bias**: O impacto de incluir ou omitir neurônios de bias nas camadas da rede.
3.  A **função de ativação**: Comparação do desempenho ao utilizar diferentes funções (pelo menos duas entre Sigmoide, Tangente Hiperbólica e ReLU) na camada oculta.

Este documento apresentará a teoria do Backpropagation, a arquitetura da rede implementada, os detalhes da implementação, os resultados dos experimentos e as conclusões obtidas. O código-fonte desenvolvido no Jupyter Notebook `Backpropagation_manual.ipynb` acompanha este relatório.

---

## 1. O Algoritmo Backpropagation e a Rede MLP

A Rede Neural Artificial implementada é uma **MLP (Multi-Layer Perceptron)** com uma camada de entrada, uma camada oculta e uma camada de saída. O aprendizado é realizado pelo algoritmo Backpropagation, que ajusta os pesos e biases da rede para minimizar uma função de custo (erro).

### 1.1. Arquitetura da Rede

* **Camada de Entrada**: Recebe <span class="math-inline">n</span> características booleanas (0 ou 1).
* **Camada Oculta**: Número configurável de neurônios (ex: <span class="math-inline">N\_h</span>), cada um aplicando uma função de ativação não-linear (Sigmoide, Tanh ou ReLU).
* **Camada de Saída**: Um único neurônio (para classificação binária) que aplica uma função de ativação (ex: Sigmoide) para produzir uma probabilidade entre 0 e 1.
* **Pesos e Biases**:
    * <span class="math-inline">W\_1</span>: Matriz de pesos conectando a camada de entrada à camada oculta.
    * <span class="math-inline">b\_1</span>: Vetor de biases para os neurônios da camada oculta (se `usar_bias=True`).
    * <span class="math-inline">W\_2</span>: Matriz de pesos conectando a camada oculta à camada de saída.
    * <span class="math-inline">b\_2</span>: Bias para o neurônio da camada de saída (se `usar_bias=True`).
    Os biases são implementados adicionando uma entrada fixa de valor 1 a cada camada e um peso correspondente.

### 1.2. Funções de Ativação Implementadas

* **Sigmoide**: <span class="math-inline">\\sigma\(x\) \= \\frac\{1\}\{1 \+ e^\{\-x\}\}</span>. Saída: (0, 1). Derivada: <span class="math-inline">\\sigma'\(x\) \= \\sigma\(x\)\(1 \- \\sigma\(x\)\)</span>.
* **Tangente Hiperbólica (Tanh)**: <span class="math-inline">\\tanh\(x\) \= \\frac\{e^x \- e^\{\-x\}\}\{e^x \+ e^\{\-x\}\}</span>. Saída: (-1, 1). Derivada: <span class="math-inline">\\tanh'\(x\) \= 1 \- \\tanh^2\(x\)</span>.
* **ReLU (Rectified Linear Unit)**: <span class="math-inline">ReLU\(x\) \= \\max\(0, x\)</span>. Saída: <span class="math-inline">\[0, \\infty\)</span>. Derivada: <span class="math-inline">ReLU'\(x\) \= 1</span> se <span class="math-inline">x \> 0</span>, <span class="math-inline">0</span> caso contrário.
* **Linear**: <span class="math-inline">f\(x\) \= x</span>. Saída: <span class="math-inline">\(\-\\infty, \\infty\)</span>. Derivada: <span class="math-inline">f'\(x\) \= 1</span>.

### 1.3. Processo de Aprendizagem (Feedforward e Backpropagation)

**a) Feedforward (Propagação Direta):**

Para cada amostra de entrada <span class="math-inline">X</span>:
1.  Entrada da camada oculta: <span class="math-inline">Z\_1 \= X \\cdot W\_1 \(\+ b\_1\)</span>
2.  Saída da camada oculta: <span class="math-inline">A\_1 \= \\text\{ativacao\_oculta\}\(Z\_1\)</span>
3.  Entrada da camada de saída: <span class="math-inline">Z\_2 \= A\_1 \\cdot W\_2 \(\+ b\_2\)</span>
4.  Saída final (predição): <span class="math-inline">\\hat\{y\} \= A\_2 \= \\text\{ativacao\_saida\}\(Z\_2\)</span>

**b) Função de Custo (Erro):**

Utilizamos a **Entropia Cruzada Binária** para problemas de classificação binária:
<span class="math-block">J\(W,b\) \= \- \\frac\{1\}\{m\} \\sum\_\{i\=1\}^\{m\} \[y\_i \\log\(\\hat\{y\}\_i\) \+ \(1\-y\_i\) \\log\(1\-\\hat\{y\}\_i\)\]</span>
onde <span class="math-inline">m</span> é o número de amostras, <span class="math-inline">y\_i</span> é o valor real e <span class="math-inline">\\hat\{y\}\_i</span> é a predição.

**c) Backpropagation (Retropropagação do Erro):**

O algoritmo calcula os gradientes da função de custo em relação a cada peso e bias, propagando o erro da camada de saída para as camadas anteriores.
1.  **Erro na Camada de Saída (<span class="math-inline">\\delta\_2</span>)**:
    Se a ativação da saída é Sigmoide e o custo é Entropia Cruzada, <span class="math-inline">\\delta\_2 \= \(\\hat\{y\} \- y\)</span>.
    De forma geral: <span class="math-inline">\\delta\_2 \= \(\\hat\{y\} \- y\) \\cdot \\text\{derivada\_ativacao\_saida\}\(Z\_2\)</span>.
2.  **Gradientes para Pesos e Bias da Camada de Saída (<span class="math-inline">\\Delta W\_2, \\Delta b\_2</span>)**:
    <span class="math-inline">\\Delta W\_2 \= \\frac\{1\}\{m\} A\_1^T \\cdot \\delta\_2</span>
    (O gradiente do bias está incluído em <span class="math-inline">\\Delta W\_2</span> se <span class="math-inline">A\_1</span> contiver a entrada de bias).
3.  **Erro na Camada Oculta (<span class="math-inline">\\delta\_1</span>)**:
    <span class="math-inline">\\delta\_1 \= \(\\delta\_2 \\cdot W\_2^T\_\{\\text\{sem\_bias\_saida\}\}\) \\cdot \\text\{derivada\_ativacao\_oculta\}\(Z\_1\)</span>.
4.  **Gradientes para Pesos e Bias da Camada Oculta (<span class="math-inline">\\Delta W\_1, \\Delta b\_1</span>)**:
    <span class="math-inline">\\Delta W\_1 \= \\frac\{1\}\{m\} X^T \\cdot \\delta\_1</span>
    (O gradiente do bias está incluído em <span class="math-inline">\\Delta W\_1</span> se <span class="math-inline">X</span> contiver a entrada de bias).

**d) Atualização dos Pesos e Biases:**

Os pesos e biases são ajustados na direção oposta ao gradiente (Descida do Gradiente):
<span class="math-inline">W \\leftarrow W \- \\eta \\Delta W</span>
<span class="math-inline">b \\leftarrow b \- \\eta \\Delta b</span>
(onde <span class="math-inline">\\eta</span> é a taxa de aprendizado).

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

### 3.1. Investigação da Taxa de Aprendizado (<span class="math-inline">\\eta</span>)
* **Objetivo**: Observar como diferentes taxas de aprendizado afetam a velocidade de convergência, a estabilidade do treinamento e o custo final para a função XOR.
* **Metodologia**: Treinou-se a rede para a função XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, com bias) utilizando as seguintes taxas de aprendizado: `[0.001, 0.01, 0.1, 0.5, 1.0, 2.0]`. O número de épocas foi fixado em 3000.
* **Resultados**:
    * *(Substitua esta linha e as seguintes pelos seus resultados e observações)*
    * Taxas muito baixas (ex: 0.001): Convergência lenta.
    * Taxas adequadas (ex: 0.1, 0.5): Boa convergência.
    * Taxas muito altas (ex: 1.0, 2.0): Treinamento instável ou divergente.
* **Gráficos**:
    *(Insira aqui o gráfico comparativo das curvas de custo para diferentes taxas de aprendizado gerado pelo seu notebook. Exemplo de como incluir uma imagem no Markdown:)*
    ```
    ```
    *(Você precisará fazer upload da imagem para o GitHub ou usar um link externo)*
* **Análise**: *(Discuta suas observações sobre o impacto da taxa de aprendizado com base nos seus gráficos e dados.)*

### 3.2. Investigação da Importância do Bias
* **Objetivo**: Verificar o impacto da inclusão ou omissão dos neurônios de bias no aprendizado da função XOR.
* **Metodologia**: Treinou-se a rede para XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, taxa de aprendizado de 0.1, 3000 épocas) em duas configurações: (1) com bias e (2) sem bias.
* **Resultados**:
    * *(Substitua esta linha e as seguintes pelos seus resultados e observações)*
    * XOR com Bias: Acurácia = XX.X%, Custo Final = Y.YYY
    * XOR sem Bias: Acurácia = AA.A%, Custo Final = B.BBB
* **Gráficos**:
    *(Insira aqui os gráficos das curvas de custo para os casos com e sem bias. Exemplo:)*
    ```
    ```
* **Análise**: *(Discuta por que o bias é importante, especialmente para problemas como o XOR, com base nos seus resultados.)*

### 3.3. Investigação da Importância da Função de Ativação (Camada Oculta)
* **Objetivo**: Comparar o desempenho da rede ao usar diferentes funções de ativação na camada oculta (Sigmoide, Tanh, ReLU) para aprender a função XOR.
* **Metodologia**: Treinou-se a rede para XOR (2 entradas, 4 neurônios ocultos, taxa de aprendizado de 0.1, 3000 épocas, com bias, saída sigmoide) variando a função de ativação da camada oculta.
* **Resultados**:
    * *(Substitua esta linha e as seguintes pelos seus resultados e observações)*
    * XOR com Sigmoide Oculta: Acurácia = XX.X%, Custo Final = Y.YYY
    * XOR com Tanh Oculta: Acurácia = AA.A%, Custo Final = B.BBB
    * XOR com ReLU Oculta: Acurácia = CC.C%, Custo Final = D.DDD
* **Gráficos**:
    *(Insira aqui os gráficos das curvas de custo para cada função de ativação. Exemplo:)*
    ```
    ```
* **Análise**: *(Discuta as diferenças observadas no treinamento e desempenho. Mencione prós e contras de cada função com base nos seus resultados, como velocidade de convergência, custo final, ou possíveis problemas como "dying ReLU".)*

### 3.4. Testes com Funções Lógicas (AND, OR, XOR) e `n` Entradas
* **Objetivo**: Demonstrar a capacidade da rede MLP com Backpropagation de aprender as funções AND, OR e XOR para diferentes números de entradas.
* **Resultados**:
    *(Descreva resumidamente se a rede conseguiu aprender cada função para os `n` testados.)*
* **Tabela de Resultados (Preencha com seus dados do `Backpropagation_manual.ipynb`)**:

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
[Backpropagation_manual.ipynb](https://github.com/LuanCarrieiros/IA/blob/main/Lista10/Questão%2002/Backpropagation_manual.ipynb)

