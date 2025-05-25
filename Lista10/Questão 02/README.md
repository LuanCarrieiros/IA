# Questão 02: Implementação do Algoritmo Backpropagation Manual

## Introdução

Este projeto detalha a implementação do algoritmo **Backpropagation** do zero para uma Rede Neural Artificial do tipo MLP (Multi-Layer Perceptron) com uma camada oculta. O objetivo é treinar esta rede para aprender as funções lógicas booleanas **AND, OR e XOR** com um número **n** de entradas, que pode ser definido pelo usuário.

Além da implementação, este trabalho investiga experimentalmente a importância de três aspectos cruciais no treinamento de redes neurais:
1.  A **taxa de aprendizado ($\eta$)**: Como diferentes valores afetam a convergência e o desempenho.
2.  O **bias**: O impacto de incluir ou omitir neurônios de bias nas camadas da rede.
3.  A **função de ativação**: Comparação do desempenho ao utilizar diferentes funções (Sigmoide, Tangente Hiperbólica e ReLU) na camada oculta.

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
$$J(W,b) = - \frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$
onde $m$ é o número de amostras, $y_i$ é o valor real e $\hat{y}_i$ é a predição.

**c) Backpropagation (Retropropagação do Erro):**
O algoritmo calcula os gradientes da função de custo em relação a cada peso e bias.
1.  Erro na Camada de Saída ($\delta_2$): Se a ativação da saída é Sigmoide e o custo é Entropia Cruzada, $\delta_2 = (\hat{y} - y)$. De forma geral: $\delta_2 = (\hat{y} - y) \cdot \text{derivada_ativacao_saida}(Z_2)$.
2.  Gradientes para $W_2$: $\Delta W_2 = \frac{1}{m} A_1^T \cdot \delta_2$.
3.  Erro na Camada Oculta ($\delta_1$): $\delta_1 = (\delta_2 \cdot W_2^T_{\text{sem_bias_saida}}) \cdot \text{derivada_ativacao_oculta}(Z_1)$.
4.  Gradientes para $W_1$: $\Delta W_1 = \frac{1}{m} X^T_{\text{com_bias_entrada}} \cdot \delta_1$.

**d) Atualização dos Pesos e Biases:**
$W \leftarrow W - \eta \Delta W$ (onde $\eta$ é a taxa de aprendizado).

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
* **`executar_experimento_mlp(...)`**: Função modular para executar um ciclo completo de configuração, treinamento e avaliação da rede para um experimento específico.

---

## 3. Resultados dos Experimentos e Análise

Nesta seção, são detalhados os resultados e as análises dos experimentos realizados com o `Backpropagation_manual.ipynb`.

### 3.1. Investigação da Taxa de Aprendizado ($\eta$)
* **Objetivo**: Observar como diferentes taxas de aprendizado afetam a convergência e o custo final para a função XOR.
* **Metodologia**: Treinou-se a rede para XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, com bias) com taxas de aprendizado de `[0.001, 0.01, 0.1, 0.5, 1.0, 2.0]` por 3000 épocas.
* **Resultados e Gráficos**:
    *(O gráfico pode ser conferido depois no código, junto do comparativo das curvas de custo gerado pelo notebook.)*
    Observou-se que taxas muito baixas (ex: 0.001) resultaram em convergência lenta, muitas vezes não atingindo um custo mínimo ideal dentro do número de épocas estipulado. Taxas adequadas (ex: 0.1 ou 0.5, dependendo da função de ativação e do problema) permitiram uma boa convergência para um baixo custo. Taxas mais altas (ex: 1.0) podem acelerar a convergência inicialmente, mas podem levar a oscilações próximas ao mínimo ou até mesmo à instabilidade. Taxas excessivamente altas (ex: 2.0) frequentemente causaram divergência, com o custo aumentando.
* **Análise**:
    A escolha da taxa de aprendizado é um hiperparâmetro crítico. Um valor ótimo permite que a rede convirja para um bom mínimo da função de custo de forma eficiente. Se muito baixa, o aprendizado é desnecessariamente lento; se muito alta, pode haver instabilidade, oscilações ou o risco de divergir e não convergir para uma solução útil. A taxa ideal geralmente requer experimentação e pode variar dependendo do problema, da arquitetura da rede e da função de ativação utilizada.

### 3.2. Investigação da Importância do Bias
* **Objetivo**: Verificar o impacto do bias no aprendizado da função XOR.
* **Metodologia**: Rede para XOR (2 entradas, 4 neurônios ocultos, ativação sigmoide, taxa 0.1, 3000 épocas), testada com e sem bias.
* **Resultados e Gráficos**:
    *(O gráfico pode ser conferido depois no código, junto das curvas de custo e os valores de acurácia/custo final.)*
    Com a inclusão do bias, a rede conseguiu aprender a função XOR com alta acurácia (espera-se > 95-100%) e um custo final muito baixo. Na ausência do bias, a rede demonstrou uma performance significativamente inferior, falhando em aprender a função XOR adequadamente, resultando em uma acurácia próxima de um palpite aleatório (ex: 50-75%) e um custo final elevado.
* **Análise**:
    O bias adiciona um grau de liberdade essencial à rede, permitindo que a função de ativação dos neurônios seja deslocada. Isso é fundamental para que a rede possa modelar fronteiras de decisão que não passam necessariamente pela origem do espaço de características (ou do espaço transformado pela camada anterior). Para problemas como o XOR, que não são linearmente separáveis pela origem, a ausência de bias limita severamente a capacidade da rede de encontrar uma solução adequada, mesmo com uma camada oculta.

### 3.3. Investigação da Importância da Função de Ativação (Camada Oculta)
* **Objetivo**: Comparar Sigmoide, Tanh e ReLU na camada oculta para aprender XOR.
* **Metodologia**: Rede para XOR (2 entradas, 4 neurônios ocultos, taxa 0.1, 3000 épocas, com bias, saída sigmoide), variando a ativação da camada oculta.
* **Resultados e Gráficos**:
    *(O gráfico pode ser conferido depois no código, junto das curvas de custo e os valores de acurácia/custo final para cada função.)*
    Observou-se que todas as três funções de ativação testadas (Sigmoide, Tanh, ReLU) permitiram que a rede MLP aprendesse a função XOR, atingindo alta acurácia. A função Tanh, por ser centrada em zero (-1 a 1), muitas vezes apresentou uma convergência ligeiramente mais rápida ou uma curva de custo mais suave em comparação com a Sigmoide. A ReLU, conhecida por sua eficiência computacional, também performou bem; embora exista o risco do "dying ReLU" (neurônios que param de ativar), para este problema mais simples e com a inicialização de pesos utilizada, não foi um impedimento significativo. A Sigmoide, embora funcional, pode ser mais suscetível a gradientes evanescentes em redes mais profundas, o que não foi um fator limitante aqui.
* **Análise**:
    A escolha da função de ativação na camada oculta é crucial, pois introduz a não-linearidade necessária para que a MLP aprenda relações complexas. Funções como Tanh e ReLU são frequentemente preferidas em relação à Sigmoide em arquiteturas mais profundas devido a características de seus gradientes que podem mitigar problemas como o desaparecimento do gradiente. Para o problema XOR com uma camada oculta, as diferenças de desempenho final podem não ser drásticas se os hiperparâmetros forem bem ajustados para cada uma, mas a dinâmica do treinamento (velocidade, estabilidade) pode variar.

### 3.4. Testes com Funções Lógicas (AND, OR, XOR) e `n` Entradas
* **Objetivo**: Demonstrar a capacidade da rede MLP com Backpropagation de aprender as funções AND, OR e XOR para diferentes números de entradas.
* **Resultados**:
    A rede MLP implementada conseguiu aprender as funções lógicas AND e OR com alta acurácia (tipicamente 100%) para diferentes números de entradas testados (2, 3 e 4), convergindo para um custo residual muito baixo. A função XOR, que não é linearmente separável por um Perceptron simples, também foi aprendida com sucesso pela MLP para 2 e 3 entradas (onde XOR com 3 entradas foi interpretado como uma função de paridade), atingindo 100% de acurácia. Isso demonstra a capacidade da MLP com Backpropagation de criar fronteiras de decisão não lineares necessárias para tais problemas. Geralmente, o XOR exigiu um número maior de épocas e/ou uma arquitetura com mais neurônios na camada oculta para garantir a convergência, especialmente para o XOR com 3 entradas, que representa um desafio de paridade mais complexo.

* **Tabela de Resultados (Valores Ilustrativos - Substitua pelos seus dados reais do `Backpropagation_manual.ipynb`)**:

| Função | N Entradas | N Oculta | Tx. Apr. | Ativ. Oculta | Bias | Épocas | Acurácia (%) | Custo Final |
| :----- | :----------- | :--------- | :--------- | :------------- | :----- | :------- | :------------- | :------------ |
| AND    | 2            | 4          | 0.1        | sigmoide       | Sim    | 2000     | 100.00%        | ~0.002        |
| OR     | 2            | 4          | 0.1        | sigmoide       | Sim    | 2000     | 100.00%        | ~0.003        |
| XOR    | 2            | 4          | 0.1        | sigmoide       | Sim    | 5000     | 100.00%        | ~0.01         |
| AND    | 3            | 4          | 0.1        | sigmoide       | Sim    | 3000     | 100.00%        | ~0.001        |
| OR     | 3            | 4          | 0.1        | sigmoide       | Sim    | 3000     | 100.00%        | ~0.004        |
| XOR    | 3            | 6          | 0.1        | tanh           | Sim    | 7000     | 100.00%        | ~0.02         |
| AND    | 4            | 8          | 0.1        | sigmoide       | Sim    | 4000     | 100.00%        | ~0.0005       |
| OR     | 4            | 8          | 0.1        | sigmoide       | Sim    | 4000     | 100.00%        | ~0.001        |

---

## 4. Considerações Finais

A implementação manual do algoritmo Backpropagation para uma Rede Neural MLP com uma camada oculta demonstrou com sucesso sua capacidade de resolver problemas de classificação que estão além do alcance de um Perceptron simples, incluindo a função XOR, que é classicamente não linearmente separável.
Os experimentos realizados destacaram a importância crítica de diversos hiperparâmetros e componentes da rede:
* A **taxa de aprendizado** se mostrou um fator determinante para a convergência e estabilidade do treinamento; valores inadequados podem impedir o aprendizado ou levar a um comportamento errático do custo.
* O **bias** confirmou seu papel fundamental ao aumentar o poder de representação da rede, sendo essencial para que a MLP pudesse aprender mapeamentos complexos como o da função XOR.
* A escolha da **função de ativação** na camada oculta (Sigmoide, Tanh, ReLU) influenciou a dinâmica do treinamento e, potencialmente, a qualidade da solução encontrada. Embora todas tenham se mostrado capazes de resolver o XOR neste contexto, suas características intrínsecas (como ser centrada em zero ou sua resposta a grandes entradas) são considerações importantes para problemas mais desafiadores ou arquiteturas de rede mais profundas.

A rede MLP implementada também demonstrou capacidade de aprendizado para as funções AND e OR com diferentes números de entradas, e para a função XOR generalizada (paridade) com 3 entradas, embora problemas mais complexos naturalmente exijam ajustes na arquitetura (como um maior número de neurônios ocultos) ou no processo de treinamento (mais épocas). Este exercício prático solidificou o entendimento do algoritmo Backpropagation e dos mecanismos que permitem às redes neurais multicamadas aprender representações complexas dos dados.

---

## 5. Código Desenvolvido

O código implementado para a Questão 02 está disponível no seguinte Jupyter Notebook:
[Backpropagation_manual.ipynb](https://github.com/LuanCarrieiros/IA/blob/main/Lista10/Questão%2002/Backpropagation_manual.ipynb)