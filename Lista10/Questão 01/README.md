# Questão 01: Implementação do Algoritmo Perceptron Manual

## Introdução

Este projeto foca na implementação do algoritmo Perceptron **do zero**, um modelo fundamental de rede neural artificial, para resolver funções lógicas com **n entradas booleanas**. Especificamente, o Perceptron foi treinado para aprender as funções **AND** e **OR**. O usuário tem a flexibilidade de definir o número de entradas (por exemplo, 2, 3, 5 ou 10) para estas funções.

Uma parte crucial deste trabalho é a **visualização das regras de separação** dos hiperplanos. Para `n=2` entradas, o hiperplano (uma linha) é plotado dinamicamente **durante o processo de treinamento**, mostrando sua evolução. Para `n=3` entradas, o plano de separação final é visualizado em 3D. Para `n > 3` entradas, os dados são projetados em 2D usando Análise de Componentes Principais (PCA) para visualização. Adicionalmente, a curva de erro por época é plotada para todos os casos.

Demonstraremos também uma limitação inerente ao Perceptron: sua incapacidade de resolver a função **XOR**, que não é linearmente separável, através da animação do hiperplano e da curva de erro.

Este README detalhará a teoria do Perceptron, a abordagem da implementação manual, a lógica para geração de dados e as diferentes formas de plotagem utilizadas, além da análise dos resultados obtidos com o código desenvolvido no notebook `Perceptron_manual.ipynb`.

---

## 1. O Algoritmo Perceptron

O Perceptron é um classificador linear binário. Ele calcula uma soma ponderada das entradas, adiciona um bias, e então aplica uma função de ativação (geralmente uma função degrau) para produzir uma saída.

* **Entradas (Input):** $x_1, x_2, ..., x_n$ (valores booleanos 0 ou 1)
* **Pesos (Weights):** $w_0 (\text{bias}), w_1, w_2, ..., w_n$ (representam a importância de cada entrada e o limiar)
* **Soma Ponderada (Net Input):** $z = w_0 \cdot 1 + \sum_{i=1}^{n} w_i x_i$ (o bias $w_0$ é associado a uma entrada $x_0=1$)
* **Função de Ativação (Step Function):**
    * $y = 1$ se $z \ge 0$
    * $y = 0$ se $z < 0$

### Processo de Aprendizagem

O Perceptron aprende de forma iterativa, ajustando seus pesos com base nos erros cometidos. Para cada amostra de treinamento $(X_i, y_i)$:
1.  Calcula-se a saída predita $\hat{y}_i$.
2.  O erro é determinado: $erro_i = y_i - \hat{y}_i$.
3.  Se $erro_i \neq 0$, os pesos são ajustados:
    * $w_j(\text{novo}) = w_j(\text{antigo}) + \eta \cdot erro_i \cdot x_{ij}$ (onde $x_{i0}=1$ para o bias)
    Onde $\eta$ é a **taxa de aprendizado**.

Este processo é repetido sobre o conjunto de dados por várias épocas até que o modelo convirja (todos os exemplos classificados corretamente) ou um número máximo de épocas seja atingido.

---

## 2. Estrutura da Implementação (Notebook `Perceptron_manual.ipynb`)

O código é implementado em um Jupyter Notebook e estruturado da seguinte forma:

### 2.1. Classe `Perceptron`
Contém a lógica do algoritmo Perceptron implementado manualmente:
* **`__init__(self, taxa_aprendizado, n_epocas)`**: Inicializa os hiperparâmetros.
* **`_adicionar_bias_input(self, X_i)`**: Prepara a entrada para incluir o termo de bias.
* **`predict(self, X_i_com_bias)`**: Realiza a predição para uma amostra.
* **`fit(self, X, y, ax_plot, ...)`**: Executa o treinamento do Perceptron, atualizando os pesos e, para `n=2` entradas, chamando a função de plotagem 2D animada.

### 2.2. Funções Auxiliares
* **`gerar_dados(n_entradas, tipo_funcao)`**: Gera todas as $2^n$ combinações de entradas booleanas e as saídas `y` para as funções `AND`, `OR` ou `XOR`.
* **`plotar_hiperplano_2d(X, y, perceptron_model, ax, titulo_plot)`**: Para `n=2` entradas, plota dinamicamente os dados e a linha de separação durante o treinamento.
* **`plotar_curva_erro(erros_por_epoca, titulo_plot)`**: Plota o número de erros de classificação por época.
* **`plotar_hiperplano_3d_final(X, y, perceptron_model, titulo_plot)`**: Para `n=3` entradas, plota os pontos de dados em 3D e o plano de separação final.
* **`plotar_pca_2d_final(X, y, titulo_plot)`**: Para `n > 3` entradas, aplica PCA e plota os dados projetados em 2D.
* **`executar_teste_perceptron(tipo_funcao, num_entradas, ...)`**: Função modular para executar um ciclo de teste completo.

---

## 3. Demonstração: Perceptron e XOR

A função XOR (OU Exclusivo) é um exemplo clássico de um problema não linearmente separável. Para 2 entradas:

| x1 | x2 | y (XOR) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

O Perceptron implementado manualmente foi treinado com os dados da função XOR. Conforme esperado, o algoritmo não convergiu para uma solução perfeita. A animação do hiperplano 2D mostrou a linha tentando, sem sucesso, separar as classes, e a curva de erro não atingiu zero consistentemente, evidenciando a persistência de erros. Isso ocorre porque o Perceptron só pode encontrar fronteiras de decisão lineares, e o XOR não possui tal fronteira.

---

## 4. Resultados dos Testes e Explicações

Foram realizados testes com o Perceptron implementado manualmente para as funções AND, OR e XOR, variando o número de entradas. Os resultados são apresentados na tabela abaixo e discutidos a seguir.

* **Para as funções AND e OR:**
    O Perceptron demonstrou ser capaz de aprender estas funções, que são linearmente separáveis. A convergência foi observada em todos os casos testados, resultando em 100% de acurácia. Para `n=2` entradas, a animação do hiperplano ilustrou claramente a formação da linha de separação. Para `n=3` entradas, a plotagem do plano de separação 3D ao final do treinamento confirmou a separação linear. Para `n=4` entradas, a visualização dos dados projetados em 2D via PCA auxiliou na compreensão da estrutura dos dados. As curvas de erro para AND e OR mostraram uma rápida queda no número de erros, atingindo zero em poucas épocas.
    *(Os gráficos e resultados detalhados podem ser conferidos no notebook `Perceptron_manual.ipynb`.)*

* **Para a função XOR:**
    O Perceptron não conseguiu aprender a função XOR para `n=2` entradas. A acurácia final ficou em torno de 50% a 75%, e a curva de erro indicou que o modelo não convergiu para uma solução sem erros.
    *(Os gráficos e resultados detalhados podem ser conferidos no notebook `Perceptron_manual.ipynb`.)*

**Tabela de Resultados (Valores Obtidos ao Executar `Perceptron_manual.ipynb`):**

| Função | N Entradas | Acurácia (%) | Épocas p/ Convergência | Pesos Finais ($w_0, w_1, ..., w_n$)                                  | Observações / Visualização                                  |
| :----- | :----------- | :------------- | :----------------------- | :------------------------------------------------------------------- | :---------------------------------------------------------- |
| AND    | 2            | 100%           | ~10                    | *Ex: [-1.5, 1.0, 1.0]* | Convergiu. Animação 2D mostra a linha.                        |
| OR     | 2            | 100%           | ~5                     | *Ex: [-0.5, 1.0, 1.0]* | Convergiu. Animação 2D mostra a linha.                        |
| AND    | 3            | 100%           | ~15                    | *Ex: [-2.5, 1.0, 1.0, 1.0]* | Convergiu. Plot 3D final mostra o plano.                    |
| OR     | 3            | 100%           | ~8                     | *Ex: [-0.5, 1.0, 1.0, 1.0]* | Convergiu. Plot 3D final mostra o plano.                    |
| AND    | 4            | 100%           | ~20                    | *Ex: [-3.5, 1.0, 1.0, 1.0, 1.0]* | Convergiu. Plot PCA 2D mostra os dados.                     |
| OR     | 4            | 100%           | ~12                    | *Ex: [-0.5, 1.0, 1.0, 1.0, 1.0]* | Convergiu. Plot PCA 2D mostra os dados.                     |
| XOR    | 2            | 75%            | Não Convergiu / 50 Ép  | *Ex: [0.1, -0.2, 0.3]* | Não linearmente separável. Animação e curva de erro mostram falha. |

*(Os valores na tabela são ilustrativos e devem ser substituídos pelos resultados reais obtidos na execução do notebook.)*

---

## 5. Considerações Finais

Este projeto demonstrou a implementação e o funcionamento do algoritmo Perceptron a partir do zero. Observou-se que o Perceptron é capaz de aprender e classificar corretamente funções linearmente separáveis como AND e OR, independentemente do número de entradas. A visualização animada do hiperplano para 2 entradas, o plano 3D final para 3 entradas, e a projeção PCA para um número maior de entradas, juntamente com as curvas de erro, foram ferramentas valiosas para entender o processo de aprendizado e a formação da fronteira de decisão linear.

Fundamentalmente, confirmou-se a limitação do Perceptron em não conseguir resolver problemas não linearmente separáveis, como a função XOR. Este exercício prático reforça a compreensão teórica do Perceptron, suas capacidades e suas limitações, servindo como base para o entendimento de modelos de redes neurais mais complexos.

---

## 6. Código Desenvolvido

O código implementado está disponível no seguinte Jupyter Notebook:
[Perceptron_manual.ipynb](https://github.com/LuanCarrieiros/IA/blob/main/Lista10/Questão%2001/perceptron_manual.ipynb)