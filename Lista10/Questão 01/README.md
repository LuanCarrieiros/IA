# Questão 01: Implementação do Algoritmo Perceptron Manual

## Introdução

Este projeto foca na implementação do algoritmo Perceptron **do zero**, um modelo fundamental de rede neural artificial, para resolver funções lógicas com **n entradas booleanas**. Especificamente, o Perceptron será treinado para aprender as funções **AND** e **OR**. O usuário terá a flexibilidade de definir o número de entradas (por exemplo, 2, 3, 5 ou 10) para estas funções.

Uma parte crucial deste trabalho é a **visualização das regras de separação** dos hiperplanos. Para `n=2` entradas, o hiperplano (uma linha) é plotado dinamicamente **durante o processo de treinamento**, mostrando sua evolução. Para `n=3` entradas, o plano de separação final é visualizado em 3D. Para `n > 3` entradas, os dados são projetados em 2D usando Análise de Componentes Principais (PCA) para visualização. Adicionalmente, a curva de erro por época é plotada para todos os casos.

Demonstraremos também uma limitação inerente ao Perceptron: sua incapacidade de resolver a função **XOR**, que não é linearmente separável, através da animação do hiperplano e da curva de erro.

Este README detalhará a teoria do Perceptron, a abordagem da implementação manual, a lógica para geração de dados e as diferentes formas de plotagem utilizadas, além da análise dos resultados obtidos com o código desenvolvido.

---

## 1. O Algoritmo Perceptron

O Perceptron é um classificador linear binário. Ele calcula uma soma ponderada das entradas, adiciona um bias, e então aplica uma função de ativação (geralmente uma função degrau) para produzir uma saída.

* **Entradas (Input):** $x_1, x_2, ..., x_n$ (valores booleanos 0 ou 1)
* **Pesos (Weights):** $w_0 (bias), w_1, w_2, ..., w_n$ (representam a importância de cada entrada e o limiar)
* **Soma Ponderada (Net Input):** $z = w_0 \cdot 1 + \sum_{i=1}^{n} w_i x_i$ (o bias $w_0$ é associado a uma entrada $x_0=1$)
* **Função de Ativação (Step Function):**
    * $y = 1$ se $z \ge 0$
    * $y = 0$ se $z < 0$

### Processo de Aprendizagem

O Perceptron aprende de forma iterativa, ajustando seus pesos com base nos erros cometidos. Para cada amostra de treinamento $(X_i, y_i)$:
1.  Calcula-se a saída predita $\hat{y}_i$.
2.  O erro é determinado: $erro_i = y_i - \hat{y}_i$.
3.  Se $erro_i \neq 0$, os pesos são ajustados:
    * $w_j(\text{novo}) = w_j(\text{antigo}) + \eta \cdot erro_i \cdot x_{ij}$ (onde $x_{i0}=1$)
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
* **`fit(self, X, y, ax_plot, ...)`**: Executa o treinamento do Perceptron, atualizando os pesos e, opcionalmente, chamando a função de plotagem 2D animada.

### 2.2. Funções Auxiliares
* **`gerar_dados(n_entradas, tipo_funcao)`**:
    * Gera todas as $2^n$ combinações de entradas booleanas para `n_entradas`.
    * Calcula as saídas `y` para as funções `AND`, `OR` ou `XOR`.
* **`plotar_hiperplano_2d(X, y, perceptron_model, ax, titulo_plot)`**:
    * Chamada durante o `fit` para `n_entradas = 2`.
    * Limpa e redesenha o gráfico a cada intervalo de épocas, mostrando os pontos de dados e a linha de separação ($w_1x_1 + w_2x_2 + w_0 = 0$) atual, criando uma animação da aprendizagem.
* **`plotar_curva_erro(erros_por_epoca, titulo_plot)`**:
    * Plotada ao final de cada treinamento, mostra o número de erros de classificação por época.
* **`plotar_hiperplano_3d_final(X, y, perceptron_model, titulo_plot)`**:
    * Chamada após o treinamento para `n_entradas = 3` (se houver convergência).
    * Mostra os pontos de dados em 3D e o plano de separação final ($w_1x_1 + w_2x_2 + w_3x_3 + w_0 = 0$).
* **`plotar_pca_2d_final(X, y, titulo_plot)`**:
    * Chamada após o treinamento para `n_entradas > 3`.
    * Aplica PCA para reduzir os dados a 2 dimensões e plota os pontos projetados, coloridos por classe, para inspeção visual da separabilidade.
* **`executar_teste_perceptron(tipo_funcao, num_entradas, ...)`**:
    * Função principal que encapsula um ciclo de teste: geração de dados, instanciação e treinamento do Perceptron, e chamada das funções de plotagem apropriadas.

---

## 3. Demonstração: Perceptron e XOR

A função XOR (OU Exclusivo) é um exemplo clássico de um problema não linearmente separável.
Para 2 entradas:

| x1 | x2 | y (XOR) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

* **Procedimento**: O Perceptron implementado manualmente é treinado com os dados da função XOR.
* **Observação Esperada**: O algoritmo Perceptron não conseguirá convergir. A animação do hiperplano 2D mostrará a linha tentando, sem sucesso, separar as classes. A curva de erro não atingirá zero consistentemente.
* **Visualização**:
    * Para `n=2`, a animação do hiperplano e a plotagem dos pontos do XOR evidenciam que nenhuma reta única consegue separar as classes `{(0,0), (1,1)}` da classe `{(0,1), (1,0)}`.
    * A curva de erro mostrará a persistência de erros ao longo das épocas.

Isso ocorre porque o Perceptron só pode encontrar fronteiras de decisão lineares.

---

## 4. Resultados dos Testes e Explicações

*(Nesta seção, você deverá detalhar os resultados dos testes realizados com o `Perceptron_manual.ipynb` para as funções AND e OR com diferentes números de entradas, e para a função XOR. Inclua os pesos finais (bias $w_0$ e $w_i$), o número de épocas para convergência, a acurácia final, e as plotagens geradas.)*

* **Para AND e OR**:
    * O Perceptron manual deve convergir e aprender as funções perfeitamente, pois são linearmente separáveis.
    * A acurácia esperada é de 100%.
    * Documente os pesos ($w_0, w_1, ..., w_n$) e o número de épocas para convergência para diferentes valores de `n` (ex: 2, 3, 4).
    * Inclua exemplos das visualizações:
        * Animação do hiperplano para `n=2`.
        * Plano 3D final para `n=3`.
        * Projeção PCA para `n > 3`.
        * Curva de erro para todos os casos.
* **Para XOR**:
    * O Perceptron não deve atingir 100% de acurácia.
    * Documente a acurácia obtida, mostre a animação para `n=2` e a curva de erro que evidencia a não convergência.

**Exemplo de Tabela de Resultados (a ser preenchida com seus dados do `Perceptron_manual.ipynb`):**

| Função | N Entradas | Acurácia (%) | Épocas p/ Convergência | Pesos Finais ($w_0, w_1, ..., w_n$) | Observações                                   |
|--------|------------|--------------|------------------------|---------------------------------|-----------------------------------------------|
| AND    | 2          | 100%         | (seu valor)            | (seus valores)                  | Convergiu. Animação 2D mostra a linha.        |
| OR     | 2          | 100%         | (seu valor)            | (seus valores)                  | Convergiu. Animação 2D mostra a linha.        |
| AND    | 3          | 100%         | (seu valor)            | (seus valores)                  | Convergiu. Plot 3D final mostra o plano.      |
| OR     | 4          | 100%         | (seu valor)            | (seus valores)                  | Convergiu. Plot PCA 2D mostra os dados.       |
| XOR    | 2          | (e.g. ~50-75%)| Não Convergiu / Max Épocas | (seus valores)              | Não linearmente separável. Animação e curva de erro mostram falha. |

*(Preencha a tabela acima com resultados reais obtidos ao executar o seu notebook `Perceptron_manual.ipynb`)*

---

## 5. Considerações Finais

Este projeto demonstrou a implementação e o funcionamento do algoritmo Perceptron a partir do zero.
Observou-se que:
* O Perceptron é capaz de aprender e classificar corretamente funções linearmente separáveis como AND e OR, independentemente do número de entradas.
* A visualização animada do hiperplano para 2 entradas oferece uma compreensão intuitiva do processo de aprendizado e da formação da fronteira de decisão.
* Para 3 entradas, a visualização do plano de separação final em 3D confirma a capacidade de separação linear.
* Para um número maior de entradas, a Análise de Componentes Principais (PCA) permite uma visualização qualitativa da estrutura dos dados em um espaço de dimensão reduzida.
* A curva de erro por época é uma ferramenta valiosa para monitorar a convergência do algoritmo em todos os cenários.
* O Perceptron fundamentalmente não consegue resolver problemas não linearmente separáveis, como a função XOR. Isso foi claramente demonstrado pela incapacidade do modelo de convergir para uma solução sem erros e pela visualização do hiperplano.

Este exercício prático reforça a compreensão teórica do Perceptron, suas capacidades e suas limitações, servindo como base para o entendimento de modelos de redes neurais mais complexos.

---

## 6. Código Desenvolvido

O código implementado está disponível no seguinte Jupyter Notebook:
[Perceptron_manual.ipynb](https://github.com/LuanCarrieiros/IA/blob/main/Lista10/Questão%2001/perceptron_manual.ipynb)
