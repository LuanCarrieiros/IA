# Questão 01: Implementação do Algoritmo Perceptron

## Introdução

Este projeto foca na implementação do algoritmo Perceptron, um modelo fundamental de rede neural artificial, para resolver funções lógicas com **n entradas booleanas**. Especificamente, o Perceptron será treinado para aprender as funções **AND** e **OR**. O usuário terá a flexibilidade de definir o número de entradas (por exemplo, 2, 5 ou 10) para estas funções.

Uma parte crucial deste trabalho é a **visualização das regras de separação** dos hiperplanos que o Perceptron aprende durante o seu processo de treinamento. Adicionalmente, demonstraremos uma limitação inerente ao Perceptron: sua incapacidade de resolver a função **XOR**, que não é linearmente separável.

Este README detalhará a teoria do Perceptron, a abordagem de implementação, a lógica para geração de dados e plotagem, e a análise dos resultados obtidos. Ao final, o código desenvolvido deverá ser disponibilizado.

---

## 1. O Algoritmo Perceptron

O Perceptron é um classificador linear binário. Ele calcula uma soma ponderada das entradas, adiciona um bias, e então aplica uma função de ativação (geralmente uma função degrau) para produzir uma saída.

* **Entradas (Input):** $x_1, x_2, ..., x_n$ (valores booleanos 0 ou 1)
* **Pesos (Weights):** $w_1, w_2, ..., w_n$ (representam a importância de cada entrada)
* **Bias (b):** Permite ajustar a decisão do neurônio independentemente das entradas.
* **Soma Ponderada (Net Input):** $z = (\sum_{i=1}^{n} w_i x_i) + b$
* **Função de Ativação (Step Function):**
    * $y = 1$ se $z \ge \theta$ (limiar)
    * $y = 0$ se $z < \theta$
    (Frequentemente, o limiar é incorporado ao bias, e a ativação é $y=1$ se $z \ge 0$, e $y=0$ caso contrário).

### Processo de Aprendizagem

O Perceptron aprende de forma iterativa. Para cada exemplo de treinamento:
1.  Calcula-se a saída.
2.  O erro é determinado: $erro = \text{saida\_desejada} - \text{saida\_calculada}$.
3.  Os pesos e o bias são ajustados para minimizar o erro:
    * $w_i(\text{novo}) = w_i(\text{antigo}) + \eta \cdot erro \cdot x_i$
    * $b(\text{novo}) = b(\text{antigo}) + \eta \cdot erro$
    Onde $\eta$ é a **taxa de aprendizado**.

Este processo é repetido por várias épocas até que o modelo convirja.

---

## 2. Estrutura da Implementação

O código está estruturado com as seguintes funções principais:

### 2.1. Função `gerar_dados(n_entradas, tipo)`

* **Objetivo:** Gera todas as combinações possíveis de entradas booleanas para `n_entradas` e os respectivos vetores de saída `y` de acordo com a função lógica escolhida.
* **Lógica:**
    * Cria a matriz `X` de entradas usando `itertools.product` para gerar todas as $2^n$ combinações de `0` e `1`.
    * Calcula o vetor `y` de saídas:
        * **AND**: saída `1` se todas as entradas forem `1` (`np.all(x)`).
        * **OR**: saída `1` se pelo menos uma entrada for `1` (`np.any(x)`).
        * **XOR**: saída `1` se o número de entradas `1` for ímpar (`np.sum(x) % 2`).
* **Retorno:** Matrizes `X` (entradas) e `y` (saídas).

### 2.2. Função `treinar_perceptron(X, y, titulo)`

* **Objetivo:** Treina um modelo Perceptron utilizando a implementação da biblioteca `scikit-learn`.
* **Lógica:**
    * Inicializa um classificador `Perceptron` (da `sklearn.linear_model`).
    * Treina o modelo usando `clf.fit(X, y)`.
    * Realiza predições com `clf.predict(X)`.
    * Exibe informações como os pesos aprendidos (`clf.coef_`), o bias (`clf.intercept_`), as saídas esperadas e previstas, e a acurácia do modelo (`clf.score(X, y)`).
    * Chama a função `plotar_hiperplano` para visualização.

### 2.3. Função `plotar_hiperplano(X, y, clf, titulo)`

* **Objetivo:** Visualizar a fronteira de decisão (hiperplano) aprendida pelo Perceptron.
* **Lógica:**
    * **Para 2 entradas (n=2):** Plota os pontos de dados e o hiperplano como uma linha reta que separa as classes. A equação da linha é derivada dos pesos e do bias do classificador.
    * **Para 3 entradas (n=3):** Plota os pontos de dados em um gráfico 3D e o hiperplano como um plano.
    * **Para n > 3 entradas:** Aplica Análise de Componentes Principais (PCA) para reduzir a dimensionalidade dos dados para 2D. Em seguida, plota a projeção dos dados em 2D, permitindo uma visualização da separação, embora o hiperplano em si não seja diretamente plotado neste caso (a fronteira em 2D no espaço PCA pode ser mostrada se o classificador for treinado nos dados reduzidos).
* **Observação:** A plotagem do hiperplano é realizada após o treinamento completo do modelo `sklearn`.

---

## 3. Demonstração: Perceptron e XOR

A função XOR (OU Exclusivo) é o exemplo clássico de um problema não linearmente separável.
Para 2 entradas:

| x1 | x2 | y (XOR) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

* **Procedimento:** O Perceptron implementado (usando `sklearn`) é treinado com os dados da função XOR.
* **Observação Esperada:** O algoritmo Perceptron não conseguirá convergir para uma solução que classifique corretamente todos os pontos. A acurácia do modelo não atingirá 100% (geralmente fica em torno de 50% ou 75% para 2 entradas, dependendo da solução que o `sklearn` encontra).
* **Visualização (para n=2):** Ao plotar os pontos do XOR, fica evidente que nenhuma reta única consegue separar as classes `{(0,0), (1,1)}` da classe `{(0,1), (1,0)}`.

Isso ocorre porque o Perceptron só pode encontrar fronteiras de decisão lineares.

---

## 4. Resultados dos Testes e Explicações

*(Nesta seção, você deverá detalhar os resultados dos testes realizados para as funções AND e OR com diferentes números de entradas, e para a função XOR. Inclua os pesos finais, o bias, o número de épocas (se aplicável/observável com sklearn, ou a acurácia), e as plotagens geradas.)*

* **Para AND e OR:**
    * O Perceptron do `sklearn` deve convergir e aprender as funções perfeitamente, pois são linearmente separáveis.
    * A acurácia esperada é de 100%.
    * Documente os pesos e bias para diferentes valores de `n` (ex: 2, 3, 5, 10).
    * Inclua as visualizações do hiperplano para `n=2` e `n=3`, e a projeção PCA para `n > 3`.
* **Para XOR:**
    * O Perceptron não deve atingir 100% de acurácia.
    * Documente a acurácia obtida e mostre a plotagem para `n=2` que evidencia a não separabilidade linear.

**Exemplo de Tabela de Resultados (a ser preenchida com seus dados):**

| Função | N Entradas | Acurácia (%) | Pesos Finais (`clf.coef_`) | Bias Final (`clf.intercept_`) | Observações                                   |
|--------|------------|--------------|----------------------------|------------------------------|-----------------------------------------------|
| AND    | 2          | 100%         | (seus valores)             | (seu valor)                  | Convergiu.                                    |
| AND    | 3          | 100%         | (seus valores)             | (seu valor)                  | Convergiu.                                    |
| AND    | 10         | 100%         | (seus valores)             | (seu valor)                  | Convergiu (PCA para visualização).            |
| OR     | 2          | 100%         | (seus valores)             | (seu valor)                  | Convergiu.                                    |
| OR     | 3          | 100%         | (seus valores)             | (seu valor)                  | Convergiu.                                    |
| XOR    | 2          | (e.g. 50%)   | (seus valores)             | (seu valor)                  | Não linearmente separável, Perceptron falhou. |

---

## 5. Considerações Finais

*(Resuma as principais aprendizagens do desenvolvimento da Questão 01. Destaque a capacidade do Perceptron de resolver problemas linearmente separáveis, sua limitação com o XOR, e como a visualização ajuda a entender o modelo.)*

---

## 6. Código Desenvolvido

*(Disponibilize o seu arquivo .py ou o conteúdo do seu notebook aqui, ou um link para ele.)*