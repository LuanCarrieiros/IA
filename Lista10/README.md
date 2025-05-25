# Questão 01: Implementação do Algoritmo Perceptron

## Introdução

Este projeto foca na implementação do algoritmo Perceptron, um modelo fundamental de rede neural artificial, para resolver funções lógicas com **n entradas booleanas**. Especificamente, o Perceptron será treinado para aprender as funções **AND** e **OR**. [cite: 1] O usuário terá a flexibilidade de definir o número de entradas (por exemplo, 2, 5 ou 10) para estas funções. [cite: 2]

Uma parte crucial deste trabalho é a **visualização das regras de separação** dos hiperplanos que o Perceptron aprende durante o seu processo de treinamento. [cite: 4] Adicionalmente, demonstraremos uma limitação inerente ao Perceptron: sua incapacidade de resolver a função **XOR**, que não é linearmente separável. [cite: 5]

Este README detalhará a teoria do Perceptron, a abordagem de implementação, a lógica para geração de dados e plotagem, e a análise dos resultados obtidos. Ao final, o código desenvolvido deverá ser disponibilizado. [cite: 3, 4]

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
2.  O erro é determinado: $erro = \text{saida_desejada} - \text{saida_calculada}$.
3.  Os pesos e o bias são ajustados para minimizar o erro:
    * $w_i(\text{novo}) = w_i(\text{antigo}) + \eta \cdot erro \cdot x_i$
    * $b(\text{novo}) = b(\text{antigo}) + \eta \cdot erro$
    Onde $\eta$ é a **taxa de aprendizado**.

Este processo é repetido por várias épocas até que o modelo convirja.

---

## 2. Estrutura da Implementação (Sugestão)

Para organizar o código, sugere-se a seguinte estrutura:

### 2.1. Geração de Dados (`gerar_dados`)

* **Objetivo:** Criar o conjunto de dados (entradas `X` e saídas `y`) para as funções AND e OR com `n_entradas`.
* **Lógica:**
    * Gerar todas as $2^n$ combinações possíveis de entradas booleanas (0 ou 1) para `n_entradas`. A biblioteca `itertools` do Python (`product`) pode ser útil aqui.
    * Para cada combinação de entrada, calcular a saída `y` correspondente:
        * **AND:** `y = 1` se todas as entradas forem 1; caso contrário, `y = 0`.
        * **OR:** `y = 1` se pelo menos uma entrada for 1; caso contrário, `y = 0`.
* **Retorno:** Matriz de entradas `X` e vetor de saídas `y`.

### 2.2. Classe/Funções do Perceptron

* **Inicialização:** Definir o número de entradas, taxa de aprendizado e número máximo de épocas. Inicializar pesos (aleatoriamente ou com zeros) e o bias.
* **Função de Ativação:** Implementar a função degrau.
* **Predição (`predict`):** Calcular a saída para um conjunto de entradas usando os pesos atuais.
* **Treinamento (`train`):**
    * Iterar sobre os dados de treinamento por um número de épocas ou até a convergência.
    * Para cada amostra, fazer uma predição, calcular o erro e atualizar os pesos e o bias.
    * Registrar o histórico de erros ou mudanças nos pesos para visualização.

### 2.3. Plotagem do Hiperplano (`plotar_hiperplano`) [cite: 4]

* **Objetivo:** Visualizar a fronteira de decisão aprendida pelo Perceptron.
* A equação do hiperplano é: $w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = 0$.
* **Para 2 Entradas (n=2):** A fronteira é uma linha. Pode ser plotada como $x_2 = (-\frac{w_1}{w_2}) x_1 - (\frac{b}{w_2})$.
    * Plote os pontos de dados coloridos por classe e a linha de separação.
    * Idealmente, mostre a evolução da linha durante o treinamento.
* **Para 3 Entradas (n=3):** A fronteira é um plano. Utilize bibliotecas de plotagem 3D (como Matplotlib) para visualizar o plano e os pontos de dados.
* **Para n > 3 Entradas:** A visualização direta é complexa.
    * Uma abordagem é usar técnicas de redução de dimensionalidade (como PCA) para projetar os dados e o hiperplano em 2D ou 3D, se viável.
    * Alternativamente, foque em plotar a convergência do erro ao longo das épocas ou mostrar os pesos finais.
    * Conforme o enunciado, "Você deverá plotar as regras de separação dos hiperplanos durante o processo de treinamento." [cite: 4] Se o número de entradas for alto, é importante documentar como essa visualização será abordada.

---

## 3. Demonstração: Perceptron e XOR [cite: 5]

A função XOR (OU Exclusivo) é o exemplo clássico de um problema não linearmente separável.
Para 2 entradas:

| x1 | x2 | y (XOR) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

* **Procedimento:** Treine seu Perceptron implementado com os dados da função XOR.
* **Observação Esperada:** O algoritmo Perceptron não conseguirá convergir para uma solução que classifique corretamente todos os pontos. O erro não se estabilizará em zero (ou um valor mínimo aceitável), ou o modelo alternará entre soluções subótimas.
* **Visualização (para n=2):** Ao plotar os pontos do XOR, fica evidente que nenhuma reta única consegue separar as classes {(0,0), (1,1)} da classe {(0,1), (1,0)}.

Isso ocorre porque o Perceptron só pode encontrar fronteiras de decisão lineares.

---

## 4. Resultados Esperados e Testes

A sua lista deverá conter todas as explicações da implementação e os resultados dos testes realizados. [cite: 3]

* **Para AND e OR:**
    * O Perceptron deve convergir e aprender as funções perfeitamente para qualquer número de entradas `n`, pois são linearmente separáveis.
    * Documente os pesos finais, o bias e o número de épocas necessárias para a convergência para diferentes valores de `n` (ex: 2, 3, 5, 10). [cite: 2]
    * A acurácia do modelo nos dados de treinamento e teste (se aplicável) deve ser de 100%.
* **Para XOR:**
    * O Perceptron não deve convergir para uma solução perfeita.
    * Documente o comportamento do erro durante o treinamento. A acurácia provavelmente ficará em torno de 50% ou 75% (dependendo da implementação e dos dados de parada) para 2 entradas.

**Exemplo de Tabela de Resultados (a ser preenchida com seus dados):**

| Função | N Entradas | Acurácia (%) | Épocas p/ Convergência | Pesos Finais (Exemplo) | Bias Final (Exemplo) | Observações                                   |
|--------|------------|--------------|------------------------|------------------------|----------------------|-----------------------------------------------|
| AND    | 2          | 100%         | (seu valor)            | (seus valores)         | (seu valor)          | Convergiu.                                    |
| AND    | 5          | 100%         | (seu valor)            | (seus valores)         | (seu valor)          | Convergiu.                                    |
| OR     | 2          | 100%         | (seu valor)            | (seus valores)         | (seu valor)          | Convergiu.                                    |
| OR     | 5          | 100%         | (seu valor)            | (seus valores)         | (seu valor)          | Convergiu.                                    |
| XOR    | 2          | (ex: 50-75%) | Não convergiu / Max    | (seus valores)         | (seu valor)          | Não linearmente separável, Perceptron falhou. |

---

## 5. Considerações Finais

Resuma as principais aprendizagens do desenvolvimento da Questão 01. Destaque:
* A capacidade do Perceptron de resolver problemas linearmente separáveis.
* A importância da taxa de aprendizado e da inicialização de pesos no processo de convergência.
* A limitação fundamental do Perceptron ao lidar com dados não linearmente separáveis, como o XOR.
* As complexidades e abordagens para visualização de hiperplanos em diferentes dimensões.

---

## 6. Código Desenvolvido

Ao final, disponibilize o código desenvolvido. [cite: 4]
*(Aqui você pode colocar um link para seus arquivos de código ou incorporar snippets importantes, conforme o exemplo do seu amigo.)*