# Análise de Dados do Titanic com Pipeline de IA

Este repositório contém um projeto completo de Ciência de Dados que analisa o famoso dataset do Titanic. O objetivo é aplicar um pipeline de Inteligência Artificial para não apenas prever a sobrevivência dos passageiros, mas também para descobrir perfis e padrões ocultos através de diferentes técnicas de modelagem.

Este projeto foi desenvolvido como parte da avaliação da disciplina de Inteligência Artificial do curso de Ciência da Computação.

## 🚀 Sobre o Projeto

Utilizando a base de dados do Titanic, foram aplicadas três abordagens distintas e complementares de IA:

1.  **Aprendizagem Supervisionada:** Para treinar modelos de classificação capazes de prever a sobrevivência.
2.  **Aprendizagem Não Supervisionada:** Para segmentar os passageiros em grupos com características semelhantes (clusters).
3.  **Regras de Associação:** Para extrair regras de causa e efeito que descrevem relações fortes entre os atributos dos passageiros.

O projeto está contido no notebook `Lista_11_Titanic.ipynb`.

## 🛠️ Tecnologias e Bibliotecas

* **Linguagem:** Python 3
* **Bibliotecas Principais:**
    * `pandas` para manipulação e análise de dados.
    * `numpy` para operações numéricas.
    * `scikit-learn` para os modelos de machine learning (classificação e clusterização).
    * `matplotlib` e `seaborn` para visualização de dados.
    * `mlxtend` para a mineração de regras de associação (Apriori).

## 📈 Pipeline de Análise

O projeto seguiu um pipeline bem definido, dividido em quatro etapas principais.

### 1. Pré-processamento e Engenharia de Variáveis

A preparação dos dados foi um passo crucial para garantir a qualidade dos modelos.

* **Tratamento de Dados Ausentes:**
    * As colunas `Age` e `Fare` tiveram seus valores nulos preenchidos com a **mediana**.
    * A coluna `Embarked` foi preenchida com a **moda** ('S').
    * A coluna `Cabin` foi transformada em uma feature binária `Has_Cabin`.

* **Engenharia de Variáveis:** Para enriquecer o dataset, foram criadas novas features:
    ```python
    # Exemplo de criação de novas features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    ```

### 2. Modelagem Supervisionada (Classificação)

Dois modelos foram treinados para prever a sobrevivência: `DecisionTreeClassifier` e `RandomForestClassifier`.

* **Resultados:** Ao serem avaliados nos próprios dados de treino, ambos os modelos apresentaram uma acurácia de **~98.4%**.

| Modelo                | Acurácia (Treino) | Precision (Média) | Recall (Média) | F1-Score (Média) |
| :-------------------- | :---------------- | :---------------- | :------------- | :--------------- |
| Árvore de Decisão     | 0.9843            | 0.99              | 0.98           | 0.98             |
| Random Forest         | 0.9843            | 0.99              | 0.98           | 0.98             |

**Observação:** A alta acurácia é um sinal de **overfitting**. A avaliação em dados de treino não é uma medida fiel do poder de generalização do modelo, mas serve como uma primeira análise de seu potencial.

### 3. Modelagem Não Supervisionada (Clusterização)

Utilizamos o algoritmo K-Means para encontrar perfis de passageiros.

* **Escolha do K:** O "Método do Cotovelo" indicou que **K=3** era o número ideal de clusters.

    *Insira aqui a imagem do gráfico do Método do Cotovelo (`image_caf905.png`) se desejar.*

* **Visualização e Perfis:** A visualização com PCA mostrou três grupos distintos, que foram interpretados como:

    *Insira aqui a imagem do gráfico dos Clusters com PCA (`image_caf923.png`) se desejar.*

| Cluster | Perfil                          | Taxa de Sobrevivência | Características Principais                      |
| :------ | :------------------------------ | :-------------------- | :---------------------------------------------- |
| **0** | **A Elite Rica** | **66%** | 1ª Classe, Tarifa Alta, Adultos                 |
| **2** | **Famílias (Mulheres/Crianças)** | **50%** | Famílias grandes, Idade baixa, 2ª/3ª Classe |
| **1** | **Homens Pobres e Sozinhos** | **22%** | 3ª Classe, Tarifa Baixa, Viajando sozinhos      |

### 4. Extração de Regras de Associação

Com o algoritmo Apriori, buscamos padrões "SE-ENTÃO". As regras mais fortes encontradas estavam relacionadas ao status socioeconômico.

> **Regra Destaque:**
> * **SE** o passageiro é da `{Mulher, 1ª Classe, Com Família}`
> * **ENTÃO** há **90% de chance** de ele ter pago uma `{Tarifa Alta}`.
> * Esta associação é **7.4 vezes mais forte** do que o acaso (Lift = 7.4).

## 📄 Conclusão

A combinação das três abordagens de IA forneceu uma análise 360° do dataset do Titanic:
* A **Classificação** nos deu o poder de **prever** um resultado.
* A **Clusterização** nos permitiu **descrever** e entender as personas a bordo.
* As **Regras de Associação** nos ajudaram a **confirmar** padrões com rigor estatístico.

Juntas, elas transformaram os dados brutos em uma história coesa sobre a dinâmica social e as chances de sobrevivência na trágica viagem do Titanic.

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    ```
2.  **Instale as dependências:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn mlxtend
    ```
3.  **Execute o Notebook:**
    Abra o arquivo `Lista_11_Titanic.ipynb` em um ambiente como Jupyter Lab, Jupyter Notebook ou VS Code e execute as células na ordem.