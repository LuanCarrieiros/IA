# IA

Inteligência Artificial

# Resolução do 8-Puzzle com Métodos de Busca (BFS, DFS, A*)

Este projeto implementa a resolução do clássico problema do 8-Puzzle utilizando três algoritmos de busca fundamentais em Inteligência Artificial:

- **Busca em Largura (Breadth-First Search - BFS)**
- **Busca em Profundidade (Depth-First Search - DFS)**
- **Busca A\*** com duas heurísticas diferentes:
  - Distância de Manhattan
  - Número de peças fora do lugar

## 🔧 Estrutura do Código

O código está organizado em um único arquivo `puzzle.py`, contendo:
- Definição do estado inicial e objetivo
- Funções auxiliares para manipulação do tabuleiro
- Implementações completas dos algoritmos de busca
- Execução principal com coleta de desempenho (tempo, movimentos e nós visitados)

## 📊 Comparação de Desempenho

Durante a execução, o programa imprime os resultados obtidos por cada método, permitindo avaliar:

- Qual algoritmo encontrou a solução mais rapidamente
- Quantos nós foram visitados por cada busca
- Quantos movimentos foram necessários até a solução

## 📋 Requisitos

- Python 3.x

## 🚀 Executando

Basta rodar o comando:

```bash
python puzzle.py

