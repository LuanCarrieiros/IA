import tkinter as tk
from tkinter import messagebox
import heapq
from collections import deque
import time
import random

# --------- Lógica ---------
class EightPuzzle:
    def __init__(self):
        self.goal = list(range(1, 9)) + [0]
        self.state = self.shuffle()

    def shuffle(self):
        state = self.goal[:]
        while True:
            random.shuffle(state)
            if self.is_solvable(state) and state != self.goal:
                return state

    def is_solvable(self, state):
        inv_count = 0
        for i in range(8):
            for j in range(i + 1, 9):
                if state[i] and state[j] and state[i] > state[j]:
                    inv_count += 1
        return inv_count % 2 == 0

    def move(self, direction):
        index = self.state.index(0)
        row, col = divmod(index, 3)
        if direction == "Cima" and row > 0:
            self.swap(index, index - 3)
        elif direction == "Baixo" and row < 2:
            self.swap(index, index + 3)
        elif direction == "Esquerda" and col > 0:
            self.swap(index, index - 1)
        elif direction == "Direita" and col < 2:
            self.swap(index, index + 1)

    def swap(self, i, j):
        self.state[i], self.state[j] = self.state[j], self.state[i]

    def is_solved(self):
        return self.state == self.goal

# --------- Funções de Busca ---------

def get_neighbors(state):
    neighbors = []
    index = state.index(0)
    row, col = divmod(index, 3)
    moves = [
        ("Cima", index - 3, row > 0),
        ("Baixo", index + 3, row < 2),
        ("Esquerda", index - 1, col > 0),
        ("Direita", index + 1, col < 2),
    ]
    for direction, new_index, valid in moves:
        if valid:
            new_state = list(state)
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            neighbors.append((new_state, direction))
    return neighbors


def manhattan(state):
    dist = 0
    for i, val in enumerate(state):
        if val == 0:
            continue
        goal_row, goal_col = divmod(val - 1, 3)
        row, col = divmod(i, 3)
        dist += abs(goal_row - row) + abs(goal_col - col)
    return dist


def heuristic_misplaced(state):
    goal = list(range(1,9)) + [0]
    return sum(1 for i, val in enumerate(state) if val != 0 and val != goal[i])


def bfs(start):
    goal = list(range(1, 9)) + [0]
    queue = deque([(start, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                queue.append((neighbor, path + [direction]))
    return []


def dfs_limited(start, limit=30):
    goal = list(range(1, 9)) + [0]
    visited = set()
    def recurse(state, path, depth):
        if state == goal:
            return path
        if depth >= limit:
            return None
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            t = tuple(neighbor)
            if t not in visited:
                res = recurse(neighbor, path + [direction], depth+1)
                if res is not None:
                    return res
        return None
    result = recurse(start, [], 0)
    return result if result is not None else []


def dfs(start):
    # Chama DFS limitado para evitar loops infinitos
    return dfs_limited(start, limit=30)


def astar_manhattan(start):
    goal = list(range(1, 9)) + [0]
    queue = [(manhattan(start), 0, start, [])]
    visited = set()
    while queue:
        _, cost, state, path = heapq.heappop(queue)
        if state == goal:
            return path
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                heapq.heappush(queue, (cost + 1 + manhattan(neighbor), cost + 1, neighbor, path + [direction]))
    return []


def astar_misplaced(start):
    goal = list(range(1, 9)) + [0]
    queue = [(heuristic_misplaced(start), 0, start, [])]
    visited = set()
    while queue:
        _, cost, state, path = heapq.heappop(queue)
        if state == goal:
            return path
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                heapq.heappush(queue, (cost + 1 + heuristic_misplaced(neighbor), cost + 1, neighbor, path + [direction]))
    return []

# --------- Interface ---------
class PuzzleApp:
    def __init__(self, root):
        self.game = EightPuzzle()
        self.root = root
        self.root.title("8-Puzzle - Solucionador Visual")
        self.root.configure(bg="#f6f6fa")

        self.grid_frame = tk.Frame(root, bg="#f6f6fa")
        self.grid_frame.pack(pady=20)

        self.tiles = []
        for i in range(9):
            btn = tk.Button(self.grid_frame, font=("Segoe UI", 20, "bold"), width=4, height=2,
                            bg="#dcd6f7", fg="#2b2d42", activebackground="#c3bef0",
                            command=lambda i=i: self.click_tile(i))
            btn.grid(row=i//3, column=i%3, padx=4, pady=4)
            self.tiles.append(btn)

        self.arrow_frame = tk.Frame(root, bg="#f6f6fa")
        self.arrow_frame.pack(pady=4)
        # Teclado direcional 3x3 para setas
        for r in range(3):
            for c in range(3):
                if r == 0 and c == 1:
                    text, cmd = "↑", lambda: self.move("Cima")
                elif r == 1 and c == 0:
                    text, cmd = "←", lambda: self.move("Esquerda")
                elif r == 2 and c == 1:
                    text, cmd = "↓", lambda: self.move("Baixo")
                elif r == 1 and c == 2:
                    text, cmd = "→", lambda: self.move("Direita")
                else:
                    text, cmd = "", None
                tk.Button(self.arrow_frame, text=text, command=cmd,
                          font=("Segoe UI", 12), width=5,
                          bg="#f0d9ff", fg="#2b2d42", activebackground="#e4c1f9")\
                    .grid(row=r, column=c, padx=4, pady=2)

        self.buttons_frame = tk.Frame(root, bg="#f6f6fa")
        self.buttons_frame.pack(pady=10)

        # Botões de ação
        self.make_btn("Embaralhar", self.shuffle, 0)
        self.make_btn("Resolver BFS", self.solve_bfs, 1)
        self.make_btn("Resolver DFS", self.solve_dfs, 2)
        self.make_btn("A* Manhattan", self.solve_astar_manhattan, 3)
        self.make_btn("A* Peças Fora", self.solve_astar_misplaced, 4)

        self.update_ui()

    def make_btn(self, text, command, col):
        tk.Button(self.buttons_frame, text=text, command=command,
                  font=("Segoe UI", 11), width=14,
                  bg="#b8c0ff", fg="#2b2d42", activebackground="#cdb4db")\
            .grid(row=0, column=col, padx=6)

    def update_ui(self):
        for i in range(9):
            val = self.game.state[i]
            self.tiles[i].config(text=str(val) if val != 0 else "", 
                                 bg="#f6f6fa" if val == 0 else "#dcd6f7")

    def click_tile(self, index):
        zero = self.game.state.index(0)
        if abs(zero - index) in [1, 3]:
            self.game.swap(zero, index)
            self.update_ui()

    def move(self, direction):
        self.game.move(direction)
        self.update_ui()

    def shuffle(self):
        self.game.state = self.game.shuffle()
        self.update_ui()

    def animate(self, moves, tempo):
        if not moves:
            messagebox.showinfo("Resultado", "Nenhuma solução encontrada.")
            return

        total = len(moves)

        def step():
            nonlocal total
            if moves:
                self.game.move(moves.pop(0))
                self.update_ui()
                self.root.after(350, step)
            else:
                messagebox.showinfo("Concluído", f"Solução em {total} movimentos.\nTempo: {tempo:.4f}s")

        step()

    def solve_bfs(self):
        inicio = time.time()
        moves = bfs(self.game.state[:])
        fim = time.time()
        self.animate(moves, fim - inicio)

    def solve_dfs(self):
        inicio = time.time()
        moves = dfs(self.game.state[:])
        fim = time.time()
        self.animate(moves, fim - inicio)

    def solve_astar_manhattan(self):
        inicio = time.time()
        moves = astar_manhattan(self.game.state[:])
        fim = time.time()
        self.animate(moves, fim - inicio)

    def solve_astar_misplaced(self):
        inicio = time.time()
        moves = astar_misplaced(self.game.state[:])
        fim = time.time()
        self.animate(moves, fim - inicio)

if __name__ == '__main__':
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
