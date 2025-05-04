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

# --------- Algoritmos ---------
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

def dfs(start):
    goal = list(range(1, 9)) + [0]
    stack = [(start, [])]
    visited = set()
    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                stack.append((neighbor, path + [direction]))
    return []

def astar(start):
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
        self.seta("↑", lambda: self.move("Cima"), 1, 0)
        self.seta("←", lambda: self.move("Esquerda"), 2, 0)
        self.seta("↓", lambda: self.move("Baixo"), 2, 1)
        self.seta("→", lambda: self.move("Direita"), 2, 2)

        self.buttons_frame = tk.Frame(root, bg="#f6f6fa")
        self.buttons_frame.pack(pady=10)

        self.make_btn("Embaralhar", self.shuffle, 0)
        self.make_btn("Resolver BFS", self.solve_bfs, 1)
        self.make_btn("Resolver DFS", self.solve_dfs, 2)
        self.make_btn("Resolver A*", self.solve_astar, 3)

        self.update_ui()

    def seta(self, texto, comando, linha, coluna):
        tk.Button(self.arrow_frame, text=texto, command=comando,
                  font=("Segoe UI", 12), width=5,
                  bg="#f0d9ff", fg="#2b2d42", activebackground="#e4c1f9")\
            .grid(row=linha, column=coluna, padx=4)

    def make_btn(self, text, command, col):
        tk.Button(self.buttons_frame, text=text, command=command,
                  font=("Segoe UI", 11), width=14,
                  bg="#b8c0ff", fg="#2b2d42", activebackground="#cdb4db")\
            .grid(row=0, column=col, padx=8)

    def update_ui(self):
        for i in range(9):
            val = self.game.state[i]
            self.tiles[i].config(text=str(val) if val != 0 else "", bg="#f6f6fa" if val == 0 else "#dcd6f7")

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
            if moves:
                self.game.move(moves.pop(0))
                self.update_ui()
                self.root.after(350, step)
            else:
                messagebox.showinfo("Concluído", f"Solução encontrada com {total} movimentos.\nTempo: {tempo:.4f}s")

        step()

    def solve_bfs(self):
        start = time.time()
        moves = bfs(self.game.state[:])
        end = time.time()
        self.animate(moves[:], end - start)

    def solve_dfs(self):
        start = time.time()
        moves = dfs(self.game.state[:])
        end = time.time()
        self.animate(moves[:], end - start)

    def solve_astar(self):
        start = time.time()
        moves = astar(self.game.state[:])
        end = time.time()
        self.animate(moves[:], end - start)

if __name__ == '__main__':
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
