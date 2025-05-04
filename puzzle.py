
import heapq
import time
from collections import deque

GOAL_STATE = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
GOAL_FLAT = sum(GOAL_STATE, [])

MOVES = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

def flatten(state):
    return sum(state, [])

def state_to_tuple(state):
    return tuple(flatten(state))

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def is_goal(state):
    return flatten(state) == GOAL_FLAT

def neighbors(state):
    x, y = find_zero(state)
    result = []
    for move, (dx, dy) in MOVES.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            result.append(new_state)
    return result

def heuristic_manhattan(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val == 0:
                continue
            goal_i, goal_j = divmod(GOAL_FLAT.index(val), 3)
            dist += abs(goal_i - i) + abs(goal_j - j)
    return dist

def heuristic_misplaced(state):
    return sum(1 for i in range(9) if flatten(state)[i] != 0 and flatten(state)[i] != GOAL_FLAT[i])

def bfs(start):
    start_time = time.time()
    visited = set()
    queue = deque([(start, [])])
    while queue:
        current, path = queue.popleft()
        if is_goal(current):
            return path + [current], len(visited), time.time() - start_time
        key = state_to_tuple(current)
        if key not in visited:
            visited.add(key)
            for neighbor in neighbors(current):
                queue.append((neighbor, path + [current]))
    return None, len(visited), time.time() - start_time

def dfs(start, limit=20):
    start_time = time.time()
    visited = set()
    stack = [(start, [])]
    while stack:
        current, path = stack.pop()
        if is_goal(current):
            return path + [current], len(visited), time.time() - start_time
        if len(path) >= limit:
            continue
        key = state_to_tuple(current)
        if key not in visited:
            visited.add(key)
            for neighbor in neighbors(current):
                stack.append((neighbor, path + [current]))
    return None, len(visited), time.time() - start_time

def astar(start, heuristic_func):
    start_time = time.time()
    visited = set()
    heap = [(heuristic_func(start), 0, start, [])]
    while heap:
        f, cost, current, path = heapq.heappop(heap)
        if is_goal(current):
            return path + [current], len(visited), time.time() - start_time
        key = state_to_tuple(current)
        if key not in visited:
            visited.add(key)
            for neighbor in neighbors(current):
                g = cost + 1
                h = heuristic_func(neighbor)
                heapq.heappush(heap, (g + h, g, neighbor, path + [current]))
    return None, len(visited), time.time() - start_time

def print_result(name, result):
    if result is None or result[0] is None:
        print(f"--- {name} ---")
        print("Nenhuma solução encontrada.")
        if result:
            print(f"Nós visitados: {result[1]}")
            print(f"Tempo: {result[2]:.4f}s")
        print()
    else:
        path, nodes, duration = result
        print(f"--- {name} ---")
        print(f"Solução em {len(path) - 1} movimentos")
        print(f"Nós visitados: {nodes}")
        print(f"Tempo: {duration:.4f}s")
        print()


if __name__ == "__main__":
    initial_state = [[3, 4, 6], [5, 8, 0], [2, 1, 7]]

    bfs_result = bfs(initial_state)
    dfs_result = dfs(initial_state)
    astar_manhattan = astar(initial_state, heuristic_manhattan)
    astar_misplaced = astar(initial_state, heuristic_misplaced)

    print_result("Busca em Largura", bfs_result)
    print_result("Busca em Profundidade", dfs_result)
    print_result("A* com Manhattan", astar_manhattan)
    print_result("A* com Peças fora do lugar", astar_misplaced)
    input("\nPressione ENTER para sair...")


