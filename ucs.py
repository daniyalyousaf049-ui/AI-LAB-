"""
Lab 3: Uninformed Search Algorithms - Task 3
Implementation of Uniform-Cost Search (UCS) on Weighted Corridor Extension.
Compares hop-count vs minimum travel cost against BFS.

Date: Academic Year 2026
"""

import heapq
from collections import deque

WEIGHTED_GRAPH = {
    "A": [("B", 1), ("C", 2)],
    "B": [("D", 4), ("E", 1)],
    "C": [("F", 3), ("G", 10)],  
    "D": [],
    "E": [("G", 1)],            
    "F": [],
    "G": []
}

def uniform_cost_search(graph, start, goal):
    """
    Executes Uniform-Cost Search (UCS) using a min-priority queue ordered by accumulated path cost.
    Returns: (path, total_cost, visit_order, trace_log)
    """
 
    pq = [(0, start, [start])]
    explored = set()
    visit_order = []
    trace_log = []
    step = 1

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in explored:
            continue

        explored.add(node)
        visit_order.append(node)

        trace_log.append({
            "step": step,
            "processed": node,
            "cost": cost,
            "path": path
        })
        step += 1

        if node == goal:
            return path, cost, visit_order, trace_log

        for neighbor, edge_cost in graph[node]:
            if neighbor not in explored:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    return None, float("inf"), visit_order, trace_log

def bfs_weighted_topology(graph, start, goal):
    """
    Runs unweighted BFS on the weighted graph structure to find minimum hop path.
    Also calculates the total cost of that hop-optimal path for comparison.
    """
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if node == goal:
            # Calculate total cost of BFS path
            total_cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                for nxt, weight in graph[u]:
                    if nxt == v:
                        total_cost += weight
                        break
            return path, total_cost, len(path) - 1

        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, float("inf"), 0

def main():
    print("=" * 75)
    print("LAB 3 TASK 3: UNIFORM-COST SEARCH (UCS) VS BFS ON WEIGHTED GRAPH")
    print("=" * 75)
    
    start_node, goal_node = "A", "G"
    
    # Run UCS
    ucs_path, ucs_cost, ucs_visits, ucs_trace = uniform_cost_search(WEIGHTED_GRAPH, start_node, goal_node)
    
    print("UCS Node Expansion Trace ---")
    print(f"{'Step':<5} | {'Expanded Node':<15} | {'Accumulated Cost g(n)':<22} | {'Current Path'}")
    print("-" * 75)
    for t in ucs_trace:
        print(f"{t['step']:<5} | {t['processed']:<15} | {t['cost']:<22} | {' -> '.join(t['path'])}")
        
    print(f"UCS Final Path        : {' -> '.join(ucs_path)}")
    print(f"UCS Total Cost        : {ucs_cost} units")
    print(f"UCS Total Hop Count   : {len(ucs_path) - 1} hops")
    print(f"UCS Expansion Order   : {' -> '.join(ucs_visits)}")
    
    # Run BFS comparison
    bfs_path, bfs_cost, bfs_hops = bfs_weighted_topology(WEIGHTED_GRAPH, start_node, goal_node)
    
    print(" Comparative Analysis: UCS vs BFS ---")
    print(f"BFS Hop-Optimal Path  : {' -> '.join(bfs_path)}")
    print(f"BFS Path Total Cost   : {bfs_cost} units")
    print(f"BFS Total Hop Count   : {bfs_hops} hops")
    print("-" * 75)
    print("Key Finding: BFS selects A -> C -> G (2 hops, Cost = 12) because it ignores edge weights.")
    print("UCS selects A -> B -> E -> G (3 hops, Cost = 3) finding the true minimum cost path.")

if __name__ == "__main__":
    main()
