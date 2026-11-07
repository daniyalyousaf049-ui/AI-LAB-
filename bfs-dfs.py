"""
Lab 3: Uninformed Search Algorithms - Task 1
Implementation of Breadth-First Search (BFS) and Depth-First Search (DFS)
with detailed frontier tracing for the Emergency Exit Graph.

Date: Academic Year 2026
"""

from collections import deque

GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": []
}

def bfs(graph, start, goal):
    """
    Executes Breadth-First Search (BFS) using a FIFO queue.
    Tracks visited nodes on expansion to prevent redundant queueing.
    Returns: (path, visit_order, trace_log)
    """
    queue = deque([(start, [start])])
    visited = {start}
    visit_order = []
    trace_log = []
    step = 1

    while queue:
        frontier_snapshot = [node for node, _ in queue]
        node, path = queue.popleft()
        visit_order.append(node)
        
        trace_log.append({
            "step": step,
            "processed": node,
            "frontier": frontier_snapshot,
            "path": path
        })
        step += 1

        if node == goal:
            return path, visit_order, trace_log

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, visit_order, trace_log

def dfs(graph, start, goal):
    """
    Executes Depth-First Search (DFS) using a LIFO stack.
    Neighbors are pushed in reverse order so that left-to-right successors are processed first.
    Returns: (path, visit_order, trace_log)
    """
    stack = [(start, [start])]
    visited = set()
    visit_order = []
    trace_log = []
    step = 1

    while stack:
        frontier_snapshot = [node for node, _ in stack]
        node, path = stack.pop()
        
        if node in visited:
            continue
            
        visited.add(node)
        visit_order.append(node)

        trace_log.append({
            "step": step,
            "processed": node,
            "frontier": frontier_snapshot,
            "path": path
        })
        step += 1

        if node == goal:
            return path, visit_order, trace_log

        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, visit_order, trace_log

def main():
    print("=" * 70)
    print("LAB 3 TASK 1: BFS & DFS SEARCH TRACE ON EMERGENCY EXIT GRAPH")
    print("=" * 70)
    
    start_node, goal_node = "A", "G"
    
    # Run BFS
    bfs_path, bfs_visits, bfs_trace = bfs(GRAPH, start_node, goal_node)
    print("\n[BREADTH-FIRST SEARCH (BFS)]")
    print(f"{'Step':<5} | {'Processed Node':<15} | {'Frontier Queue State':<30} | {'Current Path'}")
    print("-" * 70)
    for t in bfs_trace:
        print(f"{t['step']:<5} | {t['processed']:<15} | {str(t['frontier']):<30} | {' -> '.join(t['path'])}")
    print(f"BFS Visit Order: {' -> '.join(bfs_visits)}")
    print(f"BFS Final Path : {' -> '.join(bfs_path)}")
    
    # Run DFS
    dfs_path, dfs_visits, dfs_trace = dfs(GRAPH, start_node, goal_node)
    print("\n[DEPTH-FIRST SEARCH (DFS)]")
    print(f"{'Step':<5} | {'Processed Node':<15} | {'Frontier Stack State':<30} | {'Current Path'}")
    print("-" * 70)
    for t in dfs_trace:
        print(f"{t['step']:<5} | {t['processed']:<15} | {str(t['frontier']):<30} | {' -> '.join(t['path'])}")
    print(f"DFS Visit Order: {' -> '.join(dfs_visits)}")
    print(f"DFS Final Path : {' -> '.join(dfs_path)}")

if __name__ == "__main__":
    main()