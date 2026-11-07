"""
Lab 3: Uninformed Search Algorithms - Task 2
Implementation of Depth-Limited Search (DLS) and Iterative Deepening DFS (IDDFS)
Distinguishing FOUND, CUTOFF, and FAILURE states.

Date: Academic Year 2026
"""

GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": []
}

def depth_limited_search(graph, start, goal, limit):
    """
    Executes Depth-Limited Search (DLS) up to a specified maximum depth limit.
    Uses path-local cycle checking (path list) to allow proper re-evaluations across iterations.
    Returns: (status, path, nodes_visited)
    """
    nodes_visited = []

    def recursive_dls(node, path, depth):
        nodes_visited.append(node)
        if node == goal:
            return "FOUND", path
        if depth >= limit:
            return "CUTOFF", None

        cutoff_occurred = False
        for neighbor in graph[node]:
            if neighbor not in path:  # Path-local cycle prevention
                status, res_path = recursive_dls(neighbor, path + [neighbor], depth + 1)
                if status == "FOUND":
                    return "FOUND", res_path
                if status == "CUTOFF":
                    cutoff_occurred = True

        return ("CUTOFF" if cutoff_occurred else "FAILURE"), None

    status, path = recursive_dls(start, [start], 0)
    return status, path, nodes_visited

def iterative_deepening_dfs(graph, start, goal, max_limit):
    """
    Executes Iterative Deepening Depth-First Search (IDDFS) up to max_limit.
    Iterates depth limits from 0 up to max_limit.
    Returns detailed trace log for each depth iteration.
    """
    history = []
    for limit in range(max_limit + 1):
        status, path, visits = depth_limited_search(graph, start, goal, limit)
        history.append({
            "limit": limit,
            "status": status,
            "path": path,
            "visits": visits
        })
        if status == "FOUND":
            break
    return history

def main():
    print("=" * 70)
    print("LAB 3 TASK 2: DEPTH-LIMITED (DLS) & ITERATIVE DEEPENING (IDDFS) TESTS")
    print("=" * 70)
    
    start_node, goal_node = "A", "G"
    limits_to_test = [1, 2, 3, 4]
    
    print("PART 1: Individual Depth-Limited Search (DLS) Runs ---")
    print(f"{'Limit':<6} | {'Status':<10} | {'Visited Nodes Order':<30} | {'Returned Path'}")
    print("-" * 75)
    
    for lim in limits_to_test:
        status, path, visits = depth_limited_search(GRAPH, start_node, goal_node, lim)
        path_str = " -> ".join(path) if path else "None"
        visits_str = " -> ".join(visits)
        print(f"{lim:<6} | {status:<10} | {visits_str:<30} | {path_str}")

    print("PART 2: Iterative Deepening DFS (IDDFS) Trace (Max Limit = 4) ---")
    iddfs_results = iterative_deepening_dfs(GRAPH, start_node, goal_node, max_limit=4)
    print(f"{'Iteration Limit':<16} | {'Outcome Status':<12} | {'Nodes Processed in Iteration'}")
    print("-" * 75)
    for res in iddfs_results:
        visits_str = " -> ".join(res["visits"])
        print(f"Limit = {res['limit']:<9} | {res['status']:<12} | {visits_str}")

if __name__ == "__main__":
    main()
