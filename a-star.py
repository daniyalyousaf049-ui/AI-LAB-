"""
Lab 4: Informed Search - Heuristics & A* Search
Implementation of A* Search, Heuristic Consistency Verification, UCS Comparison,
and Inadmissible Overestimate Failure Experiment.

Date: Academic Year 2026
"""

import heapq

GRAPH = {
    "S": [("A", 1), ("B", 4)],
    "A": [("C", 2)],
    "B": [("G", 5)],
    "C": [("G", 3)],
    "G": []
}

HEURISTIC = {
    "S": 5,
    "A": 4,
    "B": 4,
    "C": 2,
    "G": 0
}

def verify_consistency(graph, h):
    """Verifies consistency condition h(u) <= cost(u,v) + h(v) for all edges."""
    print("=" * 70)
    print("HEURISTIC CONSISTENCY & ADMISSIBILITY AUDIT")
    print("=" * 70)
    consistent = True
    for u in graph:
        for v, weight in graph[u]:
            h_u = h[u]
            h_v = h[v]
            lhs = h_u
            rhs = weight + h_v
            is_valid = lhs <= rhs
            if not is_valid:
                consistent = False
            print(f"Edge ({u} -> {v}): h({u})={h_u} <= cost({u},{v})={weight} + h({v})={h_v} ({rhs}) | Valid: {is_valid}")
    print(f"Overall Consistency Status: {consistent}")

def astar_search(graph, h, start, goal):
    """
    Executes A* Search using a min-priority queue ordered by f(n) = g(n) + h(n).
    Logs detailed trace, frontier states, and expansion counts.
    Returns: (path, total_cost, visit_order, expansion_count, trace_log)
    """
    pq = [(h[start], 0, start, [start])]
    best_g = {start: 0}
    visit_order = []
    trace_log = []
    expansion_count = 0
    step = 1

    while pq:

        frontier_snapshot = [(f, g, node) for f, g, node, _ in pq]
        f, g, node, path = heapq.heappop(pq)


        if g > best_g.get(node, float("inf")):
            continue

        visit_order.append(node)

        if node == goal:
        
            trace_log.append({
                "step": step,
                "node": node,
                "g": g,
                "h": h[node],
                "f": f,
                "path": path,
                "frontier": frontier_snapshot,
                "action": "GOAL_REACHED"
            })
            return path, g, visit_order, expansion_count, trace_log


        expansion_count += 1
        trace_log.append({
            "step": step,
            "node": node,
            "g": g,
            "h": h[node],
            "f": f,
            "path": path,
            "frontier": frontier_snapshot,
            "action": "EXPANDED"
        })
        step += 1

        for neighbor, weight in graph.get(node, []):
            ng = g + weight
            if ng < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = ng
                nf = ng + h[neighbor]
                heapq.heappush(pq, (nf, ng, neighbor, path + [neighbor]))

    return None, float("inf"), visit_order, expansion_count, trace_log

def main():
    print("=" * 70)
    print("LAB 4: A* SEARCH & UNINFORMED COMPARISON EXPERIMENT")
    print("=" * 70)
    
 
    verify_consistency(GRAPH, HEURISTIC)
    
    a_path, a_cost, a_visits, a_exp, a_trace = astar_search(GRAPH, HEURISTIC, "S", "G")
    print("--- 1. STANDARD A* SEARCH TRACE ---")
    print(f"{'Step':<5} | {'Node':<6} | {'g(n)':<6} | {'h(n)':<6} | {'f(n)':<6} | {'Current Path':<18} | {'Frontier Snapshot (f,g,node)'}")
    print("-" * 80)
    for t in a_trace:
        fr_str = str([(f, g, n) for f, g, n in t['frontier']])
        print(f"{t['step']:<5} | {t['node']:<6} | {t['g']:<6} | {t['h']:<6} | {t['f']:<6} | {'->'.join(t['path']):<18} | {fr_str}")
    print(f"A* Final Path       : {' -> '.join(a_path)}")
    print(f"A* Total Path Cost  : {a_cost}")
    print(f"A* Expansion Count  : {a_exp}")
    print(f"A* Visit Order      : {' -> '.join(a_visits)}")

    ucs_h = {n: 0 for n in HEURISTIC}
    u_path, u_cost, u_visits, u_exp, u_trace = astar_search(GRAPH, ucs_h, "S", "G")
    print("--- 2. UNIFORM COST SEARCH (UCS CONTROL: h(n)=0) ---")
    print(f"{'Step':<5} | {'Node':<6} | {'g(n)':<6} | {'h(n)':<6} | {'f(n)':<6} | {'Current Path':<18} | {'Frontier Snapshot (f,g,node)'}")
    print("-" * 80)
    for t in u_trace:
        fr_str = str([(f, g, n) for f, g, n in t['frontier']])
        print(f"{t['step']:<5} | {t['node']:<6} | {t['g']:<6} | {t['h']:<6} | {t['f']:<6} | {'->'.join(t['path']):<18} | {fr_str}")
    print(f"UCS Final Path      : {' -> '.join(u_path)}")
    print(f"UCS Total Path Cost : {u_cost}")
    print(f"UCS Expansion Count : {u_exp}")
    print(f"UCS Visit Order     : {' -> '.join(u_visits)}")

    # 3. Overestimate Experiment (h(C) = 10)
    over_h = HEURISTIC.copy()
    over_h["C"] = 10
    o_path, o_cost, o_visits, o_exp, o_trace = astar_search(GRAPH, over_h, "S", "G")
    print("--- 3. OVERESTIMATE FAILURE EXPERIMENT (h(C) = 10) ---")
    print(f"{'Step':<5} | {'Node':<6} | {'g(n)':<6} | {'h(n)':<6} | {'f(n)':<6} | {'Current Path':<18} | {'Frontier Snapshot (f,g,node)'}")
    print("-" * 80)
    for t in o_trace:
        fr_str = str([(f, g, n) for f, g, n in t['frontier']])
        print(f"{t['step']:<5} | {t['node']:<6} | {t['g']:<6} | {t['h']:<6} | {t['f']:<6} | {'->'.join(t['path']):<18} | {fr_str}")
    print(f"Overestimate Path   : {' -> '.join(o_path)}")
    print(f"Overestimate Cost   : {o_cost} (Suboptimal! True min cost is 6)")
    print(f"Overestimate Exp    : {o_exp}")
    print(f"Overestimate Visits : {' -> '.join(o_visits)}")

if __name__ == "__main__":
    main()
