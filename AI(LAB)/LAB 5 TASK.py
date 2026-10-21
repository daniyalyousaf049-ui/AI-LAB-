import math

def run_minimax(depth, node_idx, is_max, scores, target_depth):
    if depth == target_depth:
        return scores[node_idx], 1

    if is_max:
        leftsidevalue, leftcount = run_minimax(depth + 1, node_idx * 2, False, scores, target_depth)
        rightsidevalue, rightcount = run_minimax(depth + 1, node_idx * 2 + 1, False, scores, target_depth)
        return max(leftsidevalue, rightsidevalue), (leftcount + rightcount)
    else:
        leftsidevalue, leftcount = run_minimax(depth + 1, node_idx * 2, True, scores, target_depth)
        rightsidevalue, rightcount = run_minimax(depth + 1, node_idx * 2 + 1, True, scores, target_depth)
        return min(leftsidevalue, rightsidevalue), (leftcount + rightcount)

def run_alphabeta(depth, node_idx, is_max, scores, target_depth, alpha, beta):
    if depth == target_depth:
        return scores[node_idx],1
    
    total_leaves = 0

    if is_max:
        bestvalue = float('-inf')
        for i in range(2):
            value, count = run_alphabeta(depth + 1, node_idx * 2 + i, False, scores, target_depth, alpha, beta)
            total_leaves += count
            bestvalue = max(bestvalue, value)
            alpha = max(alpha, bestvalue)
            if alpha >= beta:
                break
        return bestvalue, total_leaves
    else:
        bestvalue = float('inf')
        for i in range(2):
            value, count = run_alphabeta(depth + 1, node_idx * 2 + i, True, scores, target_depth, alpha, beta)
            total_leaves += count
            bestvalue = min(bestvalue, value)
            beta = min(beta, bestvalue)
            if alpha >= beta:
                break
        return bestvalue, total_leaves

experiments=[
    ("Default Lab Tree", [3,5,2,9,3,5,2,9]),
    ("Best Move Order", [5,3,9,2,2,3,9,5]),
    ("Worst Move Order", [2,3,2,3,5,9,5,9]),
    ("Uniform Values", [4,4,4,4,4,4,4,4])
]

for title, score_list in experiments:
    tree_depth = int(math.log2(len(score_list)))
    
    mm_score, mm_leaves = run_minimax(0, 0, True, score_list, tree_depth)
    ab_score, ab_leaves = run_alphabeta(0, 0, True, score_list, tree_depth, float('-inf'), float('inf'))
    
    print("Test:", title)
    print("Array:", score_list)
    print(f"Minimax value: {mm_score} | Leaves Evaluated: {mm_leaves}")
    print(f"AlphaBeta value: {ab_score} | Leaves Evaluated: {ab_leaves}")
    print("-" * 45)
