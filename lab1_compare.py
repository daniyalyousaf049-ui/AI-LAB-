"""
Lab 1: Artificial Intelligence & Paradigms - Guided Comparison Experiment
Symbolic Rule-Based vs. Probabilistic / Data-Driven Threshold Comparison

Author: Student (Lab 1 Submission)
Date: Academic Year 2026
"""

def symbolic_rule(score: float, complete: bool) -> str:
    """Symbolic rule explicitly checking completeness and score threshold."""
    return "Review" if complete and score >= 70.0 else "Hold"

def probability_threshold(probability: float) -> str:
    """Probabilistic threshold decision rule."""
    return "Review" if probability >= 0.70 else "Hold"

def run_experiment():
    print("-" * 65)
    print("SCHOLARSHIP SCREENING EXPERIMENT: SYMBOLIC VS PROBABILISTIC")
    print("-" * 65)
    
    cases = [
        ("complete_high", 82, True, 0.81),
        ("complete_low", 68, True, 0.74),
        ("incomplete_high", 91, False, 0.88),
        ("boundary", 70, True, 0.70)
    ]
    
    print(f"{'Case Name':<18} | {'Score':<5} | {'Complete':<8} | {'Prob':<5} | {'Symbolic':<8} | {'Threshold':<9} | {'Disagreement'}")
    print("-" * 75)
    
    for name, score, complete, prob in cases:
        sym_res = symbolic_rule(score, complete)
        prob_res = probability_threshold(prob)
        disagreement = (sym_res != prob_res)
        print(f"{name:<18} | {score:<5} | {str(complete):<8} | {prob:<5.2f} | {sym_res:<8} | {prob_res:<9} | {disagreement}")

    print("\n" + "-" * 65)
    print("SENSITIVITY ANALYSIS (BOUNDARY MODIFICATION)")
    print("-" * 65)
    print("Testing Boundary modified to score=69, prob=0.69:")
    sym_mod = symbolic_rule(69, True)
    prob_mod = probability_threshold(0.69)
    print(f"Modified Boundary (Score=69, Complete=True): Symbolic -> {sym_mod}")
    print(f"Modified Boundary (Prob=0.69)                : Threshold -> {prob_mod}")

if __name__ == "__main__":
    run_experiment()
