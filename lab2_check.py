"""
Lab 2: Intelligent Agents & the PEAS Framework - Verification & Edge-Case Testing
Comparative Test Suite & Hysteresis Failure Analysis

Author: Student (Lab 2 Submission)
Date: Academic Year 2026
"""

from lab2_percepts import ReflexClassroomAgent
from lab2_model import StatefulClassroomAgent

def test_noisy_fluctuations():
    print("=" * 80)
    print("VERIFICATION SUITE & HYSTERESIS EDGE-CASE ANALYSIS")
    print("=" * 80)
    print("\n1. Comparative Run: Reflex vs Stateful Agent across Fluctuating Temperatures")
    print("-" * 80)
    
    noisy_sequence = [
        (26.5, True, "Initial warm reading (>26)"),
        (25.9, True, "Minor drop below 26 (Boundary flip)"),
        (26.2, True, "Minor rise above 26 (Boundary flip)"),
        (25.8, True, "Minor drop below 26 (Boundary flip)")
    ]
    
    reflex_agent = ReflexClassroomAgent()
    stateful_agent = StatefulClassroomAgent(initial_mode="IDLE")
    
    print(f"{'Step':<5} | {'Temp':<6} | {'Reflex Target':<14} | {'Stateful Target':<16} | {'Stateful Command'}")
    print("-" * 70)
    
    for step, (temp, occ, note) in enumerate(noisy_sequence, 1):
        ref_target = reflex_agent.act(temp, occ)
        st_target, st_cmd = stateful_agent.act(temp, occ)
        print(f"{step:<5} | {temp:<6.1f} | {ref_target:<14} | {st_target:<16} | {st_cmd}")
        
    print("\n[Hysteresis Failure Analysis]:")
    print("Notice that while the stateful agent suppresses duplicate consecutive commands,")
    print("rapid fluctuations around 26.0°C (26.5 -> 25.9 -> 26.2) cause repeated mode flips")
    print("between COOL and IDLE. Maintaining state without a deadband/hysteresis margin")
    print("fails to prevent rapid actuator chatter.")

if __name__ == "__main__":
    test_noisy_fluctuations()
