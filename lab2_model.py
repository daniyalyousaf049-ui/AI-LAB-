"""
Lab 2: Intelligent Agents & the PEAS Framework - Task 3 Implementation
Model-Based / Stateful Agent Implementation avoiding Redundant Commands

Author: Student (Lab 2 Submission)
Date: Academic Year 2026
"""

class StatefulClassroomAgent:
    """
    Model-based agent maintaining internal state of the previous active mode.
    Only issues control commands when the target mode differs from the current mode.
    """
    def __init__(self, initial_mode: str = "IDLE"):
        self.previous_mode = initial_mode

    def act(self, temp: float, occupied: bool):
        # Determine desired target mode based on operational rules
        if not occupied:
            target_mode = "ECO"
        elif temp > 26.0:
            target_mode = "COOL"
        elif temp < 20.0:
            target_mode = "WARM"
        else:
            target_mode = "IDLE"

        # Check if control command needs to be sent
        if target_mode != self.previous_mode:
            command_status = "SEND_COMMAND"
            self.previous_mode = target_mode
        else:
            command_status = "NO_COMMAND"

        return target_mode, command_status

def main():
    print("=" * 75)
    print("LAB 2: MODEL-BASED CLASSROOM AGENT (STATEFUL DECISION ENGINE)")
    print("=" * 75)
    
    agent = StatefulClassroomAgent(initial_mode="IDLE")
    
    # Test sequence: repeated percepts, exact boundaries, mode transitions, empty room
    percept_sequence = [
        (29.0, True, "Initial cooling request in occupied room"),
        (29.0, True, "Repeated high temp - should suppress redundant command"),
        (26.0, True, "Upper boundary test (26°C -> IDLE transition)"),
        (20.0, True, "Lower boundary test (20°C -> IDLE, remains IDLE)"),
        (19.0, True, "Heating request (<20°C -> WARM transition)"),
        (29.0, False, "Empty room override (Unoccupied -> ECO transition)")
    ]
    
    print(f"{'Step':<5} | {'Prev Mode':<10} | {'Percept (Temp, Occ)':<22} | {'Target':<8} | {'Command Action':<14} | {'Scenario Note'}")
    print("-" * 85)
    
    for step, (temp, occupied, note) in enumerate(percept_sequence, 1):
        prev = agent.previous_mode
        target, status = agent.act(temp, occupied)
        print(f"{step:<5} | {prev:<10} | ({temp:4.1f}°C, {str(occupied):<5})      | {target:<8} | {status:<14} | {note}")

if __name__ == "__main__":
    main()
