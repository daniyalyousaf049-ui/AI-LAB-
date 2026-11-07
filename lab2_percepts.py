"""
Lab 2: Intelligent Agents & the PEAS Framework - Guided Percept Trace
Reflex Agent Implementation for Smart Classroom Energy Management

Author: Student (Lab 2 Submission)
Date: Academic Year 2026
"""

class ReflexClassroomAgent:
    """
    Simple reflex agent mapping (temperature, occupancy) percepts to target modes.
    Note: Has no internal state; evaluates each percept independently.
    """
    def act(self, temp: float, occupied: bool) -> str:
        if not occupied:
            return "ECO"
        if temp > 26.0:
            return "COOL"
        if temp < 20.0:
            return "WARM"
        return "IDLE"

def main():
    print("=" * 65)
    print("LAB 2: REFLEX CLASSROOM AGENT - PERCEPT TRACE")
    print("=" * 65)
    
    agent = ReflexClassroomAgent()
    percepts = [
        (29.0, True),
        (29.0, True),
        (20.0, True),
        (26.0, True),
        (19.0, True),
        (29.0, False)
    ]
    
    print(f"{'Step':<6} | {'Percept (Temp °C, Occupied)':<28} | {'Target Mode':<12}")
    print("-" * 65)
    
    for step, percept in enumerate(percepts, 1):
        target = agent.act(*percept)
        print(f"{step:<6} | Temp={percept[0]:<4.1f}, Occupied={str(percept[1]):<5} | {target:<12}")

if __name__ == "__main__":
    main()
