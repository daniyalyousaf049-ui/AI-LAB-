"""
Lab 1: Artificial Intelligence & Paradigms - Task 2
Implementation of Rule-Based Decision System with Synthetic Criteria

Author: Student (Lab 1 Submission)
Date: Academic Year 2026
"""

def rule_based_scholarship(score: float, documents_complete: bool, prerequisite_gpa: float, valid_recommendation: bool) -> str:
    """
    Evaluates scholarship eligibility using four transparent, explicit criteria.
    Note: Thresholds used herein are synthetic lab assumptions.
    
    Criteria:
    1. Document completeness (documents_complete == True)
    2. Test Score >= 70.0
    3. Prerequisite GPA >= 3.0
    4. Valid Recommendation Letter (valid_recommendation == True)
    """
    if not documents_complete:
        return "Hold"
    if score < 70.0:
        return "Hold"
    if prerequisite_gpa < 3.0:
        return "Hold"
    if not valid_recommendation:
        return "Hold"
    return "Review"

def main():
    print("-" * 69)
    print("SCHOLARSHIP SCREENING PROTOTYPE - RULE-BASED SYSTEM (TASK 2)")
    print("Note: Thresholds used in this script are synthetic lab assumptions.")
    print("-" * 69)
    
    # 4 Test Cases 
    test_cases = [
        {
            "case_id": "Case 1: Standard Pass",
            "inputs": {"score": 85.0, "documents_complete": True, "prerequisite_gpa": 3.5, "valid_recommendation": True},
            "expected": "Review",
            "reason": "Meets all 4 explicit criteria (Score 85 >= 70, Docs True, GPA 3.5 >= 3.0, Rec True)."
        },
        {
            "case_id": "Case 2: Fail New Cond 1 (Prerequisite GPA)",
            "inputs": {"score": 92.0, "documents_complete": True, "prerequisite_gpa": 2.8, "valid_recommendation": True},
            "expected": "Hold",
            "reason": "High test score (92.0) and complete docs, but fails minimum GPA requirement (2.8 < 3.0)."
        },
        {
            "case_id": "Case 3: Fail New Cond 2 (Recommendation Letter)",
            "inputs": {"score": 78.0, "documents_complete": True, "prerequisite_gpa": 3.2, "valid_recommendation": False},
            "expected": "Hold",
            "reason": "Meets score, docs, and GPA, but lacks a valid recommendation letter."
        },
        {
            "case_id": "Case 4: Threshold Boundary",
            "inputs": {"score": 70.0, "documents_complete": True, "prerequisite_gpa": 3.0, "valid_recommendation": True},
            "expected": "Review",
            "reason": "Exact boundary values for score (70.0) and GPA (3.0); should pass edge-case condition."
        }
    ]
    
    for test in test_cases:
        inp = test["inputs"]
        actual = rule_based_scholarship(
            score=inp["score"],
            documents_complete=inp["documents_complete"],
            prerequisite_gpa=inp["prerequisite_gpa"],
            valid_recommendation=inp["valid_recommendation"]
        )
        print(f"\n[{test['case_id']}]")
        print(f"  Inputs          : Score={inp['score']}, Complete={inp['documents_complete']}, GPA={inp['prerequisite_gpa']}, Rec={inp['valid_recommendation']}")
        print(f"  Expected Result : {test['expected']}")
        print(f"  Actual Result   : {actual}")
        print(f"  Selection Reason: {test['reason']}")
        assert actual == test["expected"], f"Test failed for {test['case_id']}"

    print("\n" + "-" * 69)
    print("All 4 test cases executed successfully.")

if __name__ == "__main__":
    main()
