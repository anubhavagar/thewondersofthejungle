
from typing import List, Dict, Optional

class MAGDScoreCalculator:
    """
    Calculates the D-Score (Difficulty Score) for Men's Artistic Gymnastics
    based on FIG Code of Points (2025-2028).
    
    Formula: D-Score = DV (Top 10) + EG (Element Groups) + CV (Connection Value)
    Note: FIG MAG uses Top 10 skills for Senior, unlike WAG which uses Top 8.
    """
    
    def __init__(self):
        # Element Groups (EG) - Each fulfilled group gives 0.5 bonus, max 2.0 (for 4 groups)
        # These are usually different per apparatus.
        self.eg_definitions = {
            "EG1": "Group I: Floor Exercise (Acrobatic elements)",
            "EG2": "Group II: Pommel Horse (Circles and flairs)",
            "EG3": "Group III: Still Rings (Strength and swing)",
            "EG4": "Group IV: Vault / Dismounts"
        }

    def calculate_d_score(self, identified_skills: List[Dict], apparatus: str = "Floor Exercise", category: str = "Senior") -> Dict:
        """
        Calculate MAG D-Score from a list of recognized skills.
        
        Senior: FIG Rules (Top 10 DV + EG + CV)
        Junior: FIG Rules (Top 8 DV + EG + CV) - depending on specific national rules
        U10: Compulsory Model (D = 10.0 - Deductions)
        """
        total_d_score = 0.0
        top_skills_total = 0.0
        eg_score = 0.0
        cv_score = 0.0
        eg_fulfilled = []
        
        # 1. Sort skills by Difficulty Value (DV)
        valid_skills = [s for s in identified_skills if s.get('dv', 0) > 0]
        valid_skills.sort(key=lambda x: x['dv'], reverse=True)
        
        # 2. Determine number of skills to count based on category
        num_to_count = 10 if category == "Senior" else 8
        top_skills = valid_skills[:num_to_count]
        
        skill_groups_present = set(s.get('elementGroup', 0) for s in valid_skills)

        # 1. CATEGORY: U10 (Compulsory Logic)
        if category == "U10":
            base_d = 10.0
            deductions = []
            
            # Simple requirement check for U10
            required_groups = [1, 2, 3] # Mock requirement
            for group_id in required_groups:
                if group_id not in skill_groups_present:
                    base_d -= 1.0
                    deductions.append(f"Missing Element Group {group_id} (-1.0)")
            
            if len(valid_skills) < 5:
                penalty = (5 - len(valid_skills)) * 0.5
                base_d -= penalty
                deductions.append(f"Insufficient Skill Volume (-{penalty:.1f})")

            total_d_score = max(0.0, base_d)
            
            structured_rationale = {
                "formula": "D = 10.0 - [Missing Requirements]",
                "values": {
                    "base": 10.0,
                    "deductions": 10.0 - total_d_score,
                    "total": total_d_score
                },
                "reasons": [{"label": "Compulsory Fault", "text": d, "value": ""} for d in deductions]
            }
            if not deductions:
                structured_rationale["reasons"].append({"label": "Execution", "text": "All compulsory requirements met", "value": "+10.0"})

        # 2. CATEGORY: Senior / Junior (FIG Standard)
        else:
            top_skills_total = sum(s['dv'] for s in top_skills)
            
            # Element Groups Logic (EG)
            # In MAG, each apparatus has 4 groups usually. 
            # fulfilling each group contributes 0.5.
            for i in range(1, 5):
                if i in skill_groups_present:
                    eg_score += 0.5
                    eg_fulfilled.append(f"Group {i}")
            
            total_d_score = top_skills_total + eg_score + cv_score
            dv_names = ", ".join([s['name'] for s in top_skills]) if top_skills else "None"
            dv_count = len(top_skills)

            structured_rationale = {
                "formula": f"D = [DV (Top {num_to_count})] + [EG] + [CV]",
                "values": {
                    "dv": top_skills_total,
                    "eg": eg_score,
                    "cv": cv_score,
                    "total": total_d_score
                },
                "reasons": [
                    {
                        "label": "Difficulty Value (DV)",
                        "text": f"Recognized {dv_count} element(s) ({dv_names})",
                        "value": f"+{top_skills_total:.1f}"
                    },
                    {
                        "label": "Element Groups (EG)",
                        "text": (", ".join(eg_fulfilled) if eg_fulfilled else "No requirements met"),
                        "value": f"+{eg_score:.1f}"
                    },
                    {
                        "label": "Connection Value (CV)",
                        "text": "No eligible connections detected",
                        "value": f"+{cv_score:.1f}"
                    }
                ]
            }

        return {
            "total_d_score": total_d_score,
            "top_skills_total": top_skills_total,
            "top_skills": [s['name'] for s in top_skills],
            "eg_score": eg_score,
            "eg_fulfilled": eg_fulfilled,
            "cv_score": cv_score,
            "rationale": structured_rationale
        }

# Example Usage
if __name__ == "__main__":
    calculator = MAGDScoreCalculator()
    routine = [
        {"name": "Iron Cross", "dv": 0.2, "elementGroup": 2},
        {"name": "Planche", "dv": 0.3, "elementGroup": 2},
        {"name": "Press Handstand", "dv": 0.2, "elementGroup": 1},
        {"name": "Maltese", "dv": 0.4, "elementGroup": 2},
        {"name": "L-Sit", "dv": 0.1, "elementGroup": 1},
        {"name": "V-Sit", "dv": 0.2, "elementGroup": 1}
    ]
    print(calculator.calculate_d_score(routine, apparatus="Still Rings", category="Senior"))
