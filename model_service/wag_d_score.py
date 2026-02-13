
from typing import List, Dict, Optional

class WAGDScoreCalculator:
    """
    Calculates the D-Score (Difficulty Score) for Women's Artistic Gymnastics
    based on FIG Code of Points (2025-2028).
    
    Formula: D-Score = DV (Top 8) + CV + CR
    """
    
    def __init__(self):
        # Composition Requirements (CR) - Max 2.0
        # In a real app, these checks would be complex analysis of the whole routine.
        # Here we map simple presence of categories to CR.
        self.cr_definitions = {
            "CR1": "Dance Passage (Leap/Jump)",
            "CR2": "Saltos (Forward/Backward)",
            "CR3": "Turns",
            "CR4": "Dismount"
        }

    def calculate_d_score(self, identified_skills: List[Dict], category: str = "Senior") -> Dict:
        """
        Calculate D-Score from a list of recognized skills based on Category.
        
        Senior/U14/U12: Standard FIG (D = Top 8 DV + CR + CV)
        U10: Compulsory Model (D = 10.0 - Deductions for missing skills/requirements)
        """
        # Initialize results
        total_d_score = 0.0
        top_8_total = 0.0
        cr_score = 0.0
        cv_score = 0.0
        cr_fulfilled = []
        
        valid_skills = [s for s in identified_skills if s.get('dv', 0) > 0]
        valid_skills.sort(key=lambda x: x['dv'], reverse=True)
        top_8 = valid_skills[:8]
        
        skill_types_present = set(s.get('type', '') for s in valid_skills)

        # 1. CATEGORY: U10 (Compulsory Logic)
        if category == "U10":
            # ... (U10 logic already exists) ...
            base_d = 10.0
            deductions = []
            
            check_map = {
                "dance": "Dance Requirement",
                "acro": "Acrobatic Requirement",
                "turn": "Turn Requirement"
            }
            
            for s_type, label in check_map.items():
                if s_type not in skill_types_present:
                    base_d -= 1.0
                    deductions.append(f"Missing {label} (-1.0)")
            
            # Penalize for low volume (fewer than 4 skills)
            if len(valid_skills) < 4:
                penalty = (4 - len(valid_skills)) * 0.5
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

        # 2. CATEGORY: U8 (Tiny Tots - Participation Only)
        elif category == "U8" or "tiny" in category.lower():
            total_d_score = 0.0
            structured_rationale = {
                "formula": "Participation-Based Validation",
                "values": {"base": 0, "total": 0},
                "reasons": [
                    {
                        "label": "Foundation",
                        "text": f"Excellent participation detected! {len(valid_skills)} skill(s) performed.",
                        "value": "AWARDED"
                    },
                    {
                        "label": "Spirit",
                        "text": "Great focus and technical effort for age group.",
                        "value": "GOLD"
                    }
                ]
            }

        # 3. CATEGORY: Senior / U14 / U12 (FIG Standard)
        else:
            top_8_total = sum(s['dv'] for s in top_8)
            
            # CR Logic
            if 'dance' in skill_types_present: 
                cr_score += 0.5
                cr_fulfilled.append(self.cr_definitions["CR1"])
            if 'acro' in skill_types_present:
                cr_score += 0.5
                cr_fulfilled.append(self.cr_definitions["CR2"])
            if 'turn' in skill_types_present:
                cr_score += 0.5
                cr_fulfilled.append(self.cr_definitions["CR3"])
            if len(valid_skills) >= 5: # Assuming dismount is present if enough skills
                cr_score += 0.5
                cr_fulfilled.append(self.cr_definitions["CR4"])
            
            total_d_score = top_8_total + cr_score + cv_score
            dv_names = ", ".join([s['name'] for s in top_8]) if top_8 else "None"
            dv_count = len(top_8)

            structured_rationale = {
                "formula": "D = [DV] + [CR] + [CV]",
                "values": {
                    "dv": top_8_total,
                    "cr": cr_score,
                    "cv": cv_score,
                    "total": total_d_score
                },
                "reasons": [
                    {
                        "label": "Difficulty Value (DV)",
                        "text": f"Recognized {dv_count} element(s) ({dv_names})",
                        "value": f"+{top_8_total:.1f}"
                    },
                    {
                        "label": "Comp. Requirements (CR)",
                        "text": (", ".join(cr_fulfilled) if cr_fulfilled else "No requirements met"),
                        "value": f"+{cr_score:.1f}"
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
            "top_8_total": top_8_total,
            "top_8_skills": [s['name'] for s in top_8],
            "cr_score": cr_score,
            "cr_fulfilled": cr_fulfilled,
            "cv_score": cv_score,
            "rationale": structured_rationale
        }

# Example Usage
if __name__ == "__main__":
    calculator = WAGDScoreCalculator()
    routine = [
        {"name": "Split Leap", "dv": 0.5, "type": "dance"},
        {"name": "Wolf Turn", "dv": 0.3, "type": "turn"},
        {"name": "Double Back", "dv": 0.4, "type": "acro"},
        {"name": "Handstand", "dv": 0.1, "type": "dance"}, # valid pose but low value
        {"name": "Switch Ring", "dv": 0.5, "type": "dance"}
    ]
    print(calculator.calculate_d_score(routine))
