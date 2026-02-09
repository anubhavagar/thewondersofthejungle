
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

    def calculate_d_score(self, identified_skills: List[Dict]) -> Dict:
        """
        Calculate D-Score from a list of recognized skills.
        
        Args:
            identified_skills: List of dicts, e.g. [{"name": "Split Leap", "dv": 0.5, "type": "dance"}]
            
        Returns:
            Dict with breakdown: { "total_d_score": float, "top_8_total": float, "cv": float, "cr": float, "breakdown": ... }
        """
        # 1. Sort by Difficulty Value (DV) descending
        # Filter out skills with 0 DV or unrecognized
        valid_skills = [s for s in identified_skills if s.get('dv', 0) > 0]
        valid_skills.sort(key=lambda x: x['dv'], reverse=True)
        
        # 2. Take Top 8
        top_8 = valid_skills[:8]
        top_8_total = sum(s['dv'] for s in top_8)
        
        # 3. Calculate CR (Simplified for now based on skill types present)
        # We assume specific tags in the skill list to identify CRs
        cr_score = 0.0
        cr_fulfilled = []
        
        skill_types_present = set(s.get('type', '') for s in valid_skills)
        
        # Heuristic CR Logic (Placeholder)
        if 'dance' in skill_types_present: 
            cr_score += 0.5
            cr_fulfilled.append("CR1 (Dance)")
        if 'acro' in skill_types_present: 
            cr_score += 0.5
            cr_fulfilled.append("CR2 (Acro)")
        if 'turn' in skill_types_present: 
            cr_score += 0.5
            cr_fulfilled.append("CR3 (Turn)")
        if 'dismount' in skill_types_present:
            cr_score += 0.5
            cr_fulfilled.append("CR4 (Dismount)")
            
        # 4. Connection Value (CV)
        # This requires sequence analysis (e.g., direct connection between skills)
        # For now, we return 0.0 or allow it to be passed in
        cv_score = 0.0
        
        total_d_score = round(top_8_total + cr_score + cv_score, 3)
        
        return {
            "total_d_score": total_d_score,
            "top_8_total": top_8_total,
            "top_8_skills": [s['name'] for s in top_8],
            "cr_score": cr_score,
            "cr_fulfilled": cr_fulfilled,
            "cv_score": cv_score
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
