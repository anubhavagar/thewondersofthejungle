import random

class VisionModel:
    def analyze_face(self, image_data: str):
        # Simulated logic for "Face Analysis"
        # We use the length of the image data as a seed for reproducibility
        # This ensures the same image gets the same result, but results vary between images.
        seed_val = len(image_data) if image_data else 0
        random.seed(seed_val)
        
        happiness_levels = ["High", "Simba-level", "Moderate", "Timon-style (cautious)"]
        stress_levels = ["Low", "Hakuna Matata (Zero)", "Moderate", "Elevated"]
        hydration_levels = ["Good", "Water hole needed", "Excellent", "Dry as the desert"]
        energy_levels = ["Simba-like", "Nala-speed", "Sleepy Pumba", "Zazu-frantic"]
        
        # New Gymnast Insights
        recovery_scores = ["95% (Fully Reposed)", "80% (Ready to Pounce)", "60% (Needs a Nap)", "40% (Roar-less)"]
        focus_levels = ["Eagle-eyed", "Prideland-sharp", "Monkeying around", "Foggy Forest"]
        stability_levels = ["Mountain Steady", "Flamingo-still", "Tree-root solid", "Wobbly Giraffe"]
        elasticity_levels = ["Spring-loaded", "Bouncy Baboon", "Cheetah-snap", "Stiff Zebra"]
        grit_levels = ["Mufasa Strength", "Cub Persistence", "Lead-Lion Spirit", "Sleepy Cub"]
        
        tips = [
            "Great focus! Perfect time to practice those leaps.",
            "Energy is a bit low. Maybe try some bananas before training?",
            "You are in Simba-mode! Go for the high bar today.",
            "Need more water! Hydration is the key to a perfect landing.",
            "Stress is low! Perfect vibe for a competition rehearsal."
        ]

        return {
            "happiness": random.choice(happiness_levels),
            "stress": random.choice(stress_levels),
            "hydration": random.choice(hydration_levels),
            "energy": random.choice(energy_levels),
            "recovery": random.choice(recovery_scores),
            "focus": random.choice(focus_levels),
            "stability": random.choice(stability_levels),
            "elasticity": random.choice(elasticity_levels),
            "grit": random.choice(grit_levels),
            "daily_tip": random.choice(tips)
        }

vision_model = VisionModel()
