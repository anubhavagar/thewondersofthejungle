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

        return {
            "happiness": random.choice(happiness_levels),
            "stress": random.choice(stress_levels),
            "hydration": random.choice(hydration_levels),
            "energy": random.choice(energy_levels)
        }

vision_model = VisionModel()
