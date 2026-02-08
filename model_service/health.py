class HealthModel:
    def analyze_data(self, health_data):
        # Simulated logic
        steps = health_data.get("steps", 0)
        advice = "Keep moving!"
        if steps > 10000:
            advice = "Roaring success! You are active."
        elif steps > 5000:
            advice = "Good job, young cub. Keep it up."
            
        return {
            "advice": advice,
            "status": "Healthy"
        }

health_model = HealthModel()
