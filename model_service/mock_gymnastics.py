import random

class GymnasticsModel:
    def analyze_pose(self, media_data: str, media_type: str = 'image'):
        # Simulated gymnastics scoring based on data length
        seed_val = len(media_data) if media_data else 0
        random.seed(seed_val)
        
        difficulty = round(random.uniform(2.0, 6.0), 1)
        execution = round(random.uniform(7.0, 9.5), 1)
        artistry = round(random.uniform(0.5, 2.0), 1)
        
        total_score = round(difficulty + execution + artistry, 2)
        
        comments = [
            "Great toe point! 🩰",
            "Keep those legs straight! 📏",
            "Amazing flexibility! 🐍",
            "Strong landing! 🦶",
            "Beautiful extension! ✨",
            "Nice airtime! 🦅"
        ]

        result = {
            "difficulty": difficulty,
            "execution": execution,
            "artistry": artistry,
            "total_score": total_score,
            "comment": random.choice(comments),
            "is_video": media_type == 'video'
        }

        if media_type == 'video':
            # dynamic metrics
            result["advanced_metrics"] = {
                "gaze_stability": random.choice(["Target Locked ✅", "Wandering ⚠️", "Steady"]),
                "blink_rate": f"{random.randint(10, 25)} / min",
                "micro_expressions": random.choice(["None Detected", "Hesitation (0.2s)", "Fear Flicker"]),
                "head_rotation": f"{random.randint(200, 400)} deg/sec",
                "music_sync": f"{random.randint(85, 100)}%"
            }
        
        return result

gymnastics_model = GymnasticsModel()
