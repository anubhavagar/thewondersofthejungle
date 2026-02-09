
# Global Gymnastics Standards - Skill Data Structure
# Based on FIG (Fédération Internationale de Gymnastique) Code of Points

SKILL_DATA = {
    "Iron Cross": {
        "name": "Iron Cross",
        "category": "Strength",
        "difficulty": "B",
        "elementGroup": 2,
        "description": "A static strength hold on the rings where the body is suspended with arms extended horizontally.",
        "technicalRequirements": [
            "Arms must be parallel to the floor (90° relative to torso).",
            "Hold must be static for a minimum of 2 seconds.",
            "Body must maintain a straight or slightly hollow line."
        ],
        "focusAnatomy": ["Pectoralis Major", "Latissimus Dorsi", "Scapular Depressors"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Arms slightly above or below horizontal" },
            { "penalty": 0.3, "reason": "Bent elbows or broken wrist line" },
            { "penalty": 0.5, "reason": "Failure to hold for 2 seconds" }
        ],
        "technicalCue": "Depress the scapula and 'lock' the lats while turning the rings slightly out."
    },

    "Planche": {
        "name": "Planche (Straddle/Straight)",
        "category": "Strength",
        "difficulty": "C",
        "elementGroup": 2,
        "description": "A high-level static strength hold where the body is held parallel to the ground, supported only by the hands.",
        "technicalRequirements": [
            "Body must be perfectly horizontal.",
            "Arms must be fully locked (no elbow flexion).",
            "Protract the scapula to create a slight 'hump' in the upper back."
        ],
        "focusAnatomy": ["Anterior Deltoids", "Serratus Anterior", "Core Stabilizers"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Hips slightly above or below shoulder height" },
            { "penalty": 0.3, "reason": "Bent knees or touching the floor" },
            { "penalty": 0.5, "reason": "Bent arms (pumping the planche)" }
        ],
        "technicalCue": "Lean forward until your center of gravity is over your wrists and push the floor away."
    },

    "Handstand": {
        "name": "Handstand (Straight)",
        "category": "Statics",
        "difficulty": "A",
        "elementGroup": 1,
        "description": "A fundamental inverted position showing a perfectly straight line from wrists to toes.",
        "technicalRequirements": [
            "180° open shoulder angle (no 'closing' of the chest).",
            "Posterior pelvic tilt (hollow body).",
            "Head in a neutral position, ears hidden by shoulders."
        ],
        "focusAnatomy": ["Deltoids", "Trapezius", "Transverse Abdominis"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Arched back (banana back) or closed shoulders" },
            { "penalty": 0.1, "reason": "Separated legs or unpointed toes" },
            { "penalty": 0.3, "reason": "Walking on hands to maintain balance" }
        ],
        "technicalCue": "Squeeze your glutes and push through your fingertips to control the balance."
    },

    "Arabesque": {
        "name": "Arabesque",
        "category": "Flexibility",
        "difficulty": "A",
        "elementGroup": 3,
        "description": "A balance on one leg with the other leg extended backward at or above 90°.",
        "technicalRequirements": [
            "Supporting leg must be fully extended (locked knee).",
            "Raised leg must be at least 90° from the vertical axis.",
            "Upper body should remain as upright as possible."
        ],
        "focusAnatomy": ["Gluteus Maximus", "Hamstrings", "Erector Spinae"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Leg height below 90°" },
            { "penalty": 0.1, "reason": "Soft knee on supporting leg" },
            { "penalty": 0.3, "reason": "Significant torso drop to compensate for leg height" }
        ],
        "technicalCue": "Lengthen the spine and lift the back leg from the inner thigh, not just the foot."
    },

    "Maltese": {
        "name": "Maltese",
        "category": "Strength",
        "difficulty": "D",
        "elementGroup": 2,
        "description": "An elite strength hold on rings or floor where the body is held horizontal at the same level as the rings/hands.",
        "technicalRequirements": [
            "Body must be level with the hands.",
            "Arms extended wide, body in a 'cross' but horizontal.",
            "Minimum 2-second hold."
        ],
        "focusAnatomy": ["Pectoralis Major", "Biceps Brachii (Long Head)", "Core"],
        "commonDeductions": [
            { "penalty": 0.3, "reason": "Hips sagging below the line of the shoulders" },
            { "penalty": 0.3, "reason": "Excessive elbow bend" },
            { "penalty": 0.5, "reason": "Touching the rings with the body (on rings)" }
        ],
        "technicalCue": "Aggressively protract the shoulders and 'hug' the air beneath you."
    }
}
