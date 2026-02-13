
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
            "Body must be perfectly horizontal (shoulders and hips level).",
            "Arms must be fully locked (no elbow flexion).",
            "Scapula protraction for core engagement."
        ],
        "focusAnatomy": ["Anterior Deltoids", "Serratus Anterior", "Core Stabilizers"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Hips slightly above or below shoulder height" },
            { "penalty": 0.3, "reason": "Bent knees or body not horizontal" },
            { "penalty": 0.3, "reason": "Bent arms" }
        ],
        "technicalCue": "Lean forward until your center of gravity is over your wrists and push the floor away."
    },

    "Press Handstand": {
        "name": "Press Handstand",
        "category": "Strength / Statics",
        "difficulty": "B",
        "elementGroup": 1,
        "description": "Moving from a support or planche position into a handstand using pure strength, without a kick.",
        "technicalRequirements": [
            "Controlled movement throughout the press.",
            "Arms must remain fully locked.",
            "Body must pass through a controlled horizontal or angled phase."
        ],
        "focusAnatomy": ["Deltoids", "Lower Traps", "Core", "Hip Flexors"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Bent arms during the press" },
            { "penalty": 0.1, "reason": "Loss of control / wobbling" },
            { "penalty": 0.3, "reason": "Using a kick or momentum (not a pure press)" }
        ],
        "technicalCue": "Shift your weight forward and use your shoulders to lift your hips over your head."
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
    },
    
    "Pommel Support": {
        "name": "Pommel Support",
        "category": "Statics",
        "difficulty": "A",
        "elementGroup": 1,
        "description": "A basic support position on the pommel horse.",
        "technicalRequirements": [
            "Arms must be locked.",
            "Body must maintain a straight line.",
            "Neutral head position."
        ],
        "focusAnatomy": ["Triceps", "Shoulders", "Core"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Bent elbows" },
            { "penalty": 0.1, "reason": "Hips sagging" }
        ],
        "technicalCue": "Keep your arms locked and look straight ahead."
    },
    
    "Pommel Flair/Circle": {
        "name": "Pommel Flair/Circle",
        "category": "Dynamic",
        "difficulty": "C",
        "elementGroup": 2,
        "description": "A circular movement of the legs around the horse.",
        "technicalRequirements": [
            "Lifting of the body during the transition.",
            "Wide leg spread (flair).",
            "Smooth rotation."
        ],
        "focusAnatomy": ["Core", "Shoulder Girdle", "Hip Flexors"],
        "commonDeductions": [
            { "penalty": 0.3, "reason": "Touching the horse with legs" },
            { "penalty": 0.1, "reason": "Insufficient height" }
        ],
        "technicalCue": "Drive your hips up as the legs sweep through."
    },
    
    "Straddle Split": {
        "name": "Straddle Split",
        "category": "Flexibility",
        "difficulty": "B",
        "elementGroup": 3,
        "description": "A split position with legs extended sideways, often performed as a jump or leap.",
        "technicalRequirements": [
            "Legs must be at or above 180°.",
            "Torso should remain upright or slightly leaned forward for balance.",
            "Feet must be pointed."
        ],
        "focusAnatomy": ["Adductors", "Hamstrings", "Hip Abductors"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Insufficient leg spread (< 180°)" },
            { "penalty": 0.1, "reason": "Soft knees" }
        ],
        "technicalCue": "Engage your glutes to rotate the hips and push the legs out wide."
    },

    "L-Sit": {
        "name": "L-Sit",
        "category": "Statics",
        "difficulty": "A",
        "elementGroup": 1,
        "description": "A support position where the legs are held horizontal, forming an 'L' shape with the torso.",
        "technicalRequirements": [
            "Legs must be perfectly horizontal (Horizonation).",
            "Arms must be fully locked (Straight Arms).",
            "Legs must be fully extended (Straight Legs).",
            "Toes must be pointed."
        ],
        "focusAnatomy": ["Core", "Hip Flexors", "Triceps", "Quads"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Bent arms/legs" },
            { "penalty": 0.3, "reason": "Legs below horizontal" },
            { "penalty": 0.1, "reason": "Pointed toes missing" }
        ],
        "technicalCue": "Depress your shoulders and compress your abs to lift the legs to horizontal."
    },
    
    "Straddle L-Sit": {
        "name": "Straddle L-Sit",
        "category": "Statics",
        "difficulty": "B",
        "elementGroup": 1,
        "description": "A support position on parallel bars or floor where legs are spread wide and held horizontal.",
        "technicalRequirements": [
            "Wide leg spread (usually > 90°).",
            "Legs must be at or above horizontal.",
            "Arms fully locked.",
            "Torso vertical."
        ],
        "focusAnatomy": ["Core", "Hip Abductors", "Triceps"],
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Insufficient leg spread" },
            { "penalty": 0.3, "reason": "Legs below horizontal" },
            { "penalty": 0.1, "reason": "Bent knees" }
        ],
        "technicalCue": "Push the bars away and pull your hips forward while spreading your legs wide."
    },

    "V-Sit": {
        "name": "V-Sit",
        "category": "Statics / Flexibility",
        "difficulty": "B",
        "elementGroup": 1,
        "description": "A high-level support position where the legs are held significantly above horizontal, forming a 'V' shape with the torso.",
        "technicalRequirements": [
            "Legs held at 45° above horizontal or higher.",
            "Straight knees and pointed toes.",
            "Hold for 2 seconds."
        ],
        "cv_thresholds": {
            "min_hip_angle": 45,
            "hold_frames": 60 
        },
        "commonDeductions": [
            { "penalty": 0.1, "reason": "Angle between 0° and 45°" },
            { "penalty": 0.3, "reason": "Bent knees" }
        ],
        "technicalCue": "Compress your core as hard as possible and pull your feet toward your face while keeping arms locked."
    }
}
