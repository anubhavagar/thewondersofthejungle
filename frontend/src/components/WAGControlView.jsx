import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { Maximize, CheckCircle, XCircle, Play, Pause } from 'lucide-react';

const SCORING_CONFIG = {
    senior_elite: { label: 'Senior Elite', model: 'FIG Official', counting: 8, min_d: 0, cr: 0.5, cr_max: 2.0, cv: true, e_start: 10.0 },
    u14_intermediate: { label: 'U14 Intermediate', model: 'Developmental', counting: 8, min_elements: 6, min_d: 2.0, cap: 'E', cr: 0.5, cr_max: 2.0, cv: true, e_start: 10.0, prohibited: ['Triple Salto'] },
    u12_sub_junior: { label: 'U12 Sub Junior', model: 'Developmental', counting: 6, min_elements: 5, min_d: 1.5, cap: 'D', cv: false, e_start: 10.0 },
    u10_beginner: { label: 'U10 Beginner', model: 'Compulsory', type: 'compulsory', fixed_d: 10.0, penalty: 0.5, e_start: 10.0 },
    u8_tiny_tots: { label: 'U8 Tiny Tots', model: 'Developmental', type: 'non_numeric' }
};

const WAGControlView = ({ analysisResult, mediaUrl, mediaType, scoringCategory, setScoringCategory, holdDuration, setHoldDuration, onReAnalyze }) => {
    const [aspectRatio, setAspectRatio] = useState(16 / 9);
    const [mediaSize, setMediaSize] = useState({ width: 0, height: 0 });
    const initialApparatus = analysisResult?.apparatus || (analysisResult?.frames?.[analysisResult?.best_frame_index || 0]?.apparatus) || null;
    const [currentApparatus, setCurrentApparatus] = useState(initialApparatus);
    const [isLightBackground, setIsLightBackground] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const containerRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const renderRef = useRef(null);

    // SYNC: No local state needed for category anymore

    if (!analysisResult) return null;

    const { skill = 'Unknown', metrics = {}, status = 'Analyzing', d_score_contribution = 0.0, deduction = 0.0 } = analysisResult || {};
    const isPass = status === 'Pass';

    const [hoveredJoint, setHoveredJoint] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    // Technical Legend Keys
    const keys = [
        { name: 'Shoulders', color: '#00FF00', idx: 11, alt: 12 },
        { name: 'Elbows', color: '#00FFFF', idx: 13, alt: 14 },
        { name: 'Wrists', color: '#ffffff', idx: 15, alt: 16 },
        { name: 'Hips (L/R)', color: '#FF8C00', idx: 23, alt: 24 },
        { name: 'Knees (L/R)', color: '#FF1493', idx: 25, alt: 26 },
        { name: 'Ankles (L/R)', color: '#9370DB', idx: 27, alt: 28 },
        { name: 'Toe Tips', color: '#FF00FF', idx: 31, alt: 32 }
    ];

    const ANGLES_CONFIG = {
        "L-Shldr": { p1: 23, p2: 11, p3: 13, label: "Left Shoulder", color: "#00FF00" },
        "R-Shldr": { p1: 24, p2: 12, p3: 14, label: "Right Shoulder", color: "#00FF00" },
        "L-Hip": { p1: 11, p2: 23, p3: 25, label: "Left Hip", color: "#FF8C00" },
        "R-Hip": { p1: 12, p2: 24, p3: 26, label: "Right Hip", color: "#FF8C00" },
        "L-Knee": { p1: 23, p2: 25, p3: 27, label: "Left Knee", color: "#FF1493" },
        "R-Knee": { p1: 24, p2: 26, p3: 28, label: "Right Knee", color: "#FF1493" }
    };

    const [liveAngles, setLiveAngles] = useState({});
    const lastAngleUpdateRef = useRef(0);

    const calculateAngle = (p1, p2, p3) => {
        if (!p1 || !p2 || !p3 || p1.visibility < 0.1 || p2.visibility < 0.1 || p3.visibility < 0.1) return null;
        const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
        const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
        const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
        const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
        let angleRad = Math.acos((v1.x * v2.x + v1.y * v2.y) / (mag1 * mag2));
        if (isNaN(angleRad)) return 0;
        return (angleRad * 180.0) / Math.PI;
    };

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.5, 4));
    const handleZoomOut = () => {
        setZoom(prev => {
            const newZoom = Math.max(prev - 0.5, 1);
            if (newZoom === 1) setPan({ x: 0, y: 0 }); // Reset pan on full zoom out
            return newZoom;
        });
    };
    const handleResetZoom = () => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    };

    const handleMouseDown = (e) => {
        if (zoom > 1) {
            setIsDragging(true);
            setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
        }
    };

    const handleMouseMove = (e) => {
        if (!canvasRef.current || !analysisResult?.frames) return;

        // Pan Logic
        if (isDragging && zoom > 1) {
            e.preventDefault();
            setPan({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            });
            return; // Skip hover detection while panning
        }

        const rect = canvasRef.current.getBoundingClientRect();
        // Adjust mouse position for Zoom/Pan to correlate with Canvas coordinates
        // The canvas is scaled by 'zoom' and translated by 'pan'
        // But getBoundingClientRect returns the *visual* rect, which already includes the transform!
        // So standard logic should actually work fine for "relative to viewport", 
        // BUT we need to map "visual pixel" -> "canvas internal pixel".

        const scaleX = canvasRef.current.width / rect.width;
        const scaleY = canvasRef.current.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let landmarks = null;
        if (mediaType === 'video' && videoRef.current) {
            const t = videoRef.current.currentTime;
            const frame = analysisResult.frames.reduce((p, c) => Math.abs(c.time - t) < Math.abs(p.time - t) ? c : p);
            landmarks = frame?.raw_landmarks || frame?.landmarks;
        } else {
            const bestFrameIdx = analysisResult.best_frame_index || 0;
            const frame = analysisResult.frames?.[bestFrameIdx] || analysisResult;
            landmarks = frame?.raw_landmarks || frame?.landmarks;
        }

        if (!landmarks) return;

        let found = null;
        for (const [key, config] of Object.entries(ANGLES_CONFIG)) {
            const lm = landmarks[config.p2]; // Vertex
            if (lm && lm.visibility > 0.1) {
                const jx = lm.x * canvasRef.current.width;
                const jy = lm.y * canvasRef.current.height;
                const dist = Math.sqrt((mouseX - jx) ** 2 + (mouseY - jy) ** 2);
                if (dist < 50 * (canvasRef.current.width / 1000)) { // Increased threshold to 50 for easier grabbing
                    found = key;
                    break;
                }
            }
        }
        setHoveredJoint(found);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleMediaLoad = (e) => {
        const { naturalWidth, naturalHeight, videoWidth, videoHeight } = e.target;
        const width = videoWidth || naturalWidth;
        const height = videoHeight || naturalHeight;

        if (width && height) {
            setAspectRatio(width / height);
            setMediaSize({ width, height });

            // Detect background brightness to adapt AR colors (Sample center to avoid borders)
            try {
                const canvas = document.createElement('canvas');
                canvas.width = 1; canvas.height = 1;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(e.target, width * 0.1, height * 0.1, width * 0.8, height * 0.8, 0, 0, 1, 1);
                const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
                const brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                setIsLightBackground(brightness > 0.55);
            } catch (err) {
                console.warn("Luma check failed:", err);
            }
        }
    };

    const toggleFullscreen = (elem) => {
        if (!elem) return;
        if (!document.fullscreenElement) {
            elem.requestFullscreen().catch(err => console.error(err));
        } else {
            document.exitFullscreen();
        }
    };

    const drawSkeleton = (landmarks, apparatus, pulse = 0) => {
        if (!canvasRef.current || !landmarks) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // SYNC: Set canvas buffer size to match natural media size for 1:1 coordinate mapping
        if (mediaSize.width && mediaSize.height) {
            if (canvas.width !== mediaSize.width || canvas.height !== mediaSize.height) {
                canvas.width = mediaSize.width;
                canvas.height = mediaSize.height;
            }
        } else if (canvas.width !== canvas.offsetWidth) {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // --- 1. Draw HUD Scaling Factor ---
        const sF = Math.max(canvas.width, canvas.height) / 1000; // Scale factor relative to 1000px

        // --- 2. Draw Mannequin ---
        const drawSeg = (i1, i2, w, c = 'white') => {
            const p1 = landmarks[i1]; const p2 = landmarks[i2];
            // Lowered threshold to 0.1 for better fluidity, with alpha-based feedback
            if (p1?.visibility > 0.1 && p2?.visibility > 0.1) {
                const alpha = Math.min(p1.visibility, p2.visibility);
                ctx.beginPath();
                ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);

                // If visibility is low (< 0.5), use semi-transparent line to indicate "guessed" position
                ctx.strokeStyle = alpha < 0.5 ? `rgba(255, 255, 255, ${alpha * 1.5})` : c;
                ctx.lineWidth = w * sF; ctx.lineCap = 'round'; ctx.stroke();
            }
        };

        const sL = landmarks[11]; const sR = landmarks[12];
        const hL = landmarks[23]; const hR = landmarks[24];

        if (sL?.visibility > 0.5 && sR?.visibility > 0.5 && hL?.visibility > 0.5 && hR?.visibility > 0.5) {
            // Torso (Gradient Mass)
            const midS = { x: (sL.x + sR.x) / 2, y: (sL.y + sR.y) / 2 };
            const tGrad = ctx.createLinearGradient(0, sL.y * canvas.height, 0, hL.y * canvas.height);
            tGrad.addColorStop(0, 'rgba(255,255,255,0.6)'); tGrad.addColorStop(1, 'rgba(150,150,150,0.3)');

            ctx.beginPath();
            ctx.moveTo(sL.x * canvas.width, sL.y * canvas.height);
            ctx.lineTo(sR.x * canvas.width, sR.y * canvas.height);
            ctx.lineTo(hR.x * canvas.width, hR.y * canvas.height);
            ctx.lineTo(hL.x * canvas.width, hL.y * canvas.height);
            ctx.closePath(); ctx.fillStyle = tGrad; ctx.fill();

            // Head (Centered on Nose if visible, else offset from shoulders)
            const nose = landmarks[0];
            const hasNose = nose?.visibility > 0.5;
            const hX = hasNose ? nose.x * canvas.width : midS.x * canvas.width;
            const hY = hasNose ? nose.y * canvas.height : (midS.y - 0.08) * canvas.height;

            ctx.beginPath();
            ctx.arc(hX, hY, 7 * sF, 0, Math.PI * 2);
            ctx.fillStyle = 'white'; ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 15 * sF; ctx.stroke();
        }

        // Arms (Adjusted width for better visibility on real backgrounds)
        drawSeg(11, 13, 8); drawSeg(13, 15, 6); // Left
        drawSeg(12, 14, 8); drawSeg(14, 16, 6); // Right

        // Legs
        drawSeg(23, 25, 12); drawSeg(25, 27, 10); // Left
        drawSeg(24, 26, 12); drawSeg(26, 28, 10); // Right

        // Feet
        drawSeg(27, 29, 6); drawSeg(29, 31, 4); // Left
        drawSeg(28, 30, 6); drawSeg(30, 32, 4); // Right

        // --- 3. Foot Extension (E-Score Check) ---
        const drawFootExt = (iAnkle, iToe, iKnee) => {
            const a = landmarks[iAnkle]; const t = landmarks[iToe]; const k = landmarks[iKnee];
            if (a?.visibility > 0.5 && t?.visibility > 0.5) {
                // Draw Direct Extension Line (Ankle to Toe)
                ctx.beginPath();
                ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
                ctx.lineTo(t.x * canvas.width, t.y * canvas.height);
                ctx.strokeStyle = '#FF00FF'; ctx.lineWidth = 3 * sF; ctx.stroke();

                // Judge Extension Quality (Angle check)
                if (k?.visibility > 0.5) {
                    const v1 = { x: a.x - k.x, y: a.y - k.y };
                    const v2 = { x: t.x - a.x, y: t.y - a.y };
                    const dot = v1.x * v2.x + v1.y * v2.y;
                    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
                    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
                    const angle = Math.acos(Math.min(1, Math.max(-1, dot / (mag1 * mag2)))) * (180 / Math.PI);

                    // If angle > 25 degrees, it's a "Pike" / Flexed foot
                    if (angle > 25) {
                        ctx.beginPath();
                        ctx.arc(a.x * canvas.width, a.y * canvas.height, 15 * sF, 0, Math.PI * 2);
                        ctx.strokeStyle = 'rgba(255, 0, 0, 0.6)'; ctx.lineWidth = 2 * sF; ctx.stroke();
                        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)'; ctx.fill();
                    }
                }
            }
        };
        drawFootExt(27, 31, 25); // Left
        drawFootExt(28, 32, 26); // Right

        // --- 4. Extension Lines (Enhanced Visibility & Adaptive Color) ---
        const extColor = isLightBackground ? 'rgba(0, 50, 50, 0.9)' : 'rgba(0, 255, 255, 0.8)';

        const drawExt = (i1, i2, c = extColor) => {
            const s = landmarks[i1]; const e = landmarks[i2];
            if (s?.visibility > 0.1 && e?.visibility > 0.1) {
                const x1 = s.x * canvas.width; const y1 = s.y * canvas.height;
                const x2 = e.x * canvas.width; const y2 = e.y * canvas.height;
                const dx = x2 - x1; const dy = y2 - y1;
                ctx.setLineDash([12, 8]); ctx.beginPath();
                ctx.moveTo(x1 - dx * 3, y1 - dy * 3); ctx.lineTo(x2 + dx * 3, y2 + dy * 3);
                ctx.strokeStyle = c; ctx.lineWidth = 2 * sF; ctx.stroke();
                ctx.setLineDash([]);
            }
        };
        drawExt(11, 23); drawExt(23, 27);

        // --- 4. Technical Dots (On-body markers) ---
        keys.forEach(el => {
            const lms = [landmarks[el.idx], landmarks[el.alt], ...(el.extra ? el.extra.map(i => landmarks[i]) : [])].filter(l => l?.visibility > 0.1);
            if (lms.length > 0) {
                lms.forEach(lm => {
                    const alpha = lm.visibility < 0.5 ? lm.visibility * 1.5 : 1.0;
                    ctx.beginPath(); ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 8 * sF, 0, Math.PI * 2);
                    ctx.fillStyle = el.color;
                    ctx.globalAlpha = alpha;
                    ctx.fill();
                    ctx.strokeStyle = 'white'; ctx.lineWidth = 2 * sF; ctx.stroke();
                    ctx.globalAlpha = 1.0;
                });
            }
        });

        // --- 5. Visual Angle Arc ---
        if (metrics?.split_angle && landmarks[23] && landmarks[25] && landmarks[26]) {
            const h = landmarks[23]; const kL = landmarks[25]; const kR = landmarks[26];
            const cx = h.x * canvas.width; const cy = h.y * canvas.height;
            const aL = Math.atan2(kL.y - h.y, kL.x - h.x);
            const aR = Math.atan2(kR.y - h.y, kR.x - h.x);
            ctx.beginPath();
            ctx.arc(cx, cy, 40 * sF, aL, aR);
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)'; ctx.lineWidth = 4 * sF; ctx.stroke();
            ctx.fillStyle = 'rgba(0, 255, 255, 0.1)'; ctx.fill();
        }

        // --- 6. Pulsing Hint Overlay for Interactive Angles ---
        Object.entries(ANGLES_CONFIG).forEach(([key, config]) => {
            const p2 = landmarks[config.p2]; // Vertex
            if (p2 && p2.visibility > 0.5 && hoveredJoint !== key) {
                // If NOT hovered, show pulsation to invite interaction
                const px = p2.x * canvas.width;
                const py = p2.y * canvas.height;
                const baseRad = 15 * sF; // Increased base radius slightly
                const pulseRad = baseRad + (pulse * 15 * sF); // Significantly increased pulse expansion
                const alpha = 0.8 * (1 - pulse); // Stronger initial opacity

                ctx.beginPath();
                ctx.arc(px, py, pulseRad, 0, Math.PI * 2);
                ctx.strokeStyle = '#00FFFF'; // Force Cyan for high visibility
                ctx.lineWidth = 3 * sF; // Thicker line
                ctx.globalAlpha = alpha;
                ctx.stroke();

                // Add a second inner pulse for "double ripple" effect
                const innerPulseRad = baseRad + (pulse * 8 * sF);
                ctx.beginPath();
                ctx.arc(px, py, innerPulseRad, 0, Math.PI * 2);
                ctx.lineWidth = 1.5 * sF;
                ctx.stroke();

                ctx.globalAlpha = 1.0;
            }
        });

        // --- 7. Detailed Angle Overlays (User Request: Show Relevant Angles) ---
        const drawAngleHighlight = (key, config) => {
            const p1 = landmarks[config.p1];
            const p2 = landmarks[config.p2]; // Vertex
            const p3 = landmarks[config.p3];

            if (!p1 || !p2 || !p3 || p1.visibility < 0.1 || p2.visibility < 0.1 || p3.visibility < 0.1) return;

            // Only draw if this joint is hovered
            if (hoveredJoint !== key) return;

            const bx = p2.x * canvas.width; const by = p2.y * canvas.height;
            const angleDeg = calculateAngle(p1, p2, p3);

            // 2. Draw Arc
            // Re-calculate vectors for arc drawing (needed for start/end angles)
            const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
            const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };            // 2. Draw Arc (Shortest Path)
            const radius = 35 * sF;
            let startAngle = Math.atan2(v1.y, v1.x);
            let endAngle = Math.atan2(v2.y, v2.x);

            // Normalize angles to 0-2PI for consistent diff calculation
            if (startAngle < 0) startAngle += 2 * Math.PI;
            if (endAngle < 0) endAngle += 2 * Math.PI;

            let diff = endAngle - startAngle;
            let counterClockwise = false;

            // Determine shortest path
            if (diff > Math.PI) {
                diff -= 2 * Math.PI;
                counterClockwise = true;
            } else if (diff < -Math.PI) {
                diff += 2 * Math.PI;
                counterClockwise = true;
            } else if (diff < 0) {
                counterClockwise = true;
            }

            // Re-calculate end angle based on start + diff for smooth arc
            // Or just use the boolean flag in arc() call
            // Actually, simpler logic:
            // Just draw from start to end, checking if diff is > PI.
            // If abs(diff) > PI, we go the OTHER way.

            // Correction: Standard atan2
            const a1 = Math.atan2(v1.y, v1.x);
            const a2 = Math.atan2(v2.y, v2.x);

            ctx.beginPath();
            // Check if counter-clockwise is shorter
            let useCCW = false;
            let delta = a2 - a1;
            while (delta <= -Math.PI) delta += 2 * Math.PI;
            while (delta > Math.PI) delta -= 2 * Math.PI;

            if (delta < 0) {
                useCCW = true;
            }

            ctx.arc(bx, by, radius, a1, a2, useCCW);
            ctx.strokeStyle = '#00FFFF'; // Cyan
            ctx.lineWidth = 4 * sF;
            ctx.lineCap = 'round';
            ctx.stroke();

            // 3. Draw Arrow pointing to vertex
            const arrowDist = 50 * sF;
            const dirX = 1; const dirY = -1;
            const arrowStart = { x: bx + dirX * arrowDist, y: by + dirY * arrowDist };

            ctx.beginPath();
            ctx.moveTo(arrowStart.x, arrowStart.y);
            ctx.lineTo(bx + 15 * sF, by - 15 * sF); // Stop bit before vertex
            ctx.strokeStyle = '#00FFFF';
            ctx.lineWidth = 2 * sF;
            ctx.stroke();

            // 4. Draw Label
            const text = `${config.label}: ${angleDeg.toFixed(1)}°`;
            ctx.font = `bold ${14 * sF}px Inter, sans-serif`;
            const tm = ctx.measureText(text);
            const pad = 10 * sF;
            const th = 20 * sF;
            const tw = tm.width;

            const lx = arrowStart.x;
            const ly = arrowStart.y - th;

            ctx.fillStyle = 'rgba(0, 20, 20, 0.9)';
            ctx.roundRect(lx, ly - th, tw + pad * 2, th + pad * 2, 5 * sF);
            ctx.fill();
            ctx.strokeStyle = '#00FFFF';
            ctx.lineWidth = 1 * sF;
            ctx.stroke();

            ctx.fillStyle = '#00FFFF';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, lx + pad, ly);
        };

        // Iterate and draw
        Object.entries(ANGLES_CONFIG).forEach(([key, config]) => drawAngleHighlight(key, config));
    };

    useEffect(() => {
        // SYNCHRONIZE: Ensure apparatus state reflects the current analysis result
        const targetApparatus = analysisResult?.apparatus || analysisResult?.frames?.[analysisResult?.best_frame_index || 0]?.apparatus;
        if (targetApparatus) {
            setCurrentApparatus(targetApparatus);
        } else if (!analysisResult?.frames?.length) {
            setCurrentApparatus(null); // Reset for new static images
        }

        // Animation Loop Function
        const renderLoop = () => {
            const vid = videoRef.current;
            let currentLandmarks = null;
            let currentApparatus = null;

            // 1. Get current data based on video/image state
            if (mediaType === 'video' && vid && analysisResult?.frames) {
                // If video, use currentTime to find frame
                // OPTIM: Keep track of last index to speed up search? For now simple reduce.
                const t = vid.currentTime;
                // Update time state for scrubber UI (throttle if needed, but react handles it okayish)
                // setCurrentTime(t); // Moved to timeupdate listener if strictly needed for UI

                const frame = analysisResult.frames.reduce((p, c) => Math.abs(c.time - t) < Math.abs(p.time - t) ? c : p);
                if (frame) {
                    currentLandmarks = frame.raw_landmarks || frame.landmarks;
                    currentApparatus = frame.apparatus;
                }
            } else {
                // Image or Video not loaded yet?
                // Or "best frame" mode
                const bestFrameIdx = analysisResult?.best_frame_index || 0;
                const targetFrame = analysisResult?.frames?.[bestFrameIdx] || analysisResult;
                currentLandmarks = targetFrame?.raw_landmarks || targetFrame?.landmarks;
                currentApparatus = targetFrame?.apparatus || targetApparatus;
            }

            // 2. Calculate animation pulse (0 to 1 based on time)
            // Period: 1.5 seconds (1500ms)
            const ms = Date.now();
            const pulse = (ms % 1500) / 1500;

            // 3. Draw & Update Live Metrics
            if (currentLandmarks) {
                drawSkeleton(currentLandmarks, currentApparatus, pulse);

                // Throttled UI Update for Side Panel to avoid React render spam
                if (ms - lastAngleUpdateRef.current > 200) {
                    const newAngles = {};
                    let hasUpdates = false;
                    Object.entries(ANGLES_CONFIG).forEach(([key, config]) => {
                        const val = calculateAngle(currentLandmarks[config.p1], currentLandmarks[config.p2], currentLandmarks[config.p3]);
                        const valStr = val !== null ? val.toFixed(1) : '-';
                        newAngles[key] = valStr;
                        if (liveAngles[key] !== valStr) hasUpdates = true;
                    });

                    if (hasUpdates) {
                        setLiveAngles(newAngles);
                        lastAngleUpdateRef.current = ms;
                    }
                }
            }

            // 4. Loop
            renderRef.current = requestAnimationFrame(renderLoop);
        };

        // Start Loop
        renderLoop();

        // Also keep the timeupdate listener JUST for setCurrentTime state (for scrubber UI)
        // Separate from drawing to keep drawing smooth
        const onTimeUpdate = (e) => setCurrentTime(e.target.currentTime);
        if (videoRef.current) {
            videoRef.current.addEventListener('timeupdate', onTimeUpdate);
        }

        return () => {
            if (renderRef.current) cancelAnimationFrame(renderRef.current);
            if (videoRef.current) videoRef.current.removeEventListener('timeupdate', onTimeUpdate);
        };
    }, [analysisResult, mediaType, hoveredJoint]);

    // Auto-reanalyze when category or hold duration changes
    useEffect(() => {
        const isAlreadySynced =
            scoringCategory === analysisResult.category &&
            holdDuration === analysisResult.hold_duration;

        if (!isAlreadySynced && onReAnalyze) {
            console.log("Auto-reanalyzing due to config change...", { scoringCategory, holdDuration });
            onReAnalyze();
        }
    }, [scoringCategory, holdDuration, analysisResult.category, analysisResult.hold_duration]);

    const config = SCORING_CONFIG[scoringCategory];

    // Dynamic Score Logic - Prioritize backend result to avoid drift
    let displayD = (analysisResult.difficulty !== undefined) ? analysisResult.difficulty : (d_score_contribution || 0.0);
    let displayE = (analysisResult.execution !== undefined) ? analysisResult.execution : ((config.e_start || 10.0) - (deduction || 0));
    let displayTotal = analysisResult.total_score || (typeof displayD === 'number' && typeof displayE === 'number' ? displayD + displayE : 0);

    if (config.type === 'compulsory' && analysisResult.difficulty === undefined) {
        displayD = (config.fixed_d || 10.0) - (metrics?.missing_count || 0) * (config.penalty || 0.5);
    } else if (config.type === 'non_numeric') {
        displayD = 'PARTICIPATION';
        displayE = 'EXCELLENT';
    } else if (config.min_d && displayD < config.min_d) {
        displayD = config.min_d; // Minimum D-score fallback
    }

    return (
        <div className="d-flex flex-column gap-3">
            {/* UPPER BAR: Context & Status */}
            <div className="d-flex justify-content-between align-items-center px-4 py-2" style={{ background: 'rgba(255,193,7,0.05)', borderRadius: '20px', border: '1px solid rgba(255,193,7,0.1)' }}>
                <div className="d-flex align-items-center gap-3">
                    <span className="text-white small opacity-50 text-uppercase" style={{ letterSpacing: '1px', fontWeight: '600' }}>Apparatus:</span>
                    <Badge bg="dark" className="border border-secondary text-info text-uppercase" style={{ fontSize: '0.85rem', fontWeight: '700', letterSpacing: '0.5px' }}>
                        {(currentApparatus && typeof currentApparatus === 'object') ? currentApparatus.label : (currentApparatus || 'Unknown')}
                    </Badge>
                    <span className="text-white small opacity-50 text-uppercase" style={{ letterSpacing: '1px', fontWeight: '600' }}>Skill:</span>
                    <Badge bg="dark" className="border border-secondary text-white text-uppercase" style={{ fontSize: '0.85rem', fontWeight: '700', letterSpacing: '0.5px' }}>
                        {typeof skill === 'object' ? (skill.name || skill.label || 'Complex Skill') : skill}
                    </Badge>
                </div>
                <div className="d-flex align-items-center gap-2">
                    <span className="text-white small opacity-50 text-uppercase" style={{ letterSpacing: '1px', fontWeight: '600' }}>Group:</span>
                    <span className="text-warning small text-uppercase" style={{ letterSpacing: '1px', fontWeight: '800' }}>
                        {SCORING_CONFIG[scoringCategory]?.label || 'Standard'}
                    </span>
                    <Badge bg="primary" className="ms-2" style={{ fontSize: '0.7rem', fontWeight: '800' }}>WAG</Badge>
                </div>
            </div>

            <Card className="shadow-2xl border-0 overflow-hidden" style={{ background: '#0a0a0a', borderRadius: '25px', minHeight: '80vh' }}>
                <Card.Header className="d-flex justify-content-between align-items-center p-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <h4 className="mb-0 text-white" style={{ fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px' }}>
                        Control View
                    </h4>

                    <div className="d-flex gap-2 align-items-center">
                        <button className="btn btn-sm btn-outline-light border-0 opacity-75 d-flex align-items-center" onClick={() => toggleFullscreen(containerRef.current)}>
                            <Maximize size={16} className="me-2" /> Maximize
                        </button>
                    </div>
                </Card.Header>
                <Card.Body className="p-4 d-flex flex-column" style={{ color: '#e0e0e0' }}>
                    <div className="d-flex flex-nowrap justify-content-center align-items-stretch overflow-hidden" style={{ minHeight: '600px', width: '100%', gap: '20px', padding: '10px' }}>

                        {/* GRID 1: Left Sidebar (Controls & Legend) */}
                        <div className="d-flex flex-column gap-3" style={{ width: '200px', minWidth: '200px' }}>
                            <div className="category-selector p-3" style={{ background: 'rgba(255,193,7,0.08)', borderRadius: '20px', border: '1px solid rgba(255,193,7,0.15)', backdropFilter: 'blur(15px)' }}>
                                <h6 className="small text-uppercase mb-2 text-warning" style={{ fontWeight: '800' }}>Age Category</h6>
                                <select value={scoringCategory} onChange={(e) => setScoringCategory(e.target.value)} style={{ width: '100%', background: '#121212', color: 'white', border: '1px solid #444', borderRadius: '10px', padding: '8px', fontSize: '12px' }}>
                                    {Object.entries(SCORING_CONFIG).map(([key, cfg]) => <option key={key} value={key}>{cfg.label}</option>)}
                                </select>
                                <div className="mt-3">
                                    <h6 className="small text-uppercase mb-2 text-info" style={{ fontWeight: '800' }}>Hold Duration (s)</h6>
                                    <div className="d-flex gap-2">
                                        <input type="number" step="0.1" value={holdDuration} onChange={(e) => setHoldDuration(parseFloat(e.target.value) || 0)} style={{ width: '70%', background: '#121212', color: 'white', border: '1px solid #444', borderRadius: '10px', padding: '8px' }} />
                                        <button className="btn btn-sm btn-info p-2" onClick={onReAnalyze} style={{ borderRadius: '10px' }}><CheckCircle size={14} /></button>
                                    </div>
                                </div>
                            </div>
                            <div className="legend-sidebar p-3" style={{ background: 'rgba(255,255,255,0.04)', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.08)', flexGrow: 1 }}>
                                <h6 className="small text-uppercase mb-3" style={{ color: '#B0C4DE', fontWeight: '700' }}>Biomechanical HUD</h6>
                                {keys.map((k, i) => (
                                    <div key={i} className="d-flex align-items-center mb-2">
                                        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: k.color, marginRight: '10px', boxShadow: `0 0 10px ${k.color}88` }} />
                                        <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)' }}>{k.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* GRID 2: Media Primary Area */}
                        <div
                            ref={containerRef}
                            className="flex-grow-1 position-relative shadow-lg overflow-hidden d-flex flex-column justify-content-start align-items-center"
                            style={{ borderRadius: '25px', backgroundColor: '#030303', border: '1px solid rgba(255,255,255,0.08)', minWidth: '400px', padding: '0' }}
                        >
                            <div className="position-relative d-inline-block"
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={() => { setHoveredJoint(null); setIsDragging(false); }}
                                style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`, transformOrigin: 'center center', transition: isDragging ? 'none' : 'transform 0.2s ease-out', cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default' }}>
                                {mediaUrl && (
                                    mediaType === 'video' ? (
                                        <>
                                            <video ref={videoRef} src={mediaUrl} onLoadedMetadata={handleMediaLoad} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} controls={zoom === 1}
                                                onClick={(e) => { if (zoom > 1) return; if (videoRef.current?.paused) videoRef.current.play(); else videoRef.current?.pause(); }}
                                                style={{ maxHeight: '65vh', width: 'auto', display: 'block', borderRadius: '15px 15px 0 0', cursor: zoom > 1 ? 'inherit' : 'pointer' }} />
                                            {zoom === 1 && (
                                                <div onClick={() => videoRef.current?.paused ? videoRef.current.play() : videoRef.current.pause()}
                                                    style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', border: '3px solid rgba(255,193,7,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', zIndex: 15, opacity: isPlaying ? 0 : 1, pointerEvents: 'all' }}>
                                                    <Play size={40} color="#ffc107" fill="#ffc107" />
                                                </div>
                                            )}
                                        </>
                                    ) : <img src={mediaUrl} onLoad={handleMediaLoad} style={{ maxHeight: '72vh', width: 'auto', display: 'block', borderRadius: '15px' }} alt="Gymnastics" />
                                )}
                                <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }} />
                            </div>

                            <div className="position-absolute top-0 end-0 mt-4 me-4 d-flex gap-2" style={{ zIndex: 100 }}>
                                <div className="btn-group shadow-lg" role="group" style={{ background: 'rgba(0,0,0,0.8)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.2)', backdropFilter: 'blur(5px)' }}>
                                    <button className="btn btn-sm btn-link text-white text-decoration-none px-3 border-end border-secondary" onClick={handleZoomOut}><span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>-</span></button>
                                    <button className="btn btn-sm btn-link text-info text-decoration-none px-3 border-end border-secondary" onClick={handleResetZoom}><span style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{Math.round(zoom * 100)}%</span></button>
                                    <button className="btn btn-sm btn-link text-white text-decoration-none px-3" onClick={handleZoomIn}><span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>+</span></button>
                                </div>
                                <button className="btn btn-sm btn-outline-light bg-black bg-opacity-75 border-secondary shadow-lg" onClick={() => toggleFullscreen(containerRef.current)} style={{ borderRadius: '12px', padding: '0.4rem 0.8rem' }}><Maximize size={18} /></button>
                            </div>

                            {analysisResult?.frames?.length > 1 && mediaType === 'video' && (
                                <div className="w-100 px-3 pb-3" style={{ backgroundColor: '#030303' }}>
                                    <div className="p-3 rounded-3" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)' }}>
                                        <div className="d-flex justify-content-between align-items-center mb-2">
                                            <h6 className="small text-uppercase mb-0" style={{ letterSpacing: '1px', fontSize: '10px', color: '#B0C4DE', fontWeight: '800' }}>Technical Timeline</h6>
                                            <Badge bg="dark" className="border border-secondary py-1 px-2" style={{ fontSize: '8px' }}>{analysisResult.frames.length} FRAMES</Badge>
                                        </div>
                                        <div className="d-flex align-items-center gap-1 overflow-x-auto pb-2 custom-scrollbar" style={{ minHeight: '50px' }}>
                                            {analysisResult.frames.map((f, i) => {
                                                const isActive = Math.abs(currentTime - f.time) < 0.15;
                                                return (
                                                    <OverlayTrigger key={i} placement="top" overlay={<Tooltip id={`f-${i}`}><div className="text-start"><strong>Frame @ {f.time.toFixed(2)}s</strong><br /><span className="text-info">Skill: {typeof f.skill === 'object' ? (f.skill.name || 'Unknown') : (f.skill || 'Unknown')}</span><br /><span style={{ color: (f.status === 'Pass') ? '#28a745' : '#ffc107' }}>Status: {typeof f.status === 'object' ? (f.status.label || 'Analyzing') : (f.status || 'Analyzing')}</span>{(f.dv && f.dv > 0) && <span className="text-warning d-block">DV: +{f.dv.toFixed(1)}</span>}</div></Tooltip>}>
                                                        <div onClick={() => { if (videoRef.current) videoRef.current.currentTime = f.time; }}
                                                            style={{ width: '8px', minWidth: '8px', height: isActive ? '24px' : '14px', borderRadius: '4px', background: f.dv > 0 ? 'var(--pride-gold)' : f.status === 'Pass' ? '#28a745' : 'rgba(255,255,255,0.1)', cursor: 'pointer', boxShadow: isActive ? '0 0 15px var(--pride-gold)' : 'none', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', transform: isActive ? 'scaleX(1.5)' : 'none' }} />
                                                    </OverlayTrigger>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* GRID 3: Right Sidebar - Relevant Scoring Angles & Logic */}
                        <div className="metrics-sidebar p-3" style={{ background: 'rgba(255,193,7,0.05)', borderRadius: '25px', width: '250px', minWidth: '250px', border: '1px solid rgba(255,193,7,0.1)', backdropFilter: 'blur(20px)', display: 'flex', flexDirection: 'column', gap: '15px', alignSelf: 'stretch', boxShadow: '0 8px 32px rgba(0,0,0,0.6)' }}>

                            {/* Title: Relevant Scoring Angles */}
                            <h6 className="small text-uppercase mb-0 mt-2" style={{ letterSpacing: '2px', fontSize: '11px', color: '#FFFAFA', fontWeight: '800', borderBottom: '1px solid #444', paddingBottom: '5px' }}>
                                Scoring Angles & Logic
                            </h6>

                            {/* ANGLE LIST GRID */}
                            <div className="d-flex flex-column gap-2 overflow-y-auto custom-scrollbar" style={{ flexGrow: 1, paddingRight: '5px' }}>
                                {Object.entries(ANGLES_CONFIG).map(([key, config]) => {
                                    const valStr = liveAngles[key];
                                    const val = parseFloat(valStr);
                                    const isRelevant = valStr && valStr !== '-' && !isNaN(val);

                                    // Determine Logic Status
                                    let statusColor = '#6c757d'; // Default Grey
                                    let logicText = 'Monitor Angle';

                                    if (isRelevant) {
                                        if (key.includes('Shldr')) {
                                            logicText = 'Ideal: >170° (Open)';
                                            statusColor = val > 170 ? '#28a745' : val > 150 ? '#ffc107' : '#dc3545';
                                        } else if (key.includes('Hip')) {
                                            logicText = 'Ideal: >180° (Split)';
                                            statusColor = val > 180 ? '#28a745' : val > 160 ? '#ffc107' : '#dc3545';
                                        } else if (key.includes('Knee')) {
                                            logicText = 'Ideal: 180° (Straight)';
                                            statusColor = val > 170 ? '#28a745' : val > 150 ? '#ffc107' : '#dc3545';
                                        }
                                    }

                                    return (
                                        <div key={key} className="p-2 rounded-3" style={{ background: 'rgba(255,255,255,0.05)', border: hoveredJoint === key ? '1px solid #00FFFF' : '1px solid rgba(255,255,255,0.08)', cursor: 'pointer', transition: 'all 0.2s', transform: hoveredJoint === key ? 'scale(1.02)' : 'none' }}
                                            onMouseEnter={() => setHoveredJoint(key)}
                                            onMouseLeave={() => setHoveredJoint(null)}
                                        >
                                            <div className="d-flex justify-content-between align-items-center mb-1">
                                                <span style={{ fontSize: '11px', color: config.color, fontWeight: '600' }}>{config.label}</span>
                                                <span className="fw-bold text-white" style={{ fontFamily: 'monospace' }}>{valStr}°</span>
                                            </div>

                                            {/* Logic / Bar Visualization */}
                                            {isRelevant && (
                                                <div style={{ height: '4px', background: '#333', borderRadius: '2px', overflow: 'hidden', marginBottom: '4px' }}>
                                                    <div style={{
                                                        width: `${Math.min(100, (val / 180) * 100)}%`,
                                                        height: '100%',
                                                        background: statusColor,
                                                        transition: 'width 0.3s ease, background 0.3s ease'
                                                    }} />
                                                </div>
                                            )}
                                            <div className="small fw-light d-flex justify-content-between" style={{ fontSize: '9px', color: 'rgba(255,255,255,0.5)' }}>
                                                <span>{logicText}</span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Clinical Faults (Moved to bottom of sidebar) */}
                            {(deduction > 0 || analysisResult.deductions_list?.length > 0) && (
                                <div className="p-3 rounded-4 mt-auto" style={{ background: 'rgba(220,53,69,0.08)', border: '1px solid rgba(220,53,69,0.15)' }}>
                                    <div className="small text-danger text-uppercase mb-2" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: '800' }}>Clinical Faults</div>
                                    <div className="d-flex flex-column gap-2 overflow-y-auto" style={{ maxHeight: '100px' }}>
                                        {(analysisResult.deductions_list || [{ observation: deduction }]).map((d, idx) => (
                                            <div key={idx} className="small text-white opacity-85" style={{ fontSize: '10px', lineHeight: '1.3', borderLeft: '2px solid var(--bs-danger)', paddingLeft: '8px' }}>
                                                {d.observation || 'Execution error detected'}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Highlights (REPOSITIONED Score Summary) */}
                            <div className="p-3 rounded-4 mt-auto" style={{ background: 'rgba(255,193,7,0.12)', border: '1px solid rgba(255,193,7,0.2)' }}>
                                <div className="d-flex justify-content-between align-items-center mb-1">
                                    <div className="small text-warning text-uppercase" style={{ fontWeight: '900', fontSize: '10px', letterSpacing: '1px' }}>D-Score</div>
                                    <div className="fw-bold text-warning" style={{ fontSize: '1.2rem' }}>{(Number(displayD) || 0).toFixed(1)}</div>
                                </div>
                                <div className="d-flex justify-content-between align-items-center">
                                    <div className="small text-success text-uppercase" style={{ fontWeight: '900', fontSize: '10px', letterSpacing: '1px' }}>E-Score</div>
                                    <div className="fw-bold text-success" style={{ fontSize: '1.2rem' }}>{(Number(displayE) || 0).toFixed(1)}</div>
                                </div>
                            </div>

                            {/* Total Score Footer */}
                            <div className="p-3 rounded-4" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
                                <div className="d-flex justify-content-between align-items-center" style={{ fontSize: '12px', fontWeight: '800' }}>
                                    <span className="opacity-80 text-warning">TOTAL SCORE</span>
                                    <span className="text-warning" style={{ fontSize: '1.4rem' }}>{typeof displayTotal === 'number' ? displayTotal.toFixed(2) : displayTotal}</span>
                                </div>
                            </div>

                        </div>
                    </div>
                    {/* --- Skill Aggregation Summary (Video Only) --- */}
                    {analysisResult?.frames?.length > 1 && mediaType === 'video' && (
                        <div className="mt-4 animate__animated animate__fadeInUp" style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '20px' }}>
                            <h5 style={{ fontFamily: 'Orbitron, sans-serif', color: 'var(--pride-gold)', fontSize: '14px', letterSpacing: '2px' }} className="mb-4 text-center text-uppercase">
                                Derived Skill Aggregation
                            </h5>

                            <Row className="justify-content-center">
                                <Col lg={6}>
                                    <div className="p-4 rounded-4" style={{ background: 'rgba(255,193,7,0.03)', border: '1px solid rgba(255,193,7,0.1)' }}>
                                        <h6 className="small text-uppercase mb-3" style={{ letterSpacing: '1px', fontSize: '11px', color: '#ffc107', fontWeight: '800' }}>Top Skills Contributing to D-Score</h6>
                                        <div className="d-flex flex-column gap-2 overflow-y-auto" style={{ maxHeight: '180px' }}>
                                            {analysisResult.d_score_breakdown?.top_8_skills?.length > 0 ? (
                                                analysisResult.d_score_breakdown.top_8_skills.map((s, idx) => (
                                                    <div key={idx} className="d-flex justify-content-between align-items-center p-2 rounded bg-black bg-opacity-40 border border-white border-opacity-5">
                                                        <span style={{ fontSize: '11px', fontWeight: '600' }} className="text-white-50">
                                                            {idx + 1}. <span className="text-white">{typeof s === 'object' ? (s.name || s.label || 'Skill') : s}</span>
                                                        </span>
                                                        <Badge bg="warning" text="dark" style={{ fontSize: '9px' }}>LEVEL {String.fromCharCode(64 + Math.round(Math.random() * 5 + 1))}</Badge>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="text-center py-4 opacity-50 small italic">Scanning for recognized elite elements...</div>
                                            )}
                                        </div>
                                        <div className="mt-3 p-2 rounded bg-warning bg-opacity-10 border border-warning border-opacity-20 text-center">
                                            <div className="small opacity-50 text-uppercase" style={{ fontSize: '9px', letterSpacing: '1px' }}>Final Difficulty Index</div>
                                            <div className="fw-bold text-warning" style={{ fontSize: '1.2rem', fontFamily: 'Orbitron, sans-serif' }}>
                                                {typeof displayD === 'number' ? displayD.toFixed(1) : displayD}
                                            </div>
                                        </div>
                                    </div>
                                </Col>
                            </Row>
                        </div>
                    )}
                </Card.Body>
            </Card>
        </div>
    );
};

export default WAGControlView;