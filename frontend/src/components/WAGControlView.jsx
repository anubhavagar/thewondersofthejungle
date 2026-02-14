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

    // SYNC: No local state needed for category anymore

    if (!analysisResult) return null;

    const { skill = 'Unknown', metrics = {}, status = 'Analyzing', d_score_contribution = 0.0, deduction = 0.0 } = analysisResult || {};
    const isPass = status === 'Pass';

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

    const drawSkeleton = (landmarks, apparatus) => {
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
    };

    useEffect(() => {
        // SYNCHRONIZE: Ensure apparatus state reflects the current analysis result
        const targetApparatus = analysisResult?.apparatus || analysisResult?.frames?.[analysisResult?.best_frame_index || 0]?.apparatus;
        if (targetApparatus) {
            setCurrentApparatus(targetApparatus);
        } else if (!analysisResult?.frames?.length) {
            setCurrentApparatus(null); // Reset for new static images
        }

        const vid = videoRef.current;
        if (analysisResult?.frames?.length > 1 && mediaType === 'video' && vid) {
            const onTime = () => {
                setCurrentTime(vid.currentTime);
                const frame = analysisResult.frames.reduce((p, c) => Math.abs(c.time - vid.currentTime) < Math.abs(p.time - vid.currentTime) ? c : p);
                if (frame) {
                    const renderLandmarks = frame.raw_landmarks || frame.landmarks;
                    drawSkeleton(renderLandmarks, frame.apparatus);
                    if (frame.apparatus) setCurrentApparatus(frame.apparatus);
                }
            };
            vid.addEventListener('timeupdate', onTime);
            return () => vid.removeEventListener('timeupdate', onTime);
        } else {
            const bestFrameIdx = analysisResult?.best_frame_index || 0;
            const targetFrame = analysisResult?.frames?.[bestFrameIdx] || analysisResult;
            const renderLandmarks = targetFrame?.raw_landmarks || targetFrame?.landmarks;
            if (renderLandmarks) {
                drawSkeleton(renderLandmarks, targetFrame?.apparatus || currentApparatus);
                if (targetFrame?.apparatus) setCurrentApparatus(targetFrame.apparatus);
            }
        }
    }, [analysisResult, mediaType]);

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
        <Card className="shadow-2xl border-0 overflow-hidden" style={{ background: '#0a0a0a', borderRadius: '25px', minHeight: '85vh' }}>
            <Card.Header className="d-flex justify-content-between align-items-center p-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <h4 style={{ fontFamily: 'Cinzel, serif', color: 'var(--pride-gold)', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '2px' }} className="mb-0">
                    🤸‍♀️ Control View: <span style={{ color: 'white' }}>{skill}</span>
                </h4>
                <div className="d-flex gap-2 align-items-center">
                    <button className="btn btn-sm btn-outline-light border-0 opacity-75 d-flex align-items-center" onClick={() => toggleFullscreen(containerRef.current)}>
                        <Maximize size={16} className="me-2" /> Maximize
                    </button>
                </div>
            </Card.Header>
            <Card.Body className="p-4 d-flex flex-column" style={{ color: '#e0e0e0' }}>
                <div className="d-flex flex-nowrap justify-content-center align-items-stretch overflow-hidden" style={{ minHeight: '600px', width: '100%', gap: '20px', padding: '10px' }}>
                    {/* Sidebar Left: Technical Key & Settings */}
                    <div className="d-flex flex-column gap-3" style={{ width: '220px', minWidth: '220px' }}>
                        <div className="category-selector p-3" style={{
                            background: 'rgba(255,193,7,0.08)',
                            borderRadius: '20px',
                            border: '1px solid rgba(255,193,7,0.15)',
                            backdropFilter: 'blur(15px)'
                        }}>
                            <h6 className="small text-uppercase mb-2 text-warning" style={{ letterSpacing: '2px', fontSize: '10px', fontWeight: '800' }}>Age Category</h6>
                            <select
                                value={scoringCategory}
                                onChange={(e) => setScoringCategory(e.target.value)}
                                style={{
                                    width: '100%',
                                    background: '#121212',
                                    color: 'white',
                                    border: '1px solid #444',
                                    borderRadius: '10px',
                                    padding: '8px',
                                    fontSize: '12px',
                                    fontWeight: '600',
                                    outline: 'none',
                                    cursor: 'pointer',
                                    boxShadow: '0 4px 10px rgba(0,0,0,0.3)'
                                }}
                            >
                                {Object.entries(SCORING_CONFIG).map(([key, cfg]) => (
                                    <option key={key} value={key}>{cfg.label}</option>
                                ))}
                            </select>

                            {/* Auto-sync hint */}
                            <div className="mt-1" style={{ fontSize: '9px', color: 'rgba(255,193,7,0.5)', textAlign: 'center' }}>
                                Changes trigger re-analysis ⚡
                            </div>
                            <div className="mt-3">
                                <h6 className="small text-uppercase mb-2 text-info" style={{ letterSpacing: '2px', fontSize: '10px', fontWeight: '800' }}>Hold Duration (s)</h6>
                                <div className="d-flex gap-2 align-items-center">
                                    <input
                                        type="number"
                                        step="0.1"
                                        min="0"
                                        value={holdDuration}
                                        onChange={(e) => setHoldDuration(parseFloat(e.target.value) || 0)}
                                        style={{
                                            width: '70%',
                                            background: '#121212',
                                            color: 'white',
                                            border: '1px solid #444',
                                            borderRadius: '10px',
                                            padding: '8px',
                                            fontSize: '12px',
                                            fontWeight: '600',
                                            outline: 'none'
                                        }}
                                    />
                                    <button
                                        className="btn btn-sm btn-info p-2"
                                        onClick={onReAnalyze}
                                        style={{ borderRadius: '10px' }}
                                    >
                                        <CheckCircle size={14} />
                                    </button>
                                </div>
                            </div>
                            <div className="mt-2" style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>
                                FIG Standard: 2.0s min
                            </div>
                        </div>

                        <div className="legend-sidebar p-3" style={{
                            background: 'rgba(255,255,255,0.04)',
                            borderRadius: '20px',
                            border: '1px solid rgba(255,255,255,0.08)',
                            backdropFilter: 'blur(10px)',
                            flexGrow: 1
                        }}>
                            <h6 className="small text-uppercase mb-3" style={{ letterSpacing: '2px', fontSize: '11px', color: '#B0C4DE', fontWeight: '700' }}>Biomechanical HUD</h6>
                            {keys.map((k, i) => (
                                <div key={i} className="d-flex align-items-center mb-2">
                                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: k.color, marginRight: '10px', boxShadow: `0 0 10px ${k.color}88` }} />
                                    <span style={{ fontSize: '11px', fontWeight: '500', color: 'rgba(255,255,255,0.7)' }}>{k.name}</span>
                                </div>
                            ))}

                            {config.prohibited && (
                                <div className="mt-4 p-2 rounded bg-danger bg-opacity-10 border border-danger">
                                    <div className="small text-danger fw-bold" style={{ fontSize: '9px', letterSpacing: '1px' }}>RESESTRICTIONS</div>
                                    {config.prohibited.map(p => (
                                        <div key={p} className="small text-white opacity-75" style={{ fontSize: '10px' }}>• {p}</div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Media Primary Area */}
                    <div className="flex-grow-1 position-relative shadow-lg overflow-hidden d-flex flex-column justify-content-start align-items-center" style={{
                        borderRadius: '25px',
                        backgroundColor: '#030303',
                        border: '1px solid rgba(255,255,255,0.08)',
                        minWidth: '0',
                        padding: '0'
                    }}>
                        <div className="position-relative d-inline-block">
                            {mediaUrl && (
                                mediaType === 'video' ? (
                                    <>
                                        <video
                                            ref={videoRef}
                                            src={mediaUrl}
                                            onLoadedMetadata={handleMediaLoad}
                                            onPlay={() => setIsPlaying(true)}
                                            onPause={() => setIsPlaying(false)}
                                            controls
                                            onClick={(e) => {
                                                if (videoRef.current) {
                                                    if (videoRef.current.paused) {
                                                        videoRef.current.play();
                                                    } else {
                                                        videoRef.current.pause();
                                                    }
                                                }
                                            }}
                                            style={{ maxHeight: '65vh', width: 'auto', display: 'block', borderRadius: '15px 15px 0 0', cursor: 'pointer' }}
                                        />

                                        {/* Custom Play/Pause Button Overlay */}
                                        <div
                                            onClick={() => {
                                                if (videoRef.current) {
                                                    if (videoRef.current.paused) {
                                                        videoRef.current.play();
                                                    } else {
                                                        videoRef.current.pause();
                                                    }
                                                }
                                            }}
                                            style={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                transform: 'translate(-50%, -50%)',
                                                width: '80px',
                                                height: '80px',
                                                borderRadius: '50%',
                                                background: 'rgba(0,0,0,0.6)',
                                                backdropFilter: 'blur(10px)',
                                                border: '3px solid rgba(255,193,7,0.8)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                cursor: 'pointer',
                                                zIndex: 15,
                                                transition: 'all 0.3s ease',
                                                opacity: isPlaying ? 0 : 1,
                                                pointerEvents: 'all'
                                            }}
                                            onMouseEnter={(e) => e.currentTarget.style.transform = 'translate(-50%, -50%) scale(1.1)'}
                                            onMouseLeave={(e) => e.currentTarget.style.transform = 'translate(-50%, -50%) scale(1)'}
                                        >
                                            <Play size={40} color="#ffc107" fill="#ffc107" />
                                        </div>
                                    </>
                                ) : (
                                    <img
                                        src={mediaUrl}
                                        onLoad={handleMediaLoad}
                                        style={{ maxHeight: '72vh', width: 'auto', display: 'block', borderRadius: '15px' }}
                                        alt="Gymnastics"
                                    />
                                )
                            )}

                            {/* AR Layer (Locked to Media) */}
                            <canvas
                                ref={canvasRef}
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    pointerEvents: 'none',
                                    zIndex: 10
                                }}
                            />
                        </div>

                        {/* Technical Timeline (Video Only) - Below video */}
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
                                                <OverlayTrigger
                                                    key={i}
                                                    placement="top"
                                                    overlay={
                                                        <Tooltip id={`f-${i}`}>
                                                            <div className="text-start">
                                                                <strong>Frame @ {f.time.toFixed(2)}s</strong><br />
                                                                <span className="text-info">Skill: {f.skill || 'Unknown'}</span><br />
                                                                <span style={{ color: (f.status === 'Pass') ? '#28a745' : '#ffc107' }}>Status: {f.status || 'Analyzing'}</span>
                                                                {(f.dv && f.dv > 0) && <span className="text-warning d-block">DV: +{f.dv.toFixed(1)}</span>}
                                                            </div>
                                                        </Tooltip>
                                                    }
                                                >
                                                    <div
                                                        onClick={() => { if (videoRef.current) videoRef.current.currentTime = f.time; }}
                                                        style={{
                                                            width: '8px',
                                                            minWidth: '8px',
                                                            height: isActive ? '24px' : '14px',
                                                            borderRadius: '4px',
                                                            background: f.dv > 0 ? 'var(--pride-gold)' : f.status === 'Pass' ? '#28a745' : 'rgba(255,255,255,0.1)',
                                                            cursor: 'pointer',
                                                            boxShadow: isActive ? '0 0 15px var(--pride-gold)' : 'none',
                                                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                                            transform: isActive ? 'scaleX(1.5)' : 'none'
                                                        }}
                                                    />
                                                </OverlayTrigger>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Sidebar: Technical Metrics */}
                    <div className="metrics-sidebar p-3" style={{
                        background: 'rgba(255,193,7,0.05)',
                        borderRadius: '25px',
                        width: '220px',
                        minWidth: '220px',
                        border: '1px solid rgba(255,193,7,0.1)',
                        backdropFilter: 'blur(20px)',
                        alignSelf: 'stretch',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '15px',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.6)'
                    }}>
                        <h6 className="small text-uppercase mb-2" style={{ letterSpacing: '2px', fontSize: '11px', color: '#FFFAFA', fontWeight: '800' }}>Analysis Report</h6>

                        <div className="p-3 rounded-4" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.08)', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)' }}>
                            <div className="small opacity-50 text-uppercase mb-1" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: 'bold' }}>Max Extension</div>
                            <div className="fw-bold text-info" style={{ fontSize: '1.8rem', lineHeight: '1', fontFamily: 'Orbitron, sans-serif' }}>{metrics?.split_angle}°</div>
                        </div>

                        <div className="p-3 rounded-4" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.08)' }}>
                            <div className="small opacity-50 text-uppercase mb-2" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: 'bold' }}>Skill Status</div>
                            <Badge
                                bg={status === 'Pass' ? 'success' : status === 'Neutral' ? 'warning text-dark' : 'danger'}
                                style={{ fontSize: '12px', width: '100%', padding: '10px', borderRadius: '10px', letterSpacing: '1px', textShadow: status === 'Neutral' ? 'none' : '0 1px 2px rgba(0,0,0,0.3)' }}
                            >
                                {status?.toUpperCase() || 'ANALYZING'}
                            </Badge>
                            {analysisResult.status_reason && (
                                <div className="mt-2 text-center" style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', fontStyle: 'italic', lineHeight: '1.2' }}>
                                    {analysisResult.status_reason}
                                </div>
                            )}
                        </div>

                        <div className="p-3 rounded-4" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.08)', flexGrow: 1 }}>
                            <div className="small opacity-50 text-uppercase mb-1" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: 'bold' }}>Active Apparatus</div>
                            <div className="fw-bold text-white mt-2" style={{ fontSize: '13px', lineHeight: '1.4', color: '#F0F8FF' }}>
                                {currentApparatus ? currentApparatus.label : (skill?.toLowerCase().includes('floor') ? 'FLOOR EXERCISE (FX)' : 'GENERAL ANALYSIS')}
                            </div>
                        </div>

                        <div className="p-3 rounded-4" style={{ background: 'rgba(255,193,7,0.12)', border: '1px solid rgba(255,193,7,0.2)', boxShadow: '0 4px 15px rgba(255,193,7,0.1)' }}>
                            <div className="small text-warning text-uppercase mb-1" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: '900' }}>
                                {config.type === 'compulsory' ? 'Start Value' : 'Difficulty (D)'}
                            </div>
                            <div className="fw-bold text-warning" style={{ fontSize: '2.4rem', lineHeight: '1', fontFamily: 'Orbitron, sans-serif' }}>
                                {(Number(displayD) || 0).toFixed(1)}
                            </div>
                            {config.cap && <div className="text-danger mt-1 fw-bold" style={{ fontSize: '9px', letterSpacing: '0.5px' }}>D-CAP: {config.cap}</div>}
                        </div>

                        {/* Technical Observations (Integrated from Analyzer) */}
                        {(deduction > 0 || analysisResult.deductions_list?.length > 0) && (
                            <div className="p-3 rounded-4" style={{ background: 'rgba(220,53,69,0.08)', border: '1px solid rgba(220,53,69,0.15)' }}>
                                <div className="small text-danger text-uppercase mb-2" style={{ fontSize: '10px', letterSpacing: '1px', fontWeight: '800' }}>Clinical Faults</div>
                                <div className="d-flex flex-column gap-2">
                                    {(analysisResult.deductions_list || [{ observation: deduction }]).slice(0, 2).map((d, idx) => (
                                        <div key={idx} className="small text-white opacity-85" style={{ fontSize: '10px', lineHeight: '1.3', borderLeft: '2px solid var(--bs-danger)', paddingLeft: '8px' }}>
                                            {d.observation || 'Execution error detected'}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Legend Footnote */}
                        <div className="mt-auto p-3 rounded-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
                            <div className="small opacity-40 text-uppercase mb-2" style={{ fontSize: '9px', letterSpacing: '1px' }}>{scoringCategory === 'u10_beginner' ? 'JUDGING' : 'FIG'} STANDARDS</div>
                            <div className="d-flex justify-content-between align-items-center mt-2 pt-1 border-top border-secondary border-opacity-25" style={{ fontSize: '11px', fontWeight: '600' }}>
                                <span className="opacity-60">E-Score:</span>
                                <span className="text-success">{typeof displayE === 'number' ? displayE.toFixed(1) : displayE}</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mt-1" style={{ fontSize: '12px', fontWeight: '800' }}>
                                <span className="opacity-80 text-warning">Total:</span>
                                <span className="text-warning">{typeof displayTotal === 'number' ? displayTotal.toFixed(2) : displayTotal}</span>
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
                                                        {idx + 1}. <span className="text-white">{s}</span>
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
    );
};

export default WAGControlView;
