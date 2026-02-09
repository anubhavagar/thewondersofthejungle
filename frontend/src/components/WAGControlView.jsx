import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Badge } from 'react-bootstrap';
import { Maximize, CheckCircle, XCircle } from 'lucide-react';

const WAGControlView = ({ analysisResult, mediaUrl, mediaType }) => {
    const [aspectRatio, setAspectRatio] = useState(16 / 9);
    const [currentApparatus, setCurrentApparatus] = useState(analysisResult?.apparatus || null);

    const containerRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    if (!analysisResult) return null;

    const { skill, metrics, status, d_score_contribution, deduction } = analysisResult;
    const isPass = status === 'Pass';

    // Helpers
    const handleMediaLoad = (e) => {
        const { naturalWidth, naturalHeight, videoWidth, videoHeight } = e.target;
        const width = videoWidth || naturalWidth;
        const height = videoHeight || naturalHeight;
        if (width && height) setAspectRatio(width / height);
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

        if (canvas.width !== canvas.offsetWidth) {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // --- 1. Draw Environment ---
        const vanishingPoint = { x: canvas.width / 2, y: canvas.height * 0.45 };
        const floorY = canvas.height * 0.85;

        // Floor with Perspective Grid
        const floorWidthTop = canvas.width * 0.4;
        const p1 = { x: canvas.width / 2 - floorWidthTop / 2, y: vanishingPoint.y + 120 };
        const p2 = { x: canvas.width / 2 + floorWidthTop / 2, y: vanishingPoint.y + 120 };
        const grad = ctx.createLinearGradient(0, p1.y, 0, canvas.height);
        grad.addColorStop(0, '#102a43'); grad.addColorStop(1, '#081209');
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(canvas.width + 100, canvas.height); ctx.lineTo(-100, canvas.height);
        ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

        // Subtle Perspective Lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'; ctx.lineWidth = 1;
        for (let i = -5; i <= 5; i++) {
            ctx.beginPath();
            ctx.moveTo(canvas.width / 2 + i * 40, p1.y);
            ctx.lineTo(canvas.width / 2 + i * 200, canvas.height);
            ctx.stroke();
        }

        // (Phantom Apparatus logic removed as per user request)

        // YOLO Mask (Legacy) or MediaPipe BBox
        if (apparatus?.mask_polygon?.[0]) {
            ctx.save();
            ctx.beginPath();
            const points = apparatus.mask_polygon[0];
            ctx.moveTo(points[0][0] * canvas.width, points[0][1] * canvas.height);
            points.forEach(p => ctx.lineTo(p[0] * canvas.width, p[1] * canvas.height));
            ctx.closePath();
            ctx.shadowBlur = 15; ctx.shadowColor = 'rgba(0, 212, 255, 0.8)';
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.8)'; ctx.lineWidth = 3; ctx.stroke();
            ctx.fillStyle = 'rgba(0, 150, 255, 0.1)'; ctx.fill();
            ctx.restore();
        } else if (apparatus?.bbox) {
            ctx.save();
            const [bLeft, bTop, bWidth, bHeight] = apparatus.bbox;
            ctx.beginPath();
            // Use roundRect for a premium feel
            ctx.roundRect(
                bLeft * canvas.width,
                bTop * canvas.height,
                bWidth * canvas.width,
                bHeight * canvas.height,
                10
            );
            ctx.shadowBlur = 15; ctx.shadowColor = 'rgba(0, 212, 255, 0.8)';
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.8)'; ctx.lineWidth = 3; ctx.stroke();
            ctx.fillStyle = 'rgba(0, 150, 255, 0.1)'; ctx.fill();
            ctx.restore();
        }

        // --- 2. Draw Mannequin ---
        const drawSeg = (i1, i2, w, c = 'white') => {
            const p1 = landmarks[i1]; const p2 = landmarks[i2];
            if (p1?.visibility > 0.5 && p2?.visibility > 0.5) {
                ctx.beginPath();
                ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
                ctx.strokeStyle = c; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.stroke();
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

            // Head (Centered)
            ctx.beginPath();
            ctx.arc(midS.x * canvas.width, (midS.y - 0.08) * canvas.height, 9, 0, Math.PI * 2); // Shrank head for proportions
            ctx.fillStyle = 'white'; ctx.fill();
        }

        drawSeg(11, 13, 6); drawSeg(13, 15, 4);
        drawSeg(12, 14, 6); drawSeg(14, 16, 4);
        drawSeg(23, 25, 10); drawSeg(25, 27, 8);
        drawSeg(24, 26, 10); drawSeg(26, 28, 8);

        // --- 3. Extension Lines ---
        const drawExt = (i1, i2, c = 'rgba(0, 255, 255, 0.4)') => {
            const s = landmarks[i1]; const e = landmarks[i2];
            if (s?.visibility > 0.5 && e?.visibility > 0.5) {
                const x1 = s.x * canvas.width; const y1 = s.y * canvas.height;
                const x2 = e.x * canvas.width; const y2 = e.y * canvas.height;
                const dx = x2 - x1; const dy = y2 - y1;
                ctx.setLineDash([8, 8]); ctx.beginPath();
                ctx.moveTo(x1 - dx * 3, y1 - dy * 3); ctx.lineTo(x2 + dx * 3, y2 + dy * 3);
                ctx.strokeStyle = c; ctx.lineWidth = 1; ctx.stroke();
                ctx.setLineDash([]);
            }
        };
        drawExt(11, 23); drawExt(23, 27);

        // --- 4. Legend & Dots ---
        const keys = [
            { name: 'Spine Center', color: '#00BFFF', idx: 23 },
            { name: 'Shoulders', color: '#00FF00', idx: 11 },
            { name: 'Hips (L/R)', color: '#FF8C00', idx: 23, alt: 24 },
            { name: 'Knees (L/R)', color: '#FF1493', idx: 25, alt: 26 },
            { name: 'Ankles (L/R)', color: '#9370DB', idx: 27, alt: 28 }
        ];

        let ly = 25;
        keys.forEach(el => {
            const lms = [landmarks[el.idx], landmarks[el.alt]].filter(l => l?.visibility > 0.5);
            if (lms.length > 0) {
                // Legend Item with Background for legibility
                const labelWidth = ctx.measureText(el.name).width + 25;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.beginPath();
                ctx.roundRect(10, ly - 10, labelWidth, 18, 5);
                ctx.fill();

                ctx.beginPath(); ctx.arc(20, ly, 5, 0, Math.PI * 2);
                ctx.fillStyle = el.color; ctx.fill();
                ctx.font = 'bold 11px Inter, sans-serif'; ctx.fillStyle = 'white';
                ctx.fillText(el.name, 32, ly + 4);

                // World Dots
                lms.forEach(lm => {
                    ctx.beginPath(); ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 6, 0, Math.PI * 2);
                    ctx.fillStyle = el.color; ctx.fill();
                    ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke();
                });
                ly += 18;
            }
        });

        // --- 5. Visual Angle Arc ---
        if (metrics?.split_angle && landmarks[23] && landmarks[25] && landmarks[26]) {
            const h = landmarks[23]; const kL = landmarks[25]; const kR = landmarks[26];
            const cx = h.x * canvas.width; const cy = h.y * canvas.height;
            const aL = Math.atan2(kL.y - h.y, kL.x - h.x);
            const aR = Math.atan2(kR.y - h.y, kR.x - h.x);
            ctx.beginPath();
            ctx.arc(cx, cy, 30, aL, aR);
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)'; ctx.lineWidth = 4; ctx.stroke();
            ctx.fillStyle = 'rgba(0, 255, 255, 0.1)'; ctx.fill();
        }
    };

    useEffect(() => {
        const vid = videoRef.current;
        if (analysisResult?.frames?.length > 1 && mediaType === 'video' && vid) {
            const onTime = () => {
                const frame = analysisResult.frames.reduce((p, c) => Math.abs(c.time - vid.currentTime) < Math.abs(p.time - vid.currentTime) ? c : p);
                if (frame) {
                    drawSkeleton(frame.landmarks, frame.apparatus);
                    if (frame.apparatus) setCurrentApparatus(frame.apparatus);
                }
            };
            vid.addEventListener('timeupdate', onTime);
            return () => vid.removeEventListener('timeupdate', onTime);
        } else if (analysisResult?.landmarks) {
            drawSkeleton(analysisResult.landmarks, currentApparatus);
        }
    }, [analysisResult, currentApparatus]);

    return (
        <Card className="mb-4 shadow-lg border-0" style={{ background: 'rgba(13, 46, 18, 0.95)', borderRadius: '20px', color: 'white' }}>
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
            <Card.Body className="p-0 overflow-hidden" ref={containerRef} style={{ background: '#081209' }}>
                <Row className="g-0">
                    <Col md={6} className="bg-dark p-0" style={{ position: 'relative', height: '600px' }}>
                        {mediaUrl && (
                            mediaType === 'video' ?
                                <video ref={videoRef} src={mediaUrl} autoPlay loop muted playsInline className="w-100 h-100" style={{ objectFit: 'contain' }} /> :
                                <img src={mediaUrl} className="w-100 h-100" style={{ objectFit: 'contain' }} alt="Gymnastics" />
                        )}
                        <div className="position-absolute bottom-0 start-0 p-3 w-100 d-flex justify-content-end align-items-center" style={{ background: 'linear-gradient(transparent, rgba(0,0,0,0.6))' }}>
                            {/* Maximize button moved to header */}
                        </div>
                    </Col>
                    <Col md={6} className="p-0" style={{ position: 'relative', height: '600px', background: '#000' }}>
                        <canvas ref={canvasRef} className="w-100 h-100" style={{ display: 'block' }} />
                        <div className="position-absolute" style={{ top: '15px', right: '15px', background: 'rgba(0,0,0,0.85)', padding: '12px', borderRadius: '14px', display: 'flex', gap: '20px', backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.15)', boxShadow: '0 8px 32px rgba(0,0,0,0.4)', zIndex: 10 }}>
                            <div className="text-center"><div className="small opacity-50" style={{ fontSize: '10px' }}>SPLIT</div><div className="fw-bold text-info" style={{ fontSize: '1.2rem' }}>{metrics?.split_angle}</div></div>
                            <div className="text-center"><div className="small opacity-50" style={{ fontSize: '10px' }}>POS</div><Badge bg={isPass ? 'success' : 'danger'} style={{ fontSize: '12px', padding: '6px 10px' }}>{status.toUpperCase()}</Badge></div>
                            <div className="text-center" style={{ minWidth: '80px' }}>
                                <div className="small opacity-50" style={{ fontSize: '10px' }}>APPARATUS</div>
                                <div className="fw-bold text-white" style={{ fontSize: '12px', whiteSpace: 'normal', lineHeight: '1' }}>
                                    {currentApparatus ? currentApparatus.label : 'NONE'}
                                </div>
                            </div>
                            <div className="text-center">
                                <div className="small opacity-50" style={{ fontSize: '10px' }}>TRACK</div>
                                <Badge
                                    bg={currentApparatus ? 'info' : 'warning'}
                                    className={!currentApparatus ? 'animate-pulse' : ''}
                                    style={{ fontSize: '10px' }}
                                >
                                    {currentApparatus ? (currentApparatus.shorthand || 'LOCKED') : 'SEARCHING...'}
                                </Badge>
                            </div>
                        </div>
                    </Col>
                </Row>
                <Row className="mt-3">
                    <Col className="text-center">
                        <div className="p-3 rounded" style={{ background: 'rgba(255,255,255,0.05)' }}>
                            <div className="small text-warning">D-SCORE CONTRIBUTION</div>
                            <div className="display-4 fw-bold text-warning">+{d_score_contribution || 0.0}</div>
                        </div>
                    </Col>
                </Row>
            </Card.Body>
        </Card>
    );
};

export default WAGControlView;
