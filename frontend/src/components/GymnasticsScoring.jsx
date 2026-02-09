import React, { useState, useRef, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Badge, Spinner, Modal, Form, ProgressBar, OverlayTrigger, Tooltip } from 'react-bootstrap';
import api from '../services/api';
import confetti from 'canvas-confetti';
import TimelineChart from './TimelineChart'; // Import SVG Chart
import WAGControlView from './WAGControlView'; // Import new component

const SKILL_DESCRIPTIONS = {
    "Iron Cross": "A static strength hold on the rings where the gymnast holds the body suspended with arms extended horizontally, forming a T-shape. Requires immense shoulder and core strength.",
    "Y-Scale / Arabesque": "A targeted balance skill where the gymnast stands on one leg and lifts the other leg high to the side (Y-Scale) or back (Arabesque), showcasing extreme flexibility and control.",
    "Handstand": "A fundamental inverted position where the body is held straight and vertical, supported only by the hands. Key metrics include extensive shoulder angle and a hollow body position.",
    "Bridge": "A flexibility skill involving arching the back while supported by hands and feet. Demonstrates spinal and shoulder mobility.",
    "Split": "A flexibility element where legs are extended in opposite directions, forming a straight line (180 degrees).",
    "Pose": "A static gymnastic position demonstrating form and alignment."
};

const GymnasticsScoring = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [mediaType, setMediaType] = useState('image'); // 'image' or 'video'
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [category, setCategory] = useState(null);
    const [showCategoryModal, setShowCategoryModal] = useState(false);

    // Refs
    const fileInputRef = useRef(null);
    const videoInputRef = useRef(null);
    const videoPlayerRef = useRef(null); // Ref for the actual video element

    const [processingTime, setProcessingTime] = useState(0);
    const [energyLevel, setEnergyLevel] = useState(100);
    const [showSkillInfo, setShowSkillInfo] = useState(false);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            const isVideo = selectedFile.type.startsWith('video/');
            setMediaType(isVideo ? 'video' : 'image');

            const reader = new FileReader();
            reader.onloadend = () => {
                setFile(selectedFile);
                setPreview(reader.result);
                setResult(null);
                setEnergyLevel(0);
                setCategory(null); // Reset category on new upload
                setShowCategoryModal(true); // Show modal immediately after upload
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    const handleScore = async () => {
        if (!preview) return;
        if (!category) {
            setShowCategoryModal(true);
            return;
        }

        setLoading(true);
        setEnergyLevel(20); // Start energy bar animation

        // Simulate energy bar filling up while loading
        const interval = setInterval(() => {
            setEnergyLevel(prev => Math.min(prev + 10, 90));
        }, 500);

        try {
            // Send media_type, media_data and category
            const response = await api.analyzeGymnastics(preview, mediaType, category);
            clearInterval(interval);

            if (!response.data || response.data.error) {
                throw new Error(response.data?.error || "Invalid response from server");
            }

            // Calculate dynamic energy level (Safety checks added)
            const totalScore = response.data.total_score || 0;
            const skills = response.data.skill || []; // Ensure array
            const currentSkill = Array.isArray(skills) ? skills : [skills]; // Handle string vs array

            let calculatedEnergy = (totalScore / 20) * 100;

            // Artificial Boost for "Power" skills
            if (["Iron Cross", "Handstand", "Planche"].some(s => currentSkill.join(' ').includes(s))) {
                calculatedEnergy += 15;
            }
            if ((response.data.difficulty || 0) > 3.0) calculatedEnergy += 10;

            setEnergyLevel(Math.min(calculatedEnergy, 100));
            setResult(response.data);

            // Trigger confetti if score is high
            if (totalScore > 15) {
                triggerConfetti();
            }

        } catch (error) {
            clearInterval(interval);
            console.error("Error scoring gymnastics:", error);
            alert(`The judges stumbled! 😵\n${error.message || "Unknown error"}`);
            setEnergyLevel(0);
        } finally {
            setLoading(false);
        }
    };

    const handleSeek = (time) => {
        if (videoPlayerRef.current) {
            videoPlayerRef.current.currentTime = time;
            videoPlayerRef.current.play(); // Auto-play from click
        }
    };

    const handleHoverScan = (time) => {
        if (videoPlayerRef.current) {
            videoPlayerRef.current.currentTime = time;
            videoPlayerRef.current.pause(); // Pause while scrubbing/hovering
        }
    };

    const triggerFileInput = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const triggerVideoInput = () => {
        if (videoInputRef.current) {
            videoInputRef.current.click();
        }
    };

    const triggerConfetti = () => {
        confetti({
            particleCount: 150,
            spread: 70,
            origin: { y: 0.6 },
            colors: ['#46d2e1', '#e91e63', '#ffeb3b'] // Junior Olympian colors
        });
    };

    const getBadge = (score) => {
        if (score >= 18) return { text: "PERFECT FORM! 🌟", color: "var(--junior-pink)", textColor: "#fff" };
        if (score >= 15) return { text: "GREAT JOB! 🥇", color: "var(--junior-cyan)", textColor: "#fff" };
        return { text: "KEEP GOING! 💪", color: "var(--junior-yellow)", textColor: "#333" };
    };

    const getEnergyVariant = (level) => {
        if (level >= 80) return "high"; // High Energy (Red/Pink)
        if (level >= 50) return "med"; // Medium Energy (Yellow)
        return "low"; // Low Energy (Blue)
    };

    return (
        <div className="text-center font-quicksand">
            <h2 className="mb-4 display-4" style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)', textShadow: '2px 2px var(--junior-yellow)' }}>
                Agility Ace Finder 🤸‍♀️
            </h2>

            {/* 3-Column Layout for Active Preview */}
            {preview ? (
                <div className="mb-4 w-100 animate__animated animate__fadeIn" style={{ maxWidth: '1200px', margin: '0 auto' }}>
                    <Row className="align-items-center g-4">
                        {/* LEFT COLUMN: Vertical Energy Meter */}
                        <Col md={2} className="d-flex flex-column align-items-center">
                            <h5 className="fw-bold mb-3 text-center" style={{ color: 'var(--junior-cyan)', textShadow: '0 0 10px rgba(70, 210, 225, 0.3)', fontSize: '0.9rem' }}>
                                ENERGY ⚡
                            </h5>
                            <div className="energy-meter-vertical">
                                <span className="energy-text-vertical">
                                    {Math.round(energyLevel)}%
                                </span>
                                <div
                                    className={`energy-meter-bar-vertical variant-${getEnergyVariant(energyLevel)}`}
                                    style={{ height: `${Math.max(energyLevel, 5)}%` }} // Min height
                                >
                                </div>
                            </div>
                        </Col>

                        {/* CENTER COLUMN: Media + Overlay Button */}
                        <Col md={7}>
                            <div className="media-overlay-container">
                                {mediaType === 'video' ? (
                                    <video
                                        ref={videoPlayerRef}
                                        src={preview}
                                        controls
                                        style={{ width: '100%', display: 'block' }}
                                    />
                                ) : (
                                    <img src={preview} alt="Gymnastics Pose" style={{ width: '100%', display: 'block' }} />
                                )}

                                {/* Overlay "Score Me" Button */}
                                {!result && (
                                    <div className="score-me-overlay">
                                        <Button
                                            className="w-100 btn-lg rounded-pill shadow-lg border-0"
                                            style={{
                                                backgroundColor: 'var(--junior-pink)',
                                                fontFamily: 'Fredoka One',
                                                fontSize: '1.5rem',
                                                padding: '12px 0'
                                            }}
                                            onClick={handleScore}
                                            disabled={loading}
                                        >
                                            {loading ? (
                                                <>
                                                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                                                    Analyzing...
                                                </>
                                            ) : 'Score Me! 🏆'}
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </Col>

                        {/* RIGHT COLUMN: Stacked Upload Buttons (Beautified) */}
                        <Col md={3} className="d-flex flex-column gap-3 justify-content-center">
                            <div
                                className="card-jungle-interactive photo-variant p-2 w-100 d-flex flex-column justify-content-center align-items-center"
                                style={{ minHeight: '140px', borderRadius: '20px' }}
                                onClick={triggerFileInput}
                            >
                                <div className="icon-bounce" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>📸</div>
                                <h6 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-cyan)', margin: 0 }}>
                                    {mediaType === 'video' ? "Try Photo" : "New Photo"}
                                </h6>
                            </div>

                            <div
                                className="card-jungle-interactive video-variant p-2 w-100 d-flex flex-column justify-content-center align-items-center"
                                style={{ minHeight: '140px', borderRadius: '20px' }}
                                onClick={triggerVideoInput}
                            >
                                <div className="icon-bounce" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>🎥</div>
                                <h6 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)', margin: 0 }}>
                                    {mediaType === 'photo' ? "Try Video" : "New Video"}
                                </h6>
                            </div>
                        </Col>
                    </Row>
                </div>
            ) : (
                /* INITIAL STATE: "Beautified" Compacter Upload Boxes */
                <Row className="g-4 justify-content-center mb-5">
                    <Col md={5}>
                        <div
                            className="card-jungle-interactive photo-variant p-3 h-100 d-flex flex-column justify-content-center align-items-center"
                            style={{ minHeight: '220px' }}
                            onClick={triggerFileInput}
                        >
                            <div className="icon-bounce" style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>📸</div>
                            <h4 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-cyan)' }}>Upload Photo</h4>
                            <p className="text-muted small mb-0">Strike a pose!</p>
                        </div>
                    </Col>
                    <Col md={5}>
                        <div
                            className="card-jungle-interactive video-variant p-3 h-100 d-flex flex-column justify-content-center align-items-center"
                            style={{ minHeight: '220px' }}
                            onClick={triggerVideoInput}
                        >
                            <div className="icon-bounce" style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>🎥</div>
                            <h4 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)' }}>Upload Video</h4>
                            <p className="text-muted small mb-0">Show us your routine!</p>
                        </div>
                    </Col>
                </Row>
            )}

            {/* Hidden Inputs */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
            />
            <input
                type="file"
                ref={videoInputRef}
                onChange={handleFileChange}
                accept="video/*"
                style={{ display: 'none' }}
            />

            {/* Old Score Button & Energy Bar removed (now integrated above) */}

            {result && (
                <div className="animate__animated animate__bounceIn">
                    {/* Badge Popup */}
                    <div className="mb-3">
                        <span className="badge rounded-pill px-4 py-2 shadow-sm" style={{ backgroundColor: getBadge(result.total_score).color, fontSize: '1.2rem', color: getBadge(result.total_score).textColor }}>
                            {getBadge(result.total_score).text}
                        </span>
                        <Badge bg="dark" className="ms-2 p-2" style={{ fontSize: '0.9rem', opacity: 0.8 }}>
                            Group: {category}
                        </Badge>
                    </div>

                    {/* WAG Control View Integration */}
                    {/* WAG Control View Integration */}
                    <WAGControlView analysisResult={result} mediaUrl={preview} mediaType={mediaType} />

                    <Card className="p-4 mx-auto text-start shadow-lg border-0" style={{ maxWidth: '600px', borderRadius: '30px', background: 'linear-gradient(135deg, #ffffff 0%, #fffde7 100%)' }}>
                        <div className="text-center mb-4">
                            <h3 className="display-3 fw-bold" style={{ fontFamily: 'Fredoka One', color: 'var(--junior-cyan)' }}>
                                {result.total_score} <span style={{ fontSize: '1.5rem', color: '#ccc' }}>/ 20</span>
                            </h3>
                            <div className="d-flex justify-content-center">
                                {[...Array(5)].map((_, i) => (
                                    <span key={i} style={{ fontSize: '2rem', color: i < Math.round(result.total_score / 4) ? 'var(--junior-yellow)' : '#e0e0e0' }}>★</span>
                                ))}
                            </div>
                        </div>

                        {/* Classification & Skill */}
                        <div className="mb-4 text-center">
                            {result.classification && (
                                <Badge bg="info" className="me-2 p-2" style={{ fontSize: '1rem' }}>{result.classification}</Badge>
                            )}
                            {result.skill && result.skill !== "Pose" && (
                                <>
                                    <Badge bg="warning" text="dark" className="p-2" style={{ fontSize: '1rem' }}>
                                        {result.skill}
                                    </Badge>
                                    <OverlayTrigger
                                        placement="top"
                                        overlay={<Tooltip id="skill-tooltip">Click for {result.skill} details</Tooltip>}
                                    >
                                        <Button
                                            variant="link"
                                            className="ms-3 p-0 text-decoration-none"
                                            onClick={() => setShowSkillInfo(true)}
                                            style={{
                                                fontSize: '1.8rem',
                                                verticalAlign: 'middle',
                                                filter: 'drop-shadow(0 0 5px gold)',
                                                cursor: 'pointer',
                                                border: 'none',
                                                background: 'transparent'
                                            }}
                                            aria-label="Skill Info"
                                        >
                                            💡
                                        </Button>
                                    </OverlayTrigger>

                                    <Modal show={showSkillInfo} onHide={() => setShowSkillInfo(false)} centered className="font-quicksand text-dark">
                                        <Modal.Header closeButton className="bg-light">
                                            <Modal.Title className="font-cinzel text-success fw-bold">
                                                {result.skill}
                                            </Modal.Title>
                                        </Modal.Header>
                                        <Modal.Body>
                                            <div className="text-center mb-3">
                                                <span style={{ fontSize: '3rem' }}>
                                                    {result.skill === "Iron Cross" ? "🪐" :
                                                        result.skill.includes("Scale") ? "Yz" :
                                                            result.skill === "Handstand" ? "🤸‍♂️" : "✨"}
                                                </span>
                                            </div>
                                            <div className="fs-5">
                                                {SKILL_DESCRIPTIONS[result.skill] || "A fundamental gymnastics skill requiring strength, balance, and precision."}
                                            </div>
                                            <div className="mt-3 small text-muted fst-italic">
                                                Source: MasterClass & Professional Guidelines.
                                            </div>
                                        </Modal.Body>
                                        <Modal.Footer>
                                            <Button variant="success" onClick={() => setShowSkillInfo(false)}>
                                                Got it!
                                            </Button>
                                        </Modal.Footer>
                                    </Modal>
                                </>
                            )}
                        </div>

                        {/* FIG Scoring Breakdown (D-Score & E-Score) */}
                        <div className="mb-4">
                            <Row className="text-center g-3">
                                <Col xs={6}>
                                    <div className="p-3 rounded-4 shadow-sm h-100" style={{ backgroundColor: '#fff', border: '2px solid var(--junior-pink)' }}>
                                        <div className="text-muted small fw-bold text-uppercase">D-SCORE (DIFF)</div>
                                        <div className="display-4 fw-bold mb-0" style={{ color: 'var(--junior-pink)', fontFamily: 'Fredoka One' }}>
                                            {result.difficulty}
                                        </div>
                                        {result.d_score_breakdown && (
                                            <div className="mt-2 pt-2 border-top text-start" style={{ fontSize: '0.75rem', color: '#666' }}>
                                                <div className="d-flex justify-content-between"><span>Value (DV):</span> <span className="fw-bold">+{result.d_score_breakdown.top_8_total?.toFixed(1) ?? '0.0'}</span></div>
                                                <div className="d-flex justify-content-between"><span>Req (CR):</span> <span className="fw-bold">+{result.d_score_breakdown.cr_score?.toFixed(1) ?? '0.0'}</span></div>
                                                <div className="d-flex justify-content-between"><span>Conn (CV):</span> <span className="fw-bold">+{result.d_score_breakdown.cv_score?.toFixed(1) ?? '0.0'}</span></div>
                                            </div>
                                        )}
                                    </div>
                                </Col>
                                <Col xs={6}>
                                    <div className="p-3 rounded-4 shadow-sm h-100" style={{ backgroundColor: '#fff', border: '2px solid var(--junior-cyan)' }}>
                                        <div className="text-muted small fw-bold text-uppercase">E-SCORE (EXEC)</div>
                                        <div className="display-4 fw-bold mb-0" style={{ color: 'var(--junior-cyan)', fontFamily: 'Fredoka One' }}>
                                            {result.execution}
                                        </div>
                                        <div className="mt-2 pt-2 border-top text-start" style={{ fontSize: '0.75rem', color: '#666' }}>
                                            <div className="d-flex justify-content-between"><span>Base:</span> <span className="fw-bold">10.0</span></div>
                                            <div className="d-flex justify-content-between text-danger"><span>Deductions:</span> <span className="fw-bold">-{((10 - (result.execution ?? 10)) || 0).toFixed(1)}</span></div>
                                        </div>
                                    </div>
                                </Col>
                            </Row>

                            {/* Recognized Skills List */}
                            {result.d_score_breakdown?.top_8_skills && result.d_score_breakdown.top_8_skills.length > 0 && (
                                <div className="mt-3 p-2 rounded-3" style={{ background: 'rgba(233, 30, 99, 0.05)', border: '1px dashed var(--junior-pink)' }}>
                                    <div className="small fw-bold text-uppercase mb-1" style={{ fontSize: '0.7rem', color: 'var(--junior-pink)' }}>🏆 Recognized Routine Elements:</div>
                                    <div className="d-flex flex-wrap gap-1">
                                        {result.d_score_breakdown.top_8_skills.map((skill, idx) => (
                                            <Badge key={idx} bg="none" style={{ border: '1px solid var(--junior-pink)', color: 'var(--junior-pink)', fontSize: '0.7rem' }}>
                                                {skill}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Deductions List */}
                            {result.deductions && result.deductions.length > 0 && (
                                <div className="mt-3 p-3 rounded-3" style={{ backgroundColor: '#ffebee', border: '1px solid #ffcdd2' }}>
                                    <h6 className="text-danger fw-bold mb-2">🚩 Deductions:</h6>
                                    <ul className="list-unstyled mb-0 small text-start">
                                        {result.deductions.map((d, i) => (
                                            <li key={i} className="mb-1">
                                                <Badge bg="danger" className="me-2">-{d.points}</Badge>
                                                {d.reason}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Technical Analysis & Coach's Corner */}
                            {(result.technical_cue || (result.focus_anatomy && result.focus_anatomy.length > 0)) && (
                                <div className="mt-4 p-3 rounded-4 text-start" style={{ background: 'linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%)', border: '1px solid #e0e0e0' }}>
                                    <h5 className="fw-bold mb-3" style={{ color: '#2c3e50', fontFamily: 'Cinzel, serif' }}>
                                        🎓 Coach's Corner
                                    </h5>

                                    {/* Technical Cue */}
                                    {result.technical_cue && (
                                        <div className="mb-3 d-flex align-items-start">
                                            <span className="me-2 fs-4">💡</span>
                                            <div>
                                                <strong className="d-block text-primary">Technical Cue:</strong>
                                                <span className="fst-italic text-muted">"{result.technical_cue}"</span>
                                            </div>
                                        </div>
                                    )}

                                    {/* Focus Anatomy */}
                                    {result.focus_anatomy && result.focus_anatomy.length > 0 && (
                                        <div className="mb-3">
                                            <strong className="d-block text-primary mb-1">💪 Focus Anatomy:</strong>
                                            <div className="d-flex flex-wrap gap-1">
                                                {result.focus_anatomy.map((muscle, idx) => (
                                                    <Badge key={idx} bg="secondary" className="fw-normal">
                                                        {muscle}
                                                    </Badge>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Common Errors (Knowledge Base) */}
                                    {result.common_errors && result.common_errors.length > 0 && (
                                        <div>
                                            <strong className="d-block text-danger mb-1">⚠️ Watch Out For:</strong>
                                            <ul className="mb-0 ps-3 small text-muted">
                                                {result.common_errors.map((err, idx) => (
                                                    <li key={idx}>
                                                        {err.reason} <span className="text-danger fw-bold">(-{err.penalty})</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* VIDEO SPECIFIC METRICS: TIMELINE & APEX */}
                        {result.is_video && result.video_analysis && (
                            <div className="mb-4">
                                {/* Hold Duration & Stability Cards */}
                                <Row className="mb-3 text-center">
                                    <Col xs={6}>
                                        <div className="p-3 rounded-3 shadow-sm" style={{ backgroundColor: '#e1f5fe', border: '2px solid var(--junior-cyan)' }}>
                                            <h6 className="fw-bold text-muted">⏱️ Hold Duration</h6>
                                            <h3 style={{ color: 'var(--junior-pink)', fontFamily: 'Fredoka One' }}>{result.video_analysis?.hold_duration ?? '0s'}</h3>
                                        </div>
                                    </Col>
                                    <Col xs={6}>
                                        <div className="p-3 rounded-3 shadow-sm" style={{ backgroundColor: '#e8f5e9', border: '2px solid #66bb6a' }}>
                                            <h6 className="fw-bold text-muted">⚖️ Stability</h6>
                                            <h3 style={{ color: '#2e7d32', fontFamily: 'Fredoka One' }}>{result.video_analysis?.stability_rating ?? 'N/A'}</h3>
                                        </div>
                                    </Col>
                                </Row>

                                {/* Apex Metrics - The "Best" Frame */}
                                {result.technical_metrics && (
                                    <div className="text-center mb-3">
                                        <Badge bg="danger" className="p-2 mb-2">🔥 APEX MOMENT DETECTED 🔥</Badge>
                                    </div>
                                )}

                                {/* Timeline Chart - Pass handleSeek */}
                                {result.video_analysis.timeline && (
                                    <TimelineChart
                                        data={result.video_analysis.timeline}
                                        onSeek={handleSeek}
                                        onHover={handleHoverScan}
                                        videoSrc={preview} // Pass video blob for tooltip
                                    />
                                )}
                            </div>
                        )}

                        <Row className="mb-3 align-items-center">
                            <Col xs={4} className="fw-bold text-end" style={{ color: 'var(--junior-pink)' }}>
                                <OverlayTrigger placement="left" overlay={<Tooltip>Difficulty Value + Composition Requirements + Connection Value</Tooltip>}>
                                    <span>Difficulty (D)</span>
                                </OverlayTrigger>
                            </Col>
                            <Col xs={8}>
                                <ProgressBar variant="warning" now={(result.difficulty / 10) * 100} label={`${result.difficulty} `} style={{ height: '15px', borderRadius: '10px' }} />
                            </Col>
                        </Row>
                        <Row className="mb-3 align-items-center">
                            <Col xs={4} className="fw-bold text-end" style={{ color: 'var(--junior-cyan)' }}>
                                <OverlayTrigger placement="left" overlay={<Tooltip>10.0 Base - Technical Deductions</Tooltip>}>
                                    <span>Execution (E)</span>
                                </OverlayTrigger>
                            </Col>
                            <Col xs={8}>
                                <ProgressBar variant="success" now={(result.execution / 10) * 100} label={`${result.execution} `} style={{ height: '15px', borderRadius: '10px' }} />
                            </Col>
                        </Row>
                        <Row className="mb-4 align-items-center">
                            <Col xs={4} className="fw-bold text-end" style={{ color: '#9c27b0' }}>
                                <OverlayTrigger placement="left" overlay={<Tooltip>Choreography, Composition, and Body Posture stability (Max 5.0)</Tooltip>}>
                                    <span>Artistry (A)</span>
                                </OverlayTrigger>
                            </Col>
                            <Col xs={8}>
                                <ProgressBar variant="info" now={(result.artistry / 5) * 100} label={`${result.artistry} `} style={{ height: '15px', borderRadius: '10px' }} />
                            </Col>
                        </Row>

                        {/* Technical Metrics Section (Apex or Static) */}
                        {result.technical_metrics && (
                            <div className="mb-4 p-3 rounded-3" style={{ backgroundColor: '#e3f2fd', border: '2px dashed var(--junior-cyan)' }}>
                                <h4 className="text-center fw-bold mb-3" style={{ color: 'var(--junior-cyan)', fontFamily: 'Fredoka One' }}>📏 Tech Specs 📏</h4>
                                <Row className="text-center g-2">
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">Left Knee: {result.technical_metrics.left_knee_extension}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">Right Knee: {result.technical_metrics.right_knee_extension}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">Hip Flex: {result.technical_metrics.hip_flexibility}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">Toe Point: {result.technical_metrics.toe_point}</div></Col>
                                </Row>
                            </div>
                        )}

                        {result.advanced_metrics && (
                            <div className="mb-4 p-3 rounded-3" style={{ backgroundColor: 'rgba(70, 210, 225, 0.1)' }}>
                                <h4 className="text-center fw-bold mb-3" style={{ color: 'var(--junior-cyan)' }}>✨ Magic Metrics ✨</h4>
                                <Row className="text-center g-2">
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">👀 Gaze: {result.advanced_metrics.gaze_stability}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">😉 Blink: {result.advanced_metrics.blink_rate}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">🎭 Face: {result.advanced_metrics.micro_expressions}</div></Col>
                                    <Col xs={6}><div className="bg-white p-2 rounded shadow-sm">🔄 Rotation: {result.advanced_metrics.head_rotation}</div></Col>
                                </Row>
                            </div>
                        )}

                        <div className="text-center p-3 rounded-3" style={{ backgroundColor: '#fce4ec', border: '2px dashed var(--junior-pink)' }}>
                            <strong style={{ color: 'var(--junior-pink)', fontSize: '1.1rem' }}>📣 Coach says:</strong> <br />
                            <span className="fst-italic">"{result.comment}"</span>
                        </div>
                    </Card>
                </div >
            )}
            {/* Category Selection Modal */}
            <Modal
                show={showCategoryModal}
                onHide={() => setShowCategoryModal(false)}
                centered
                backdrop="static"
                className="font-quicksand"
            >
                <Modal.Header className="border-0 pb-0 justify-content-center">
                    <Modal.Title style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)', fontSize: '2rem' }}>
                        Who's the Lion? 🦁
                    </Modal.Title>
                </Modal.Header>
                <Modal.Body className="p-4">
                    <p className="text-center text-muted mb-4 fs-5">Pick your category for the best judging results!</p>
                    <div className="d-grid gap-3">
                        <Button
                            variant="light"
                            className="p-3 fs-5 rounded-4 shadow-sm border-2"
                            style={{ borderColor: 'var(--junior-cyan)', transition: 'all 0.2s' }}
                            onClick={() => { setCategory("U8 (Tiny Tots)"); setShowCategoryModal(false); }}
                        >
                            👶 U8 (Tiny Tots)
                        </Button>
                        <Button
                            variant="light"
                            className="p-3 fs-5 rounded-4 shadow-sm border-2"
                            style={{ borderColor: 'var(--junior-cyan)', transition: 'all 0.2s' }}
                            onClick={() => { setCategory("U10 (Beginner)"); setShowCategoryModal(false); }}
                        >
                            🐣 U10 (Beginner)
                        </Button>
                        <Button
                            variant="light"
                            className="p-3 fs-5 rounded-4 shadow-sm border-2"
                            style={{ borderColor: 'var(--junior-yellow)', transition: 'all 0.2s' }}
                            onClick={() => { setCategory("U12 (Sub Junior)"); setShowCategoryModal(false); }}
                        >
                            🦁 U12 (Sub Junior)
                        </Button>
                        <Button
                            variant="light"
                            className="p-3 fs-5 rounded-4 shadow-sm border-2"
                            style={{ borderColor: 'var(--junior-yellow)', transition: 'all 0.2s' }}
                            onClick={() => { setCategory("U14 (Intermediate)"); setShowCategoryModal(false); }}
                        >
                            🐯 U14 (Intermediate)
                        </Button>
                        <Button
                            variant="light"
                            className="p-3 fs-5 rounded-4 shadow-sm border-2"
                            style={{ borderColor: 'var(--junior-pink)', transition: 'all 0.2s' }}
                            onClick={() => { setCategory("Senior (Elite)"); setShowCategoryModal(false); }}
                        >
                            👑 Senior (Elite)
                        </Button>
                    </div>
                </Modal.Body>
                <Modal.Footer className="border-0 pt-0">
                    <Button variant="link" className="text-muted mx-auto" onClick={() => setShowCategoryModal(false)}>
                        Skip for now
                    </Button>
                </Modal.Footer>
            </Modal>
        </div >
    );
};

export default GymnasticsScoring;
