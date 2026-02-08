
import React, { useState, useRef, useEffect } from 'react';
import { Container, Row, Col, Card, ProgressBar, Button, Badge, Spinner, Modal, OverlayTrigger, Tooltip } from 'react-bootstrap';
import api from '../services/api';
import confetti from 'canvas-confetti';
import TimelineChart from './TimelineChart'; // Import SVG Chart

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
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    const handleScore = async () => {
        if (!preview) return;

        setLoading(true);
        setEnergyLevel(20); // Start energy bar animation

        // Simulate energy bar filling up while loading
        const interval = setInterval(() => {
            setEnergyLevel(prev => Math.min(prev + 10, 90));
        }, 500);

        try {
            // Send media_type and media_data
            const response = await api.analyzeGymnastics(preview, mediaType);
            clearInterval(interval);
            setEnergyLevel(100);
            setResult(response.data);

            // Trigger confetti if score is high
            if (response.data.total_score > 15) {
                triggerConfetti();
            }

        } catch (error) {
            clearInterval(interval);
            console.error("Error scoring gymnastics:", error);
            alert("The judges need a break! (API Error)");
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
        if (score >= 18) return { text: "PERFECT FORM! 🌟", color: "var(--junior-pink)" };
        if (score >= 15) return { text: "GREAT JOB! 🥇", color: "var(--junior-cyan)" };
        return { text: "KEEP GOING! 💪", color: "var(--junior-yellow)" };
    };

    return (
        <div className="text-center font-quicksand">
            <h2 className="mb-4 display-4" style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)', textShadow: '2px 2px var(--junior-yellow)' }}>
                Agility Ace Finder 🤸‍♀️
            </h2>

            {/* Energy Bar / Star Meter - Only show after upload */}
            {preview && (
                <div className="mb-4 mx-auto animate__animated animate__fadeIn" style={{ maxWidth: '600px' }}>
                    <h5 className="fw-bold" style={{ color: 'var(--junior-cyan)' }}>Energy Meter ⚡</h5>
                    <ProgressBar
                        now={energyLevel}
                        striped
                        animated
                        variant="info"
                        style={{ height: '25px', borderRadius: '15px', backgroundColor: '#e0f7fa' }}
                    />
                </div>
            )}

            {/* Upload Area */}
            <div className="mb-4 w-100" style={{ maxWidth: '800px', margin: '0 auto' }}>
                {!preview ? (
                    /* INITIAL STATE: Side-by-Side Options */
                    <Row className="g-4 justify-content-center">
                        <Col md={6}>
                            <div
                                className="card-jungle p-4 shadow-lg text-center h-100 d-flex flex-column justify-content-center align-items-center"
                                style={{
                                    cursor: 'pointer',
                                    minHeight: '250px',
                                    backgroundColor: '#fff',
                                    border: '4px solid var(--junior-cyan)',
                                    borderRadius: '30px',
                                    transition: 'transform 0.2s'
                                }}
                                onClick={triggerFileInput}
                                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
                                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            >
                                <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>📸</div>
                                <h3 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-cyan)' }}>Upload Photo</h3>
                                <p className="text-muted">Strike a pose!</p>
                            </div>
                        </Col>
                        <Col md={6}>
                            <div
                                className="card-jungle p-4 shadow-lg text-center h-100 d-flex flex-column justify-content-center align-items-center"
                                style={{
                                    cursor: 'pointer',
                                    minHeight: '250px',
                                    backgroundColor: '#fff',
                                    border: '4px solid var(--junior-pink)',
                                    borderRadius: '30px',
                                    transition: 'transform 0.2s'
                                }}
                                onClick={triggerVideoInput}
                                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
                                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            >
                                <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🎥</div>
                                <h3 style={{ fontFamily: 'Fredoka One', color: 'var(--junior-pink)' }}>Upload Video</h3>
                                <p className="text-muted">Show us your routine!</p>
                            </div>
                        </Col>
                    </Row>
                ) : (
                    /* PREVIEW STATE: Single Card + Buttons */
                    <div className="d-flex flex-column align-items-center position-relative">
                        <div
                            className="card-jungle p-1 mx-auto shadow-lg mb-3"
                            style={{
                                width: '100%',
                                maxWidth: '500px',
                                minHeight: '300px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                backgroundColor: '#fff',
                                border: '5px solid var(--junior-cyan)',
                                borderRadius: '30px'
                            }}
                        >
                            {mediaType === 'video' ? (
                                <video
                                    ref={videoPlayerRef}
                                    src={preview}
                                    controls
                                    style={{ width: '100%', borderRadius: '25px' }}
                                />
                            ) : (
                                <img src={preview} alt="Gymnastics Pose" style={{ width: '100%', borderRadius: '25px' }} />
                            )}
                        </div>

                        {/* Dynamic Upload Options */}
                        <div className="d-flex gap-2 justify-content-center">
                            <Button
                                variant="outline-info"
                                className="rounded-pill px-4 py-2 shadow-sm font-quicksand fw-bold"
                                style={{ backgroundColor: 'white', border: '2px solid var(--junior-cyan)', color: 'var(--junior-cyan)' }}
                                onClick={triggerFileInput}
                            >
                                {mediaType === 'video' ? "Upload Image Instead 📸" : "Upload Another Photo 📸"}
                            </Button>
                            <Button
                                variant="outline-info"
                                className="rounded-pill px-4 py-2 shadow-sm font-quicksand fw-bold"
                                style={{ backgroundColor: 'white', border: '2px solid var(--junior-pink)', color: 'var(--junior-pink)' }}
                                onClick={triggerVideoInput}
                            >
                                {mediaType === 'video' ? "Upload Another Video 🎥" : "Upload Video Instead 🎥"}
                            </Button>
                        </div>
                    </div>
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
            </div>

            {preview && !result && (
                <Button
                    className="btn-lg mb-4 px-5 py-3 rounded-pill shadow"
                    style={{ backgroundColor: 'var(--junior-pink)', border: 'none', fontFamily: 'Fredoka One', fontSize: '1.5rem' }}
                    onClick={handleScore}
                    disabled={loading}
                >
                    {loading ? (
                        <>
                            <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                            Analyzing Video... {Math.round(energyLevel)}%
                        </>
                    ) : 'Score Me! 🏆'}
                </Button>
            )}

            {result && (
                <div className="animate__animated animate__bounceIn">
                    {/* Badge Popup */}
                    <div className="mb-3">
                        <span className="badge rounded-pill px-4 py-2 shadow-sm" style={{ backgroundColor: getBadge(result.total_score).color, fontSize: '1.2rem', color: '#fff' }}>
                            {getBadge(result.total_score).text}
                        </span>
                    </div>

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
                                        <div className="text-muted small fw-bold text-uppercase">D-Score (Diff)</div>
                                        <div className="display-4 fw-bold" style={{ color: 'var(--junior-pink)', fontFamily: 'Fredoka One' }}>
                                            {result.difficulty}
                                        </div>
                                    </div>
                                </Col>
                                <Col xs={6}>
                                    <div className="p-3 rounded-4 shadow-sm h-100" style={{ backgroundColor: '#fff', border: '2px solid var(--junior-cyan)' }}>
                                        <div className="text-muted small fw-bold text-uppercase">E-Score (Exec)</div>
                                        <div className="display-4 fw-bold" style={{ color: 'var(--junior-cyan)', fontFamily: 'Fredoka One' }}>
                                            {result.execution}
                                        </div>
                                    </div>
                                </Col>
                            </Row>

                            {/* Deductions List */}
                            {result.deductions && result.deductions.length > 0 && (
                                <div className="mt-3 p-3 rounded-3" style={{ backgroundColor: '#ffebee', border: '1px solid #ffcdd2' }}>
                                    <h6 className="fw-bold text-danger mb-2">🚩 Judges' Deductions:</h6>
                                    <ul className="mb-0 text-start small text-danger ps-3">
                                        {result.deductions.map((d, i) => (
                                            <li key={i}>{d}</li>
                                        ))}
                                    </ul>
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
                                            <h3 style={{ color: 'var(--junior-pink)', fontFamily: 'Fredoka One' }}>{result.video_analysis.hold_duration}</h3>
                                        </div>
                                    </Col>
                                    <Col xs={6}>
                                        <div className="p-3 rounded-3 shadow-sm" style={{ backgroundColor: '#e8f5e9', border: '2px solid #66bb6a' }}>
                                            <h6 className="fw-bold text-muted">⚖️ Stability</h6>
                                            <h3 style={{ color: '#2e7d32', fontFamily: 'Fredoka One' }}>{result.video_analysis.stability_rating}</h3>
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
                            <Col xs={4} className="fw-bold text-end" style={{ color: 'var(--junior-pink)' }}>Difficulty</Col>
                            <Col xs={8}>
                                <ProgressBar variant="warning" now={(result.difficulty / 10) * 100} style={{ height: '15px', borderRadius: '10px' }} />
                            </Col>
                        </Row>
                        <Row className="mb-3 align-items-center">
                            <Col xs={4} className="fw-bold text-end" style={{ color: 'var(--junior-cyan)' }}>Execution</Col>
                            <Col xs={8}>
                                <ProgressBar variant="success" now={(result.execution / 10) * 100} style={{ height: '15px', borderRadius: '10px' }} />
                            </Col>
                        </Row>
                        <Row className="mb-4 align-items-center">
                            <Col xs={4} className="fw-bold text-end" style={{ color: '#9c27b0' }}>Artistry</Col>
                            <Col xs={8}>
                                <ProgressBar variant="info" now={(result.artistry / 5) * 100} style={{ height: '15px', borderRadius: '10px' }} />
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
                </div>
            )}
        </div>
    );
};

export default GymnasticsScoring;
