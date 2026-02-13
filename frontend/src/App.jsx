import { useState, useEffect } from 'react'
import { Activity } from 'lucide-react';
import { Container, Row, Col, Navbar, Button, Nav } from 'react-bootstrap';
import { Routes, Route, Navigate, NavLink, Link } from 'react-router-dom';
import CameraCapture from './components/CameraCapture';
import GoogleFitConnect from './components/GoogleFitConnect';
import HealthTable from './components/HealthTable';
import LatestInsights from './components/LatestInsights';
import GymnasticsScoring from './components/GymnasticsScoring';
import Auth from './components/Auth';
import api from './services/api';

function App() {
    const [user, setUser] = useState(null);
    const [history, setHistory] = useState([]);
    const [currentResult, setCurrentResult] = useState(null);
    const [selectedDetail, setSelectedDetail] = useState(null);
    const [loading, setLoading] = useState(false);

    const getJungleWisdom = (seed) => {
        const wisdoms = [
            "Your spirit is legendary, but even Kings need rest to truly conquer the Pride Lands.",
            "Hakuna Matata means no worries, but a strong heart requires wise balance and proper hydration.",
            "The sun will rise again tomorrow, bringing new strength to those who hunt with purpose.",
            "Wisdom comes not from speed, but from knowing when to pause and observe your surroundings.",
            "A lion's roar is most powerful when matched with the quiet focus of a hunter.",
            "Every trail you blaze through the thickest brush brings you closer to your true potential.",
            "Protect your energy like a precious water hole during the longest and driest summer season.",
            "Strength is found in the pride, but character is forged in the silence of solitude.",
            "Listen to the wind whispering through the grass; it holds the secrets of ancient strength.",
            "The horizon is wide, yet every great journey begins with a single, steady, determined step."
        ];
        // Use a simple hash or character sum from the timestamp/ID as index
        const index = [...(seed || '0')].reduce((acc, char) => acc + char.charCodeAt(0), 0) % wisdoms.length;
        return wisdoms[index];
    };

    // Persistence: Check for user in localStorage
    useEffect(() => {
        const savedUser = localStorage.getItem('jungle_user');
        if (savedUser) {
            const parsedUser = JSON.parse(savedUser);
            setUser(parsedUser);
            fetchHistory(parsedUser.user_id);
        }
    }, []);

    const fetchHistory = async (userId) => {
        try {
            const response = await api.getHistory(userId);
            setHistory(response.data);
        } catch (error) {
            console.error("Error fetching history:", error);
        }
    };

    const handleLoginSuccess = (userData) => {
        localStorage.setItem('jungle_user', JSON.stringify(userData));
        setUser(userData);
        fetchHistory(userData.user_id);
    };

    const handleLogout = () => {
        localStorage.removeItem('jungle_user');
        setUser(null);
        setHistory([]);
    };

    const handleImageCapture = async (imageSrc) => {
        if (!user) return;
        setLoading(true);
        setCurrentResult(null);
        try {
            const response = await api.analyzeFace(imageSrc);
            const result = response.data;
            setCurrentResult(result);

            await api.saveHistory(user.name, result, imageSrc, user.user_id);
            await fetchHistory(user.user_id);
        } catch (error) {
            console.error("Error analyzing/saving:", error);
            alert("Rafiki could not read the stars (API Error)");
        } finally {
            setLoading(false);
        }
    };

    if (!user) {
        return (
            <>
                <div className="watermark-jungle" style={{ backgroundImage: "url('/jungle_bg.jpg')" }}></div>
                <div className="py-5">
                    <div className="text-center mb-5">
                        <span className="fs-1 d-block mb-3">🦁</span>
                        <h1 className="fw-bold font-cinzel text-vibrant-gradient">The Wonders of the Jungle</h1>
                    </div>
                    <Auth onLoginSuccess={handleLoginSuccess} />
                </div>
            </>
        );
    }

    return (
        <>
            <div className="watermark-jungle" style={{ backgroundImage: "url('/jungle_bg.jpg')" }}></div>

            <Navbar expand="lg" className="mb-4 pt-4" style={{ backgroundColor: 'transparent' }}>
                <Container>
                    <Navbar.Brand as={Link} to="/" className="d-flex align-items-center" style={{ textDecoration: 'none' }}>
                        <span className="fs-1 me-3">🦁</span>
                        <div className="d-flex flex-column">
                            <span className="fw-bold fs-2 font-cinzel text-vibrant-gradient lh-1">The Wonders of the Jungle</span>
                            <span className="fs-6 font-quicksand text-white opacity-75 mt-1">Gymnast Health & Skill Tracker</span>
                        </div>
                    </Navbar.Brand>
                    <Nav className="ms-auto align-items-center">
                        <div className="text-end me-3 d-none d-md-block">
                            <div className="fw-bold text-cream">Hello, {user.name}</div>
                            <small className="opacity-50 text-cream" style={{ fontSize: '0.8em' }}>{user.about}</small>
                        </div>
                        <Button variant="outline-warning" size="sm" onClick={handleLogout} className="font-cinzel">Logout</Button>
                    </Nav>
                </Container>
            </Navbar>

            <Container className="mb-5 position-relative" style={{ zIndex: 1 }}>
                <div className="d-flex justify-content-center mb-5 gap-4">
                    <NavLink
                        to="/health"
                        className={({ isActive }) => `nav-link-custom ${isActive ? 'active' : ''}`}
                        style={{ textDecoration: 'none' }}
                    >
                        Health Ritual
                    </NavLink>
                    <NavLink
                        to="/gymnastics"
                        className={({ isActive }) => `nav-link-custom ${isActive ? 'active' : ''}`}
                        style={{ textDecoration: 'none' }}
                    >
                        Gymnastics Arena
                    </NavLink>
                </div>

                <Routes>
                    <Route path="/" element={<Navigate to="/health" replace />} />
                    <Route path="/health" element={
                        <Row className="g-5">
                            <Col md={12} lg={5}>
                                <div className="card-dashboard d-flex flex-column align-items-center justify-content-center text-center">
                                    <img src="/simba_cub.png" alt="Guide" className="simba-guide mb-2" style={{ width: '100px', borderRadius: '15px' }} />
                                    <h2 className="mb-3">May I see your face, {user.name}?</h2>

                                    <div className="position-relative mb-4 mt-2">
                                        <div className="frame-tribal p-3" style={{ width: '100%', maxWidth: '350px', display: 'inline-block', overflow: 'visible' }}>
                                            <CameraCapture
                                                onCapture={handleImageCapture}
                                                loading={loading}
                                                hideButton={!!currentResult}
                                            />
                                        </div>

                                        {currentResult && (
                                            <div className="position-absolute top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center">
                                                <div className="glass-overlay-dashboard p-3 text-center d-flex flex-column" style={{ width: '280px', maxHeight: '320px', overflow: 'hidden' }}>
                                                    <div className="mb-2">
                                                        <h4 className="fw-bold mb-0 text-pride-gold" style={{ fontSize: '1.5rem', lineHeight: '1.2' }}>{currentResult.happiness}</h4>
                                                        <small className="text-uppercase text-cream opacity-75">Analysis Result</small>
                                                    </div>

                                                    <div className="glass-scroll px-2 py-1">
                                                        <div className="d-flex justify-content-around w-100 flex-wrap gap-2 mb-3">
                                                            <div className="text-center min-w-50"><span className="d-block fw-bold text-warning" style={{ fontSize: '0.9rem' }}>{currentResult.energy}</span><small className="text-cream opacity-50" style={{ fontSize: '0.65rem' }}>Energy</small></div>
                                                            <div className="text-center min-w-50"><span className="d-block fw-bold text-success" style={{ fontSize: '0.9rem' }}>{currentResult.recovery}</span><small className="text-cream opacity-50" style={{ fontSize: '0.65rem' }}>Recovery</small></div>
                                                            <div className="text-center min-w-50"><span className="d-block fw-bold text-info" style={{ fontSize: '0.9rem' }}>{currentResult.stability}</span><small className="text-cream opacity-50" style={{ fontSize: '0.65rem' }}>Stability</small></div>
                                                            <div className="text-center min-w-50"><span className="d-block fw-bold text-primary" style={{ fontSize: '0.9rem' }}>{currentResult.elasticity}</span><small className="text-cream opacity-50" style={{ fontSize: '0.65rem' }}>Elasticity</small></div>
                                                            <div className="text-center min-w-50"><span className="d-block fw-bold text-danger" style={{ fontSize: '0.9rem' }}>{currentResult.grit}</span><small className="text-cream opacity-50" style={{ fontSize: '0.65rem' }}>Spirit/Grit</small></div>
                                                        </div>
                                                        {currentResult.daily_tip && (
                                                            <div className="mt-2 small opacity-75 font-quicksand italic text-start" style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '8px' }}>
                                                                <span className="text-warning fw-bold">Tip:</span> {currentResult.daily_tip}
                                                            </div>
                                                        )}
                                                    </div>

                                                    <button
                                                        className="btn-lion btn-sm mt-auto w-100 py-2 shadow-sm"
                                                        onClick={() => setCurrentResult(null)}
                                                        style={{ fontSize: '0.9rem', borderRadius: '12px' }}
                                                    >
                                                        RETAKE PHOTO 🔄
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    <p className="font-quicksand opacity-50 mt-4 italic">"{user.about || "Align your spirit, reveal your truth."}"</p>
                                </div>
                            </Col>

                            <Col md={12} lg={7}>
                                <div className="card-dashboard">
                                    {selectedDetail ? (
                                        <div className="detail-view">
                                            <div className="d-flex justify-content-between align-items-center mb-4">
                                                <h2 className="mb-0 text-pride-gold d-flex align-items-center">
                                                    <Activity className="me-2 text-warning" size={28} />
                                                    Details
                                                </h2>
                                                <Button variant="outline-light" size="sm" onClick={() => setSelectedDetail(null)}>Back to History</Button>
                                            </div>
                                            <Row className="g-4">
                                                <Col md={5}>
                                                    <div className={`jungle-photo-frame shadow-lg rounded-4 ${selectedDetail.result.energy && (selectedDetail.result.energy.includes('Simba') || selectedDetail.result.energy.includes('Nala'))
                                                        ? 'frame-energy-high'
                                                        : (selectedDetail.result.energy && (selectedDetail.result.energy.includes('Pumba') || selectedDetail.result.energy.includes('Zazu')))
                                                            ? 'frame-energy-low'
                                                            : 'frame-energy-med'
                                                        }`}>
                                                        <img
                                                            src={selectedDetail.image_path}
                                                            className="w-100"
                                                            style={{ display: 'block' }}
                                                            alt="Historical Insight"
                                                            onError={(e) => {
                                                                e.target.style.display = 'none';
                                                                e.target.parentElement.innerHTML = '<div class="insight-avatar-placeholder" style="font-size: 5rem; padding: 2rem;">🦁</div>';
                                                            }}
                                                        />
                                                    </div>
                                                    <div className="mt-3 text-center opacity-75">
                                                        <div className="fw-bold text-cream">
                                                            {new Intl.DateTimeFormat('en-US', {
                                                                weekday: 'short',
                                                                month: 'short',
                                                                day: 'numeric',
                                                                hour: '2-digit',
                                                                minute: '2-digit'
                                                            }).format(new Date(selectedDetail.timestamp))}
                                                        </div>
                                                    </div>
                                                    <div className="mt-4 p-3 highlight-card rounded-4 border-warning border-opacity-25 shadow-sm" style={{ background: 'rgba(255, 215, 64, 0.05)' }}>
                                                        <div className="small text-pride-gold mb-1 font-cinzel italic" style={{ fontSize: '0.75rem', letterSpacing: '2px' }}>Jungle Wisdom</div>
                                                        <div className="text-white opacity-75" style={{ fontSize: '0.9rem', lineHeight: '1.4' }}>
                                                            {getJungleWisdom(selectedDetail.timestamp || selectedDetail.id)}
                                                        </div>
                                                    </div>
                                                </Col>
                                                <Col md={7}>
                                                    <div className="detail-metrics-grid">
                                                        <div className="p-3 mb-3 glass-card rounded-4 border-start-warning">
                                                            <h4 className="text-pride-gold mb-1">{selectedDetail.result.happiness}</h4>
                                                            <div className="small opacity-50">HEART STATE</div>
                                                        </div>
                                                        <Row className="g-3">
                                                            {[
                                                                { label: 'Energy Level', val: selectedDetail.result.energy || 'Simba-strong', score: 85, icon: '⚡', color: 'text-warning' },
                                                                { label: 'Recovery Score', val: selectedDetail.result.recovery || '70%', score: 70, icon: '🔋', color: 'text-success' },
                                                                { label: 'Balance Stability', val: selectedDetail.result.stability || 'Solid', score: 92, icon: '🎯', color: 'text-info' },
                                                                { label: 'Muscle Elasticity', val: selectedDetail.result.elasticity || 'Springy', score: 88, icon: '📈', color: 'text-primary' },
                                                                { label: 'Spirit & Grit', val: selectedDetail.result.grit || 'Mufasa Core', score: 98, icon: '🔥', color: 'text-danger' }
                                                            ].map(m => (
                                                                <Col xs={12} key={m.label}>
                                                                    <div className="p-2 glass-card rounded-3 d-flex align-items-center gap-3 shadow-sm border border-white border-opacity-10">
                                                                        <div className={`fs-3 ${m.color} bg-black bg-opacity-20 rounded-pill d-flex align-items-center justify-content-center`} style={{ width: '45px', height: '45px' }}>{m.icon}</div>
                                                                        <div className="flex-grow-1">
                                                                            <div className="d-flex justify-content-between align-items-start">
                                                                                <div className="small opacity-50" style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '1px' }}>{m.label}</div>
                                                                                <div className={`fw-bold ${m.color}`} style={{ fontSize: '1.2rem' }}>{m.score}</div>
                                                                            </div>
                                                                            <div className="fw-bold fs-5 text-white">{m.val}</div>
                                                                        </div>
                                                                    </div>
                                                                </Col>
                                                            ))}
                                                        </Row>
                                                        {selectedDetail.result.daily_tip && (
                                                            <div className="mt-4 p-3 rounded-4" style={{ background: 'rgba(255, 215, 64, 0.05)', borderLeft: '4px solid var(--pride-gold)' }}>
                                                                <div className="fw-bold text-pride-gold mb-1">RAFIKI'S WISDOM</div>
                                                                <div className="opacity-75 italic">"{selectedDetail.result.daily_tip}"</div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </Col>
                                            </Row>
                                        </div>
                                    ) : (
                                        <LatestInsights history={history} onSelectDetail={setSelectedDetail} />
                                    )}
                                </div>
                            </Col>
                        </Row>
                    } />
                    <Route path="/gymnastics" element={<GymnasticsScoring />} />
                </Routes>

                <div className="mt-5 text-center opacity-50 font-cinzel text-small">
                    The Wonders of the Jungle • Private Pride Access • Gymnast Analyzer
                </div>
            </Container >
        </>
    )
}

export default App
