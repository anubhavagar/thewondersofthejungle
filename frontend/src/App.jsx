import { useState, useEffect } from 'react'
import { Container, Row, Col, Navbar, Tab, Tabs, Button, Nav } from 'react-bootstrap';
import CameraCapture from './components/CameraCapture';
import GoogleFitConnect from './components/GoogleFitConnect';
import HealthTable from './components/HealthTable';
import GymnasticsScoring from './components/GymnasticsScoring';
import Auth from './components/Auth';
import api from './services/api';

function App() {
    const [user, setUser] = useState(null);
    const [history, setHistory] = useState([]);
    const [currentResult, setCurrentResult] = useState(null);
    const [googleFitData, setGoogleFitData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [key, setKey] = useState('health');

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
                    <Navbar.Brand href="#" className="d-flex align-items-center">
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
                <Tabs
                    id="app-tabs"
                    activeKey={key}
                    onSelect={(k) => setKey(k)}
                    className="mb-5 justify-content-center border-0"
                >
                    <Tab eventKey="health" title={<span className="nav-link-custom">Health Ritual</span>}>

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
                                            />
                                        </div>

                                        {currentResult && (
                                            <div className="position-absolute top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center">
                                                <div className="glass-overlay-dashboard p-4 text-center d-flex flex-column justify-content-center" style={{ width: '260px', height: '260px' }}>
                                                    <h3 className="display-4 fw-bold mb-0">{currentResult.happiness}</h3>
                                                    <small className="text-uppercase text-cream opacity-75 mb-2">Happiness</small>
                                                    <div className="d-flex justify-content-around w-100 mt-2">
                                                        <div><span className="d-block fw-bold">{currentResult.energy}</span><small className="text-cream opacity-50" style={{ fontSize: '0.7em' }}>Energy</small></div>
                                                        <div><span className="d-block fw-bold">{currentResult.stress}</span><small className="text-cream opacity-50" style={{ fontSize: '0.7em' }}>Stress</small></div>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    <p className="font-quicksand opacity-50 mt-4 italic">"{user.about || "Align your spirit, reveal your truth."}"</p>
                                </div>
                            </Col>

                            <Col md={12} lg={7}>
                                <div className="card-dashboard">
                                    <div className="d-flex align-items-center mb-4 border-bottom border-secondary pb-3">
                                        <h2 className="mb-0">Your History</h2>
                                        <div className="ms-auto opacity-50 font-cinzel text-small">Private Jungle Records</div>
                                    </div>

                                    <div style={{ maxHeight: '600px', overflowY: 'auto', paddingRight: '10px' }}>
                                        {history.length > 0 ? (
                                            <HealthTable history={history} />
                                        ) : (
                                            <div className="text-center py-5 opacity-50">
                                                <h4>No records found in the stars.</h4>
                                                <p>Begin the ritual to create your journey.</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </Col>
                        </Row>
                    </Tab>

                    <Tab eventKey="gymnastics" title={<span className="nav-link-custom">Gymnastics Arena</span>}>
                        <GymnasticsScoring />
                    </Tab>
                </Tabs>

                <div className="mt-5 text-center opacity-50 font-cinzel text-small">
                    The Wonders of the Jungle • Private Pride Access • Gymnast Analyzer
                </div>
            </Container >
        </>
    )
}

export default App
