import React from 'react';
import { Row, Col, Badge } from 'react-bootstrap';
import { Smile, Zap, Droplet, Clock, User, Eye, Battery, Lightbulb, Target, Activity, Flame } from 'lucide-react';

const LatestInsights = ({ history, onSelectDetail }) => {
    // Get latest 3 records
    const latest = history.slice(0, 3);

    if (latest.length === 0) {
        return (
            <div className="text-center py-5 opacity-50">
                <h4>No records found in the stars.</h4>
                <p>Begin the ritual to create your journey.</p>
            </div>
        );
    }

    return (
        <div className="latest-insights-container">
            <h2 className="mb-3 d-flex align-items-center">
                <Zap size={24} className="me-2 text-warning" />
                Private Jungle Records <span style={{ fontSize: '0.6em', color: 'white', opacity: 0.7 }}>(recent 3)</span>
            </h2>
            <hr className="mb-4" style={{ borderTop: '2px solid rgba(255, 215, 64, 0.3)', opacity: 1 }} />
            <div className="d-flex flex-column gap-3">
                {latest.map((entry, idx) => (
                    <div
                        key={entry.id}
                        className={`insight-card p-3 ${idx === 0 ? 'highlight-card' : ''}`}
                        onClick={() => onSelectDetail && onSelectDetail(entry)}
                    >
                        <Row className="align-items-center g-3">
                            <Col xs="auto">
                                <div className="position-relative">
                                    <div
                                        className="insight-avatar-container jungle-photo-frame shadow-sm"
                                        title="Click to view details"
                                        style={{ transition: 'transform 0.2s' }}
                                        onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                        onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
                                    >
                                        {entry.image_path ? (
                                            <img
                                                src={entry.image_path}
                                                alt="User"
                                                className="insight-avatar"
                                                onError={(e) => {
                                                    e.target.style.display = 'none';
                                                    e.target.parentElement.innerHTML = '<div class="insight-avatar-placeholder">🦁</div>';
                                                }}
                                            />
                                        ) : (
                                            <div className="insight-avatar-placeholder">🦁</div>
                                        )}
                                    </div>
                                    {idx === 0 && (
                                        <Badge
                                            bg="warning"
                                            className="position-absolute top-0 start-0 translate-middle rounded-pill shadow-sm"
                                            style={{ zIndex: 10 }}
                                        >
                                            New
                                        </Badge>
                                    )}
                                </div>
                            </Col>
                            <Col>
                                <div className="d-flex justify-content-between align-items-start mb-1">
                                    <h5 className="mb-0 text-pride-gold">{entry.result.happiness}</h5>
                                    <small className="opacity-50 d-flex align-items-center">
                                        <Clock size={12} className="me-1" /> {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </small>
                                </div>
                                <div className="d-flex gap-3 mt-2">
                                    <div className="metric-box hover-scale" title="Energy Level">
                                        <Zap size={14} className="text-warning me-1" />
                                        <span>{entry.result.energy}</span>
                                    </div>
                                    <div className="metric-box hover-scale" title="Recovery Score">
                                        <Battery size={14} className="text-success me-1" />
                                        <span>{entry.result.recovery || '70%'}</span>
                                    </div>
                                    <div className="metric-box hover-scale" title="Balance Stability">
                                        <Target size={14} className="text-info me-1" />
                                        <span>{entry.result.stability || 'Solid'}</span>
                                    </div>
                                    <div className="metric-box hover-scale" title="Muscle Elasticity">
                                        <Activity size={14} className="text-primary me-1" />
                                        <span>{entry.result.elasticity || 'Springy'}</span>
                                    </div>
                                    <div className="metric-box hover-scale" title="Spirit & Grit">
                                        <Flame size={14} className="text-danger me-1" />
                                        <span>{entry.result.grit || 'Mufasa Core'}</span>
                                    </div>
                                </div>
                                {idx === 0 && entry.result.daily_tip && (
                                    <div className="mt-3 p-2 rounded-3" style={{ background: 'rgba(255, 215, 64, 0.1)', borderLeft: '3px solid var(--pride-gold)' }}>
                                        <div className="small fw-bold text-pride-gold d-flex align-items-center">
                                            <Lightbulb size={12} className="me-1" /> LION TIP
                                        </div>
                                        <div className="small opacity-75 mt-1 italic">{entry.result.daily_tip}</div>
                                    </div>
                                )}
                            </Col>
                        </Row>
                    </div>
                ))}
            </div>
            {history.length > 3 && (
                <div className="text-center mt-4 opacity-50 small font-cinzel">
                    + {history.length - 3} more records in the archives
                </div>
            )}
        </div>
    );
};

export default LatestInsights;
