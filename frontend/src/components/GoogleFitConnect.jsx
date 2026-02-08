import React from 'react';
import { Button, Card } from 'react-bootstrap';

const GoogleFitConnect = ({ onConnect, data, loading }) => {
    return (
        <div className="d-flex flex-column align-items-center justify-content-center h-100">
            {!data ? (
                <div className="text-center">
                    <p className="mb-4">Connect to the Great Circle to see your prowess!</p>
                    <Button variant="warning" className="btn-gold btn-lg" onClick={onConnect} disabled={loading}>
                        {loading ? 'Connecting...' : 'Connect Google Fit 🔗'}
                    </Button>
                </div>
            ) : (
                <div className="w-100">
                    <Card className="mb-3 border-0 bg-transparent">
                        <Card.Body>
                            <h5 className="card-title text-success">Connected! ✅</h5>
                            <p className="card-text"><strong>Analysis:</strong> {data.advice}</p>
                            <p className="card-text text-muted">Status: {data.status}</p>
                        </Card.Body>
                    </Card>
                </div>
            )}
        </div>
    );
};

export default GoogleFitConnect;
