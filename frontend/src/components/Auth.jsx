import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Alert, Card } from 'react-bootstrap';
import api from '../services/api';

const Auth = ({ onLoginSuccess }) => {
    const [step, setStep] = useState(1); // 1: Mobile, 2: OTP, 3: Registration
    const [formData, setFormData] = useState({
        mobile: '',
        otp: '',
        name: '',
        about: ''
    });
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleRequestOtp = async (e) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        try {
            await api.requestOtp(formData.mobile);
            setStep(2);
        } catch (err) {
            setError(err.response?.data?.detail || "Failed to send OTP. Try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleVerifyOtp = async (e) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        try {
            const response = await api.verifyOtp(formData.mobile, formData.otp, formData.name, formData.about);
            if (response.data.is_new_user && step !== 3) {
                setStep(3);
            } else {
                onLoginSuccess(response.data);
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Invalid OTP. Check the console!");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container className="d-flex align-items-center justify-content-center" style={{ minHeight: '80vh' }}>
            <Card className="card-dashboard p-4" style={{ width: '450px', maxWidth: '100%' }}>
                <h2 className="text-center mb-4">
                    {step === 1 && "Jungle Entrance"}
                    {step === 2 && "Enter the Secret Code"}
                    {step === 3 && "Tell us about yourself"}
                </h2>

                {error && <Alert variant="danger" className="bg-danger text-white border-0 opacity-75">{error}</Alert>}

                {step === 1 && (
                    <Form onSubmit={handleRequestOtp}>
                        <Form.Group className="mb-4">
                            <Form.Label>Mobile Number</Form.Label>
                            <Form.Control
                                className="input-dashboard"
                                type="tel"
                                name="mobile"
                                placeholder="Ex: 9876543210"
                                value={formData.mobile}
                                onChange={handleInputChange}
                                required
                            />
                        </Form.Group>
                        <Button
                            variant="warning"
                            type="submit"
                            className="w-100 fw-bold"
                            disabled={loading}
                            style={{ background: 'linear-gradient(45deg, #FFB300, #FF6F00)', border: 'none' }}
                        >
                            {loading ? "Sending Signal..." : "Send OTP"}
                        </Button>
                    </Form>
                )}

                {step === 2 && (
                    <Form onSubmit={handleVerifyOtp}>
                        <Form.Group className="mb-4">
                            <Form.Label>Enter 4-digit Code</Form.Label>
                            <Form.Control
                                className="input-dashboard text-center fs-2"
                                type="text"
                                name="otp"
                                placeholder="0000"
                                maxLength="4"
                                value={formData.otp}
                                onChange={handleInputChange}
                                required
                            />
                            <Form.Text className="text-cream opacity-50">
                                Check the backend console for your roar!
                            </Form.Text>
                        </Form.Group>
                        <Button
                            variant="warning"
                            type="submit"
                            className="w-100 fw-bold"
                            disabled={loading}
                            style={{ background: 'linear-gradient(45deg, #FFB300, #FF6F00)', border: 'none' }}
                        >
                            {loading ? "Verifying..." : "Enter Jungle"}
                        </Button>
                        <div className="text-center mt-3">
                            <small className="opacity-75 cursor-pointer" onClick={() => setStep(1)} style={{ textDecoration: 'underline' }}>
                                Change number
                            </small>
                        </div>
                    </Form>
                )}

                {step === 3 && (
                    <Form onSubmit={handleVerifyOtp}>
                        <Form.Group className="mb-3">
                            <Form.Label>Name</Form.Label>
                            <Form.Control
                                className="input-dashboard"
                                type="text"
                                name="name"
                                placeholder="Simba"
                                value={formData.name}
                                onChange={handleInputChange}
                                required
                            />
                        </Form.Group>
                        <Form.Group className="mb-4">
                            <Form.Label>About Self (1 Line)</Form.Label>
                            <Form.Control
                                className="input-dashboard"
                                type="text"
                                name="about"
                                placeholder="I just can't wait to be king!"
                                value={formData.about}
                                onChange={handleInputChange}
                                required
                            />
                        </Form.Group>
                        <Button
                            variant="warning"
                            type="submit"
                            className="w-100 fw-bold"
                            disabled={loading}
                            style={{ background: 'linear-gradient(45deg, #FFB300, #FF6F00)', border: 'none' }}
                        >
                            {loading ? "Saving Profile..." : "Complete Ritual"}
                        </Button>
                    </Form>
                )}
            </Card>
        </Container>
    );
};

export default Auth;
