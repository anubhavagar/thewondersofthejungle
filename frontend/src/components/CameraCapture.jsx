import React, { useRef, useState, useCallback } from 'react';
import { Button } from 'react-bootstrap';

const CameraCapture = ({ onCapture, loading }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoRef.current.srcObject = mediaStream;
            setStream(mediaStream);
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access camera. Make sure you have permissions!");
        }
    };

    const takePhoto = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            onCapture(dataUrl);
        }
    };

    return (
        <div className="d-flex flex-column align-items-center">
            <div className="position-relative mb-3" style={{ width: '100%', maxWidth: '400px', aspectRatio: '4/3', backgroundColor: '#333', borderRadius: '10px', overflow: 'hidden' }}>
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>

            {!stream ? (
                <Button variant="primary" className="btn-lion btn-lg" onClick={startCamera}>
                    Start Camera 🎥
                </Button>
            ) : (
                <Button variant="success" className="btn-lion btn-lg" onClick={takePhoto} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Take Photo 📸'}
                </Button>
            )}
        </div>
    );
};

export default CameraCapture;
