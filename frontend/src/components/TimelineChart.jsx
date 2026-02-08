import React, { useState, useRef } from 'react';

const TimelineChart = ({ data, onSeek, onHover, videoSrc }) => {
    if (!data || data.length === 0) return null;

    const [hoverData, setHoverData] = useState(null);
    const svgRef = useRef(null);
    const tooltipVideoRef = useRef(null);

    // Sync tooltip video time
    React.useEffect(() => {
        if (hoverData && tooltipVideoRef.current && isFinite(hoverData.time)) {
            tooltipVideoRef.current.currentTime = hoverData.time;
        }
    }, [hoverData]);

    // SVG Dimensions
    const width = 600;
    const height = 200;
    const padding = 20;

    // Scales
    const maxTime = Math.max(...data.map(d => d.time));
    const maxAngle = 180;

    // Helper to map values to coordinates
    const xScale = (time) => ((time / maxTime) * (width - 2 * padding)) + padding;
    const yScale = (angle) => height - (((angle / maxAngle) * (height - 2 * padding)) + padding);

    // Generate Path
    const points = data.map(d => `${xScale(d.time)},${yScale(d.angle)}`).join(' ');

    const handleMouseMove = (e) => {
        if (!svgRef.current) return;

        const svgRect = svgRef.current.getBoundingClientRect();
        const mouseX = e.clientX - svgRect.left;

        // Find closest data point based on X coordinate
        // Inverse X scale: time = ((x - padding) / (width - 2*padding)) * maxTime
        const timeAtMouse = ((mouseX - padding) / (width - 2 * padding)) * maxTime;

        // Find nearest data point
        const closest = data.reduce((prev, curr) => {
            return (Math.abs(curr.time - timeAtMouse) < Math.abs(prev.time - timeAtMouse) ? curr : prev);
        });

        setHoverData(closest);

        if (onHover) {
            onHover(closest.time);
        }
    };

    const handleMouseLeave = () => {
        setHoverData(null);
    };

    const handleClick = () => {
        if (hoverData && onSeek) {
            onSeek(hoverData.time);
        }
    };

    return (
        <div className="timeline-chart-container mb-4 p-3 rounded-3 shadow-sm" style={{ backgroundColor: '#fff', border: '2px solid var(--junior-cyan)' }}>
            <h5 className="text-center fw-bold mb-3" style={{ color: 'var(--junior-pink)', fontFamily: 'Fredoka One' }}>📈 Extension Angle Timeline</h5>

            <div style={{ position: 'relative', cursor: 'pointer' }}>
                <svg
                    ref={svgRef}
                    width="100%"
                    height="200"
                    viewBox={`0 0 ${width} ${height}`}
                    style={{ overflow: 'visible' }}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    onClick={handleClick}
                >
                    {/* Axes */}
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#ccc" strokeWidth="2" />
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#ccc" strokeWidth="2" />

                    {/* Data Path */}
                    <polyline
                        points={points}
                        fill="none"
                        stroke="var(--junior-cyan)"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />

                    {/* Hover Indicator Line & Point */}
                    {hoverData && (
                        <g>
                            <line
                                x1={xScale(hoverData.time)}
                                y1={padding}
                                x2={xScale(hoverData.time)}
                                y2={height - padding}
                                stroke="#ff9800"
                                strokeWidth="2"
                                strokeDasharray="5,5"
                            />
                            <circle
                                cx={xScale(hoverData.time)}
                                cy={yScale(hoverData.angle)}
                                r="6"
                                fill="#ff9800"
                            />
                        </g>
                    )}
                </svg>

                {/* Floating Tooltip with Video */}
                {hoverData && (
                    <div
                        style={{
                            position: 'absolute',
                            left: `${(xScale(hoverData.time) / width) * 100}%`,
                            top: '-160px', /* Adjusted to make room for video */
                            transform: 'translateX(-50%)',
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '2px solid var(--junior-pink)',
                            color: '#333',
                            padding: '10px',
                            borderRadius: '12px',
                            pointerEvents: 'none',
                            fontSize: '0.9rem',
                            whiteSpace: 'nowrap',
                            zIndex: 100, /* Ensure it's on top */
                            boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
                        }}
                    >
                        {videoSrc && (
                            <div className="mb-2 rounded overflow-hidden" style={{ width: '160px', height: '90px', backgroundColor: '#000' }}>
                                <video
                                    ref={tooltipVideoRef}
                                    src={videoSrc}
                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                    muted
                                />
                            </div>
                        )}
                        <div><strong>Time:</strong> {hoverData.time}s</div>
                        <div><strong>Angle:</strong> {hoverData.angle}°</div>
                        <div><strong>Stability:</strong> {hoverData.stability}%</div>
                        <div className="small text-muted mt-1">Click to Seek ⏩</div>
                    </div>
                )}
            </div>

            <div className="d-flex justify-content-between text-muted small mt-2">
                <span>0s</span>
                <span>Time (Seconds)</span>
                <span>{maxTime.toFixed(1)}s</span>
            </div>
        </div>
    );
};

export default TimelineChart;
