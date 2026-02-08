import React, { useState } from 'react';
import { Table, Button } from 'react-bootstrap';

const HealthTable = ({ history }) => {
    return (
        <div className="mt-2 text-cream">
            <Table responsive className="table-dashboard" borderless>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Name</th>
                        <th>Photo</th>
                        <th>Happiness</th>
                        <th>Energy</th>
                        <th>Hydration</th>
                    </tr>
                </thead>
                <tbody>
                    {history.map((entry) => (
                        <tr key={entry.id} className="text-nowrap">
                            <td>{entry.timestamp}</td>
                            <td className="fw-bold text-pride-gold">{entry.name}</td>
                            <td>
                                {entry.image_path ? (
                                    <img
                                        src={`http://localhost:8000${entry.image_path}`}
                                        alt="User"
                                        className="avatar-circle"
                                    />
                                ) : <span style={{ fontSize: '24px' }}>🦁</span>}
                            </td>
                            <td>{entry.result.happiness}</td>
                            <td>{entry.result.energy}</td>
                            <td>{entry.result.hydration}</td>
                        </tr>
                    ))}
                </tbody>
            </Table>
        </div>
    );
};

export default HealthTable;
