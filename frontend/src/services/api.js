import axios from 'axios';

const API_URL = '/api';

const api = {
    analyzeFace: (imageBase64) => {
        // In a real app, send MultiPart form data or base64 json
        // Here we'll just send a JSON payload for simplicity in simulation
        return axios.post(`${API_URL}/analyze/face`, { image: imageBase64 });
    },
    analyzeWellness: (healthData) => {
        return axios.post(`${API_URL}/analyze/wellness`, healthData);
    },
    getHistory: (userId = null) => {
        const url = userId ? `${API_URL}/history?user_id=${userId}` : `${API_URL}/history`;
        return axios.get(url);
    },
    saveHistory: (name, result, image = null, userId = null) => {
        return axios.post(`${API_URL}/history`, { name, result, image, user_id: userId });
    },
    analyzeGymnastics: (mediaData, mediaType = 'image', category = 'Senior') => {
        return axios.post(`${API_URL}/analyze/gymnastics`, {
            media_data: mediaData,
            media_type: mediaType,
            category: category
        });
    },
    requestOtp: (mobile) => {
        return axios.post(`${API_URL}/auth/request-otp`, { mobile });
    },
    verifyOtp: (mobile, otp, name = null, about = null) => {
        return axios.post(`${API_URL}/auth/verify-otp`, { mobile, otp, name, about });
    }
};

export default api;
