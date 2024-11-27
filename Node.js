const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.post('/api/search', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    // Example: Using Pexels API
    const API_KEY = 'YOUR_PEXELS_API_KEY'; // Replace with your API key
    const response = await fetch(`https://api.pexels.com/v1/search?query=example`, {
        headers: {
            Authorization: API_KEY
        }
    });

    const data = await response.json();
    res.json(data);
});

app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
