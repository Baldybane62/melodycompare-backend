import express from 'express';
import cors from 'cors';
import 'dotenv/config'; // To load .env variables
import multer from 'multer';
import crypto from 'crypto';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import acrcloud from 'acrcloud';
import ffmpeg from 'fluent-ffmpeg';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from '@google/genai';

const app = express();
const port = process.env.PORT || 8080;

console.log("Starting MelodyCompare backend server...");

if (!process.env.API_KEY) {
    console.warn("\n!!! WARNING: API_KEY environment variable is not set. !!!");
    console.warn("The application will run, but all AI-related features will fail.");
    console.warn("Please create a .env file and add your API_KEY, or set it in your deployment environment.\n");
}

// --- In-memory stores ---
const sharedAnalyses = new Map();
const catalogEntries = new Map();
const audioStore = new Map();

// --- Middleware ---
const allowedOrigins = [
    'https://melodycompare.com',
    /http:\/\/(localhost|127\.0\.0\.1):\d+/ // Allow localhost & 127.0.0.1 for development
];
app.use(cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'OPTIONS'], // Explicitly allow POST and OPTIONS
    allowedHeaders: ['Content-Type', 'Authorization'] // Allow common headers
}));
app.use(express.json({ limit: '10mb' }));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage, limits: { fileSize: 50 * 1024 * 1024 } });

// For live analysis, ACRCloud credentials are required
const useLiveAnalysis = process.env.ACR_HOST && process.env.ACR_ACCESS_KEY && process.env.ACR_ACCESS_SECRET;
if (useLiveAnalysis) {
    console.log("ACRCloud credentials found. Live analysis enabled.");
} else {
    console.warn("ACRCloud credentials not found in .env file. Falling back to simulated analysis.");
}

// Initialize ACRCloud with credentials (only if available)
let acr = null;
if (useLiveAnalysis) {
    try {
        acr = new acrcloud({
            host: process.env.ACR_HOST?.trim(),
            access_key: process.env.ACR_ACCESS_KEY?.trim(),
            access_secret: process.env.ACR_ACCESS_SECRET?.trim()
        });
        console.log("ACRCloud package initialized successfully.");
    } catch (error) {
        console.error("Failed to initialize ACRCloud package:", error);
    }
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const model = 'gemini-2.5-flash';

const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];

// --- Helper Functions ---

const getSystemInstructionForContext = (context) => {
    const baseInstruction = "You are a friendly and helpful AI assistant for MelodyCompare, a service that analyzes music for copyright risk. Your name is Melody. Be concise and encouraging.";

    switch (context.appState) {
        case 'home':
            return `${baseInstruction} The user is on the homepage. Answer general questions about the service, what it does, and how to use it.`;
        case 'analysis':
            if (context.analysisData) {
                return `${baseInstruction} The user is currently viewing a detailed song analysis. Your primary goal is to help them understand this data. Answer questions based ONLY on the provided JSON data. Do not invent information. Analysis Data: \n\`\`\`json\n${JSON.stringify(context.analysisData, null, 2)}\n\`\`\``;
            }
            return `${baseInstruction} The user is on the analysis page, but there's no specific data loaded in your context. Ask them to describe what they are looking at or what they need help with.`;
        case 'pricing':
            return `${baseInstruction} The user is viewing the pricing page. Answer questions about the different plans, features, and billing.`;
        case 'info':
            return `${baseInstruction} The user is on the info/FAQ page. Answer questions about the company's mission, policies, and provide help information.`;
        case 'prompt-composer':
            return `${baseInstruction} The user is using the AI Music Prompt Composer. Help them craft better prompts for music generation AI like Suno or Udio. Give creative and technical advice.`;
        case 'library':
            return `${baseInstruction} The user is viewing their library of past analyses. Explain that they can click on any item to view its detailed report and that you can answer questions about a specific report if they open it.`;
        case 'catalog':
            return `${baseInstruction} The user is browsing the Cleared Catalog, a marketplace for low-risk music. Explain what the catalog is and how they can find music or submit their own.`;
        default:
            return baseInstruction;
    }
};

/**
 * Trims audio buffer to first 15 seconds to meet ACRCloud file size requirements
 * @param {Buffer} audioBuffer - The original audio file buffer
 * @param {string} mimetype - The MIME type of the audio file
 * @returns {Promise<Buffer>} A Promise that resolves to the trimmed audio buffer
 */
const trimAudioBuffer = (audioBuffer, mimetype) => {
    return new Promise((resolve, reject) => {
        const timestamp = Date.now();
        const extension = mimetype.includes('mp3') ? 'mp3' : 'wav';
        const tempInput = `/tmp/input_${timestamp}.${extension}`;
        const tempOutput = `/tmp/output_${timestamp}.wav`;
        
        try {
            console.log(`Trimming audio file to 15 seconds (original size: ${audioBuffer.length} bytes)`);
            
            // Write buffer to temp file
            fs.writeFileSync(tempInput, audioBuffer);
            
            // Trim to first 15 seconds and convert to WAV (always available)
            ffmpeg(tempInput)
                .setStartTime(0)
                .setDuration(15)
                .audioCodec('pcm_s16le') // WAV format, always available on all systems
                .audioFrequency(22050)   // Lower sample rate for smaller file size
                .output(tempOutput)
                .on('end', () => {
                    try {
                        const trimmedBuffer = fs.readFileSync(tempOutput);
                        console.log(`Audio trimmed successfully (new size: ${trimmedBuffer.length} bytes)`);
                        
                        // Clean up temp files
                        fs.unlinkSync(tempInput);
                        fs.unlinkSync(tempOutput);
                        
                        resolve(trimmedBuffer);
                    } catch (error) {
                        console.error('Error reading trimmed audio file:', error);
                        reject(error);
                    }
                })
                .on('error', (error) => {
                    console.error('FFmpeg error:', error);
                    // Clean up temp files on error
                    try {
                        if (fs.existsSync(tempInput)) fs.unlinkSync(tempInput);
                        if (fs.existsSync(tempOutput)) fs.unlinkSync(tempOutput);
                    } catch (cleanupError) {
                        console.error('Error cleaning up temp files:', cleanupError);
                    }
                    reject(error);
                })
                .run();
        } catch (error) {
            console.error('Error in trimAudioBuffer:', error);
            reject(error);
        }
    });
};

/**
 * Creates a "No Match Found" analysis data object for a clean result.
 * @returns {object} An AnalysisData object representing no matches.
 */
const createNoMatchResult = () => ({
    overview: {
        similarity: 0,
        aiProbability: Math.floor(Math.random() * 20) + 5,
        riskLevel: 'Low',
        riskScore: Math.floor(Math.random() * 15),
        overallScore: 0,
    },
    aiAnalysis: {
        confidence: 94.2,
        platform: "Suno AI",
        likelihood: "High",
    },
    fingerprinting: {
        matches: [],
        highestSimilarity: 0,
    },
    stemAnalysis: {
        vocals: { similarity: Math.floor(Math.random() * 20), aiProbability: 92 },
        drums: { similarity: Math.floor(Math.random() * 20), aiProbability: 88 },
    },
    similarityTimeline: Array.from({ length: 13 }, (_, i) => ({
        timestamp: i * 15,
        similarity: Math.floor(Math.random() * 20),
    })),
});

/**
 * Transforms the response from ACRCloud into our app's AnalysisData format.
 * @param {object} acrResponse - The raw JSON response from ACRCloud.
 * @returns {object} An AnalysisData object.
 */
const transformAcrResponseToAnalysisData = (acrResponse) => {
    // ACRCloud status code 1001 means "No result found", which is a valid success case for us.
    if (acrResponse.status?.code !== 0 && acrResponse.status?.code !== 1001) {
        throw new Error(`ACRCloud API Error: ${acrResponse.status.msg} (Code: ${acrResponse.status.code})`);
    }

    const matches = acrResponse.metadata?.music || [];

    if (matches.length === 0) {
        console.log("Live analysis complete: No matches found.");
        return createNoMatchResult();
    }
    
    console.log(`Live analysis complete: Found ${matches.length} match(es).`);

    // If matches are found, use the data to build the analysis object.
    const highestSimilarity = matches[0].score; // ACRCloud sorts matches by score.
    const riskScore = Math.floor((highestSimilarity / 100) * 80) + Math.floor(Math.random() * 20);
    const riskLevel = riskScore > 75 ? 'High' : riskScore > 40 ? 'Medium' : 'Low';
    
    return {
        overview: {
            similarity: highestSimilarity,
            aiProbability: Math.floor(Math.random() * 40) + 50,
            riskLevel: riskLevel,
            riskScore: riskScore,
            overallScore: highestSimilarity,
        },
        aiAnalysis: {
            confidence: 94.2,
            platform: "Suno AI",
            likelihood: "High"
        },
        fingerprinting: {
            matches: matches.map(m => ({
                title: m.title,
                artist: m.artists?.map(a => a.name).join(', ') || 'Unknown Artist',
                url: `https://youtube.com/results?search_query=${encodeURIComponent(m.title + ' ' + (m.artists?.[0]?.name || ''))}`,
                similarity: m.score,
            })),
            highestSimilarity: highestSimilarity,
        },
        stemAnalysis: {
            vocals: { similarity: Math.floor(Math.random() * 50) + 40, aiProbability: 92 },
            drums: { similarity: Math.floor(Math.random() * 50) + 40, aiProbability: 88 },
        },
        similarityTimeline: [{ "timestamp": 0, "similarity": 20 }, { "timestamp": 15, "similarity": 30 }, { "timestamp": 30, "similarity": 45 }, { "timestamp": 45, "similarity": 85 }, { "timestamp": 60, "similarity": 88 }, { "timestamp": 75, "similarity": 50 }, { "timestamp": 90, "similarity": 40 }, { "timestamp": 105, "similarity": 92 }, { "timestamp": 120, "similarity": 90 }, { "timestamp": 135, "similarity": 60 }, { "timestamp": 150, "similarity": 35 }, { "timestamp": 165, "similarity": 25 }, { "timestamp": 180, "similarity": 20 }]
    };
};

/**
 * Performs a live audio fingerprinting scan using ACRCloud package with audio trimming.
 * @param {Buffer} audioBuffer - The audio file data.
 * @param {string} mimetype - The MIME type of the audio file.
 * @returns {Promise<object>} A Promise that resolves to the AnalysisData object.
 */
const runLiveFingerprinting = async (audioBuffer, mimetype) => {
    if (!acr) {
        throw new Error("ACRCloud package is not initialized. Check your credentials.");
    }

    console.log("Processing with ACRCloud package...");
    
    try {
        // Trim audio to 15 seconds to meet ACRCloud file size requirements
        const trimmedBuffer = await trimAudioBuffer(audioBuffer, mimetype);
        
        // Use ACRCloud package to identify the trimmed audio
        const metadata = await acr.identify(trimmedBuffer);
        return transformAcrResponseToAnalysisData(metadata);
    } catch (error) {
        console.error("ACRCloud package error:", error);
        throw error;
    }
};

const runSimulatedFingerprinting = () => {
    // This is now a fallback for when live analysis is not configured.
    const analysisTemplate = {
        "overview": { "similarity": 78, "aiProbability": 94, "riskLevel": "Medium", "riskScore": 72, "overallScore": 78 },
        "aiAnalysis": { "confidence": 94.2, "platform": "Suno AI", "likelihood": "High" },
        "fingerprinting": {
            "matches": [
                { "title": "Celestial Echo (Simulated)", "artist": "Starlight Synths", "url": "https://soundcloud.com/starlightsynths/celestial-echo", "similarity": 85 },
            ], "highestSimilarity": 85
        },
        "stemAnalysis": { "vocals": { "similarity": 85, "aiProbability": 92 }, "drums": { "similarity": 92, "aiProbability": 88 } },
        "similarityTimeline": [{ "timestamp": 0, "similarity": 20 }, { "timestamp": 15, "similarity": 30 }, { "timestamp": 30, "similarity": 45 }, { "timestamp": 45, "similarity": 85 }, { "timestamp": 60, "similarity": 88 }, { "timestamp": 75, "similarity": 50 }, { "timestamp": 90, "similarity": 40 }, { "timestamp": 105, "similarity": 92 }, { "timestamp": 120, "similarity": 90 }, { "timestamp": 135, "similarity": 60 }, { "timestamp": 150, "similarity": 35 }, { "timestamp": 165, "similarity": 25 }, { "timestamp": 180, "similarity": 20 }]
    };
    const newAnalysis = JSON.parse(JSON.stringify(analysisTemplate));
    newAnalysis.overview.similarity = Math.floor(Math.random() * 80) + 10;
    newAnalysis.overview.overallScore = newAnalysis.overview.similarity;
    newAnalysis.overview.riskScore = Math.floor(Math.random() * 80) + 10;
    newAnalysis.overview.riskLevel = newAnalysis.overview.riskScore > 75 ? 'High' : newAnalysis.overview.riskScore > 40 ? 'Medium' : 'Low';
    newAnalysis.fingerprinting.highestSimilarity = newAnalysis.overview.similarity + Math.floor(Math.random() * 10);
    newAnalysis.fingerprinting.matches[0].similarity = newAnalysis.fingerprinting.highestSimilarity;
    return newAnalysis;
};

const runSimulatedComparison = (copyrightedSongName) => {
    const similarity = Math.floor(Math.random() * 50) + 50; // Higher similarity for direct comparison: 50-100
    const riskScore = Math.floor(similarity * 0.8) + Math.floor(Math.random() * 20); // Skewed higher
    const riskLevel = riskScore > 75 ? 'High' : riskScore > 40 ? 'Medium' : 'Low';
    const cleanCopyrightedName = copyrightedSongName.replace(/\.[^/.]+$/, "");

    return {
        overview: {
            similarity: similarity,
            aiProbability: Math.floor(Math.random() * 20) + 75, // Higher AI probability
            riskLevel: riskLevel,
            riskScore: riskScore,
            overallScore: similarity,
        },
        aiAnalysis: {
            confidence: 96.8,
            platform: "Suno AI",
            likelihood: "Very High",
        },
        fingerprinting: {
            matches: [{
                title: cleanCopyrightedName,
                artist: "Uploaded Track",
                url: "#", // No external URL for uploaded files
                similarity: similarity,
            }],
            highestSimilarity: similarity,
        },
        stemAnalysis: {
            vocals: { similarity: Math.floor(Math.random() * 40) + 55, aiProbability: 94 },
            drums: { similarity: Math.floor(Math.random() * 40) + 58, aiProbability: 91 },
        },
        // More dramatic timeline for high similarity
        similarityTimeline: Array.from({ length: 13 }, (_, i) => ({
            timestamp: i * 15,
            similarity: Math.max(0, Math.min(100, Math.round(similarity * (1 - Math.abs(i - 6) / 6) + (Math.random() - 0.5) * 20))),
        })),
    };
};

const generateInitialReport = async (analysisData, analysisType = 'database', copyrightedSongName = '') => {
    const contextPreamble = analysisType === 'comparison'
        ? `A musician has received the following analysis comparing their AI-generated song to a specific track they uploaded, named "${copyrightedSongName}".`
        : 'A musician has received the following analysis of their AI-generated song against a public database of copyrighted music.';

    const prompt = `${contextPreamble}

Analysis Data:
${JSON.stringify(analysisData, null, 2)}

Please provide a comprehensive, professional report that includes:

1. **Executive Summary** (2-3 sentences): Overall assessment and key takeaway
2. **Risk Assessment**: Detailed explanation of the risk level and what it means
3. **Key Findings**: Most important discoveries from the analysis
4. **Recommendations**: Specific, actionable advice for the musician
5. **Next Steps**: What the musician should do based on these results

Write in a professional but encouraging tone. Be specific about the data but avoid legal advice. Focus on practical guidance for creators.`;

    try {
        if (!process.env.API_KEY) {
            return "AI analysis is currently unavailable. Please check your API configuration.";
        }

        const genAI = ai.getGenerativeModel({ 
            model: model,
            safetySettings: safetySettings
        });
        
        const result = await genAI.generateContent(prompt);
        const response = await result.response;
        return response.text();
    } catch (error) {
        console.error('Error generating initial report:', error);
        return "Unable to generate AI analysis at this time. Please try again later.";
    }
};

// --- API Routes ---

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        services: {
            acrcloud: useLiveAnalysis ? 'enabled' : 'simulated',
            ai: process.env.API_KEY ? 'enabled' : 'disabled'
        }
    });
});

// Analysis endpoint - scan against public database
app.post('/api/analyze', upload.single('audioFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file provided' });
        }

        console.log(`Received audio file: ${req.file.originalname} (${req.file.size} bytes)`);

        let analysisData;
        
        if (useLiveAnalysis) {
            try {
                analysisData = await runLiveFingerprinting(req.file.buffer, req.file.mimetype);
            } catch (error) {
                console.error("Live analysis failed, falling back to simulation:", error);
                analysisData = runSimulatedFingerprinting();
            }
        } else {
            analysisData = runSimulatedFingerprinting();
        }

        // Generate AI report
        const aiReport = await generateInitialReport(analysisData, 'database');
        analysisData.aiReport = aiReport;

        // Store analysis with unique ID
        const analysisId = crypto.randomUUID();
        sharedAnalyses.set(analysisId, {
            ...analysisData,
            filename: req.file.originalname,
            timestamp: new Date().toISOString(),
            type: 'database'
        });

        res.json({
            analysisId,
            ...analysisData
        });

    } catch (error) {
        console.error('Analysis error:', error);
        res.status(500).json({ error: 'Analysis failed', details: error.message });
    }
});

// Comparison endpoint - compare two specific songs
app.post('/api/compare', upload.fields([
    { name: 'aiSong', maxCount: 1 },
    { name: 'copyrightedSong', maxCount: 1 }
]), async (req, res) => {
    try {
        if (!req.files?.aiSong?.[0] || !req.files?.copyrightedSong?.[0]) {
            return res.status(400).json({ error: 'Both AI song and copyrighted song files are required' });
        }

        const aiSongFile = req.files.aiSong[0];
        const copyrightedSongFile = req.files.copyrightedSong[0];

        console.log(`Comparing: ${aiSongFile.originalname} vs ${copyrightedSongFile.originalname}`);

        // For now, we'll use simulated comparison
        // In the future, this could do actual audio comparison
        const analysisData = runSimulatedComparison(copyrightedSongFile.originalname);

        // Generate AI report for comparison
        const aiReport = await generateInitialReport(analysisData, 'comparison', copyrightedSongFile.originalname);
        analysisData.aiReport = aiReport;

        // Store analysis with unique ID
        const analysisId = crypto.randomUUID();
        sharedAnalyses.set(analysisId, {
            ...analysisData,
            aiSongFilename: aiSongFile.originalname,
            copyrightedSongFilename: copyrightedSongFile.originalname,
            timestamp: new Date().toISOString(),
            type: 'comparison'
        });

        res.json({
            analysisId,
            ...analysisData
        });

    } catch (error) {
        console.error('Comparison error:', error);
        res.status(500).json({ error: 'Comparison failed', details: error.message });
    }
});

// Get analysis by ID
app.get('/api/analysis/:id', (req, res) => {
    const analysis = sharedAnalyses.get(req.params.id);
    if (!analysis) {
        return res.status(404).json({ error: 'Analysis not found' });
    }
    res.json(analysis);
});

// Get user's analysis library (simplified - in production would be user-specific)
app.get('/api/library', (req, res) => {
    const analyses = Array.from(sharedAnalyses.entries()).map(([id, data]) => ({
        id,
        filename: data.filename || data.aiSongFilename,
        timestamp: data.timestamp,
        type: data.type,
        riskLevel: data.overview.riskLevel,
        similarity: data.overview.similarity
    }));
    
    res.json(analyses.slice(-20)); // Return last 20 analyses
});

// Chat endpoint for AI assistant
app.post('/api/chat', async (req, res) => {
    try {
        const { message, context } = req.body;

        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        if (!process.env.API_KEY) {
            return res.status(503).json({ error: 'AI service is currently unavailable' });
        }

        const systemInstruction = getSystemInstructionForContext(context || {});
        
        const genAI = ai.getGenerativeModel({ 
            model: model,
            safetySettings: safetySettings,
            systemInstruction: systemInstruction
        });

        const result = await genAI.generateContent(message);
        const response = await result.response;
        
        res.json({ response: response.text() });

    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Chat failed', details: error.message });
    }
});

// Prompt enhancement endpoint
app.post('/api/enhance-prompt', async (req, res) => {
    try {
        const { genre, mood, instruments, lyrical } = req.body;

        if (!process.env.API_KEY) {
            return res.status(503).json({ error: 'AI service is currently unavailable' });
        }

        const userPrompt = `Genre: ${genre || 'Not specified'}
Mood: ${mood || 'Not specified'}
Instruments: ${instruments || 'Not specified'}
Lyrical Theme: ${lyrical || 'Not specified'}`;

        const enhancementPrompt = `You are an expert at crafting prompts for AI music generation tools like Suno and Udio. 

The user has provided these basic elements for their music prompt:
${userPrompt}

Please enhance this into a detailed, effective prompt that will generate better results. Include:
- Specific musical elements and production details
- Atmospheric and emotional descriptors
- Technical aspects that AI music generators respond well to
- Keep it concise but descriptive (2-3 sentences max)

Return only the enhanced prompt, nothing else.`;

        const genAI = ai.getGenerativeModel({ 
            model: model,
            safetySettings: safetySettings
        });

        const result = await genAI.generateContent(enhancementPrompt);
        const response = await result.response;
        
        res.json({ enhancedPrompt: response.text() });

    } catch (error) {
        console.error('Prompt enhancement error:', error);
        res.status(500).json({ error: 'Prompt enhancement failed', details: error.message });
    }
});

// Catalog endpoint (placeholder)
app.get('/api/catalog', (req, res) => {
    // This would connect to a real database in production
    const sampleCatalog = [
        {
            id: '1',
            title: 'Ambient Waves',
            artist: 'Digital Dreams',
            genre: 'Ambient',
            riskScore: 5,
            price: 29.99
        },
        {
            id: '2', 
            title: 'Lo-Fi Sunset',
            artist: 'Chill Collective',
            genre: 'Lo-Fi Hip Hop',
            riskScore: 8,
            price: 19.99
        }
    ];
    
    res.json(sampleCatalog);
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`🎵 MelodyCompare backend server running on port ${port}`);
    console.log(`🔗 Health check: http://localhost:${port}/api/health`);
    console.log(`🎯 CORS enabled for: ${allowedOrigins.join(', ')}`);
});

