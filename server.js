import express from 'express';
import cors from 'cors';
import 'dotenv/config'; // To load .env variables
import multer from 'multer';
import crypto from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from '@google/genai';

const app = express();
const port = process.env.PORT || 3001;

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
    'https://www.melodycompare.com',
    /http:\/\/(localhost|127\.0\.0\.1):\d+/ // Allow localhost & 127.0.0.1 for development
];
app.use(cors({ origin: allowedOrigins }));
app.use(express.json({ limit: '10mb' }));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage, limits: { fileSize: 20 * 1024 * 1024 } }); // 20MB limit

// For live analysis, ACRCloud credentials are required
const useLiveAnalysis = process.env.ACR_HOST && process.env.ACR_ACCESS_KEY && process.env.ACR_ACCESS_SECRET;
if (useLiveAnalysis) {
    console.log("ACRCloud credentials found. Live analysis enabled.");
} else {
    console.warn("ACRCloud credentials not found in .env file. Falling back to simulated analysis.");
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
 * Performs a live audio fingerprinting scan using ACRCloud.
 * @param {Buffer} audioBuffer - The audio file data.
 * @returns {Promise<object>} A Promise that resolves to the AnalysisData object.
 */
const runLiveFingerprinting = async (audioBuffer) => {
    const { ACR_HOST, ACR_ACCESS_KEY, ACR_ACCESS_SECRET } = process.env;

    const timestamp = Math.floor(Date.now() / 1000);
    const stringToSign = `POST\n/v1/identify\n${ACR_ACCESS_KEY}\naudio\n1\n${timestamp}`;
    const signature = crypto.createHmac('sha1', ACR_ACCESS_SECRET).update(stringToSign).digest('base64');

    const formData = new FormData();
    formData.append('sample', audioBuffer, { filename: 'track.mp3', contentType: 'audio/mp3' });
    formData.append('access_key', ACR_ACCESS_KEY);
    formData.append('data_type', 'audio');
    formData.append('signature_version', '1');
    formData.append('signature', signature);
    formData.append('timestamp', timestamp);
    
    const response = await fetch(`https://${ACR_HOST}/v1/identify`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`ACRCloud API HTTP Error: ${response.status} ${errorText}`);
    }

    const acrResponse = await response.json();
    return transformAcrResponseToAnalysisData(acrResponse);
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
        : 'A musician has received the following analysis comparing their AI-generated song to tracks in a public database.';

    const prompt = `
You are an expert Music Licensing Advisor. ${contextPreamble}

Analysis Data:
\`\`\`json
${JSON.stringify(analysisData, null, 2)}
\`\`\`

Your task is to provide a detailed, encouraging, and actionable report in Markdown format. The report MUST include the following sections in this exact order:

1.  A main title for the report, using a single '#' in Markdown (e.g., # Your Song Analysis Report).
2.  A subtitle "Understanding Your Risk Score" using '##'. Explain the 'Overall Risk' score and 'Risk Level' in plain, easy-to-understand English. Avoid overly technical jargon.
3.  A subtitle "Actionable Next Steps" using '##'. Provide a clear, numbered list of concrete steps the musician should take next.
4.  A subtitle "Creative Tune-Up Suggestions" using '##'. Give creative, specific ideas on how to modify the song to reduce similarity. Focus on musical elements like melody, rhythm, and instrumentation, referencing the high-similarity areas from the stem analysis (vocals and drums).
5.  A subtitle "A Final Note of Encouragement" using '##'. End with a positive and encouraging paragraph. Reassure the musician that this is a common part of the creative process and they have a clear path forward.

Format your entire response strictly in Markdown. Do not include any other text, greetings, or explanations before or after the Markdown report.
    `;
    
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: { safetySettings }
    });
    
    return response.text;
};

const handleApiError = (res, error, context) => {
    console.error(`Error in ${context}:`, error);
    res.status(500).json({ error: `Failed to ${context}. Please check the server logs.` });
};

// --- API Routes ---

const apiRouter = express.Router();

apiRouter.post('/analyze', upload.single('audioFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file was uploaded.' });
        }
        console.log(`Received file for DB scan: ${req.file.originalname}, Size: ${req.file.size} bytes`);

        let analysisData;
        if (useLiveAnalysis) {
            console.log("Processing with live ACRCloud analysis...");
            analysisData = await runLiveFingerprinting(req.file.buffer);
        } else {
            console.log("Processing with simulated database analysis...");
            analysisData = runSimulatedFingerprinting();
        }

        const reportText = await generateInitialReport(analysisData, 'database');
        res.json({ analysisData, reportText });

    } catch (error) {
        handleApiError(res, error, 'perform analysis and generate report');
    }
});

apiRouter.post('/compare', upload.fields([
    { name: 'aiSong', maxCount: 1 },
    { name: 'copyrightedSong', maxCount: 1 }
]), async (req, res) => {
    try {
        const aiSongFile = req.files?.aiSong?.[0];
        const copyrightedSongFile = req.files?.copyrightedSong?.[0];

        if (!aiSongFile || !copyrightedSongFile) {
            return res.status(400).json({ error: 'Both an AI song and a copyrighted song must be uploaded.' });
        }
        
        console.log(`Received files for comparison: ${aiSongFile.originalname} vs ${copyrightedSongFile.originalname}`);
        
        console.log("Processing with simulated comparison...");
        const analysisData = runSimulatedComparison(copyrightedSongFile.originalname);
        
        const reportText = await generateInitialReport(analysisData, 'comparison', copyrightedSongFile.originalname);
        res.json({ analysisData, reportText });

    } catch (error) {
        handleApiError(res, error, 'perform song comparison');
    }
});

apiRouter.post('/share', (req, res) => {
    try {
        const { analysisData, reportText } = req.body;
        if (!analysisData || !reportText) {
            return res.status(400).json({ error: 'Analysis data and report text are required to create a shareable link.' });
        }
        const id = crypto.randomBytes(6).toString('hex');
        sharedAnalyses.set(id, { analysisData, reportText });
        console.log(`Created shareable link with id: ${id}`);
        setTimeout(() => {
            sharedAnalyses.delete(id);
            console.log(`Expired and deleted shared analysis: ${id}`);
        }, 24 * 60 * 60 * 1000); 

        res.status(201).json({ id });
    } catch (error) {
        handleApiError(res, error, 'create share link');
    }
});

apiRouter.post('/analysis-audio/:id', upload.single('audioFile'), (req, res) => {
    try {
        const { id } = req.params;
        const audioFile = req.file;

        if (!audioFile) {
            return res.status(400).json({ error: 'No audio file was uploaded.' });
        }
        if (!id) {
            return res.status(400).json({ error: 'An analysis ID is required.' });
        }
        
        audioStore.set(id, { buffer: audioFile.buffer, mimetype: audioFile.mimetype });
        console.log(`Stored audio for analysis ID: ${id}`);
        
        res.status(204).send();

    } catch (error) {
        handleApiError(res, error, 'store analysis audio');
    }
});

apiRouter.get('/analysis/:id', (req, res) => {
    try {
        const { id } = req.params;
        const data = sharedAnalyses.get(id);

        if (data) {
            console.log(`Retrieved shared analysis: ${id}`);
            res.json(data);
        } else {
            res.status(404).json({ error: 'Shared analysis not found. It may have expired.' });
        }
    } catch (error) {
        handleApiError(res, error, 'retrieve shared analysis');
    }
});

apiRouter.post('/generate-report', async (req, res) => {
    try {
        const { analysisData } = req.body;
        if (!analysisData) {
            return res.status(400).json({ error: 'analysisData is required.' });
        }
        const reportText = await generateInitialReport(analysisData, 'database');
        res.json({ reportText });

    } catch (error) {
        handleApiError(res, error, 'generate initial report');
    }
});

apiRouter.post('/assistant-chat', async (req, res) => {
    try {
        const { history, message, context } = req.body;
        if (!history || !message || !context) {
            return res.status(400).json({ error: 'History, message, and context are required.' });
        }
        
        const systemInstruction = getSystemInstructionForContext(context);

        const geminiHistory = history.map(msg => ({
            role: msg.role === 'model' ? 'model' : 'user',
            parts: [{ text: msg.content }],
        }));
        
        const chat = ai.chats.create({ 
            model, 
            history: geminiHistory, 
            config: { 
                systemInstruction: { role: 'system', parts: [{ text: systemInstruction }] },
                safetySettings 
            }
        });
        const stream = await chat.sendMessageStream({ message });

        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        
        for await (const chunk of stream) {
            res.write(chunk.text);
        }
        res.end();

    } catch (error) {
        handleApiError(res, error, 'process assistant chat stream');
    }
});

apiRouter.post('/brainstorm', async (req, res) => {
    try {
        const { analysisData, mode, theme } = req.body;

        const getBrainstormingPrompt = (data, mode, theme) => {
            const highSimilarityStems = Object.entries(data.stemAnalysis)
                .filter(([, value]) => value.similarity > 50)
                .map(([key]) => key);
            
            const context = `The user's song has a ${data.overview.riskLevel} risk of copyright issues. The overall similarity is ${data.overview.overallScore}%. The most similar parts are: ${highSimilarityStems.join(', ') || 'N/A'}. The AI platform used was likely ${data.aiAnalysis.platform}.`;
            let instruction = '';
            switch (mode) {
                case 'titles': instruction = `Generate 5 creative and unique alternative song titles.`; break;
                case 'lyrics': instruction = `Generate 3 short, distinct lyrical concepts (2-3 lines each) that could fit a new direction for the song.`; break;
                case 'chords': instruction = `Suggest 3 alternative chord progressions that could replace a high-similarity section. Provide them in a standard format (e.g., C - G - Am - F).`; break;
            }
            const themeInstruction = theme ? ` The user wants the ideas to fit a theme of "${theme}".` : '';
            return `You are a creative songwriting partner. Based on the following musical analysis, ${instruction}${themeInstruction}\n\nAnalysis Context:\n${context}`;
        };

        const prompt = getBrainstormingPrompt(analysisData, mode, theme);
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: { type: Type.ARRAY, items: { type: Type.STRING }, description: 'A list of creative ideas.' },
                safetySettings,
            },
        });
        
        res.json(JSON.parse(response.text.trim()));

    } catch (error) {
        handleApiError(res, error, 'generate brainstorming ideas');
    }
});

apiRouter.post('/enhance-prompt', async (req, res) => {
    try {
        const { basePrompt } = req.body;
        if (!basePrompt) {
            return res.status(400).json({ error: 'basePrompt is required' });
        }
        
        const systemPrompt = `You are an expert AI music prompt engineer. Your task is to take a user's basic idea and expand it into a rich, detailed, and effective prompt for an AI music generator like Suno or Udio. 
        - Use descriptive adjectives and evocative language.
        - Structure the prompt clearly, often using comma-separated tags or descriptive phrases.
        - Specify instrumentation, mood, genre, and vocal style if mentioned.
        - Maintain the core creative intent of the user's input.
        - Return ONLY the enhanced prompt, without any explanations, greetings, or extra text.`;

        const response = await ai.models.generateContent({
            model: model,
            contents: basePrompt,
            config: { systemInstruction: systemPrompt, safetySettings },
        });
        
        res.json({ enhancedPrompt: response.text });

    } catch (error) {
        handleApiError(res, error, 'enhance music prompt');
    }
});

apiRouter.post('/feedback', (req, res) => {
    try {
        const { type, message, email } = req.body;

        if (!type || !message) {
            return res.status(400).json({ error: 'Feedback type and message are required.' });
        }
        
        console.log("--- NEW USER FEEDBACK ---");
        console.log(`Type: ${type}`);
        console.log(`From: ${email || 'Anonymous'}`);
        console.log(`Message: ${message}`);
        console.log("-------------------------");

        res.status(204).send();

    } catch (error) {
        handleApiError(res, error, 'process feedback');
    }
});

apiRouter.post('/catalog/submit', upload.single('audioFile'), (req, res) => {
    try {
        const { title, genre, tags, analysisId, riskScore, userId, userName } = req.body;
        const audioFile = req.file;

        if (!title || !genre || !tags || !analysisId || !riskScore || !userId || !userName || !audioFile) {
            return res.status(400).json({ error: 'Missing required fields for catalog submission.' });
        }

        const id = crypto.randomUUID();
        const newEntry = {
            id,
            analysisId,
            userId,
            userName,
            title,
            genre,
            tags: tags.split(',').map(t => t.trim()),
            dateSubmitted: new Date().toISOString(),
            riskScore: Number(riskScore),
        };
        
        // Use the catalog item's ID for the audio key to ensure uniqueness
        catalogEntries.set(id, newEntry);
        audioStore.set(id, { buffer: audioFile.buffer, mimetype: audioFile.mimetype });
        
        console.log(`New track submitted to catalog: "${title}" (ID: ${id})`);

        res.status(201).json(newEntry);

    } catch (error) {
        handleApiError(res, error, 'submit to catalog');
    }
});

apiRouter.get('/catalog/entries', (req, res) => {
    try {
        const entries = Array.from(catalogEntries.values()).sort((a,b) => new Date(b.dateSubmitted).getTime() - new Date(a.dateSubmitted).getTime());
        res.json(entries);
    } catch (error) {
        handleApiError(res, error, 'get catalog entries');
    }
});

apiRouter.get('/audio/:id', (req, res) => {
    try {
        const { id } = req.params;
        const audio = audioStore.get(id);

        if (audio) {
            res.setHeader('Content-Type', audio.mimetype);
            res.setHeader('Content-Length', audio.buffer.length);
            res.send(audio.buffer);
        } else {
            res.status(404).json({ error: 'Audio file not found.' });
        }
    } catch (error) {
        handleApiError(res, error, 'retrieve audio file');
    }
});


// ** NEW SERVER STRUCTURE **

// 1. Handle API routes FIRST.
app.use('/api', apiRouter);

// 2. Serve static files from the 'dist' folder.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(path.join(__dirname, 'dist')));

// 3. For any other GET request that is not an API route, serve the SPA.
// This is the fallback for client-side routing.
app.get(/^(?!\/api).*/, (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// --- Server Startup ---
app.listen(port, () => {
    console.log(`MelodyCompare backend server is running on http://localhost:${port}`);
});
