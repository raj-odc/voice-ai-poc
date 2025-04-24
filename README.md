# Voice Assistant PoC

A Python-based voice assistant with speaker recognition, speech-to-text conversion, MongoDB integration, and GPT-powered responses.

## Features

- Speaker Recognition using voice embeddings
- Speech-to-Text conversion using Whisper
- MongoDB integration for data storage and retrieval
- OpenAI GPT integration for intelligent responses
- Text-to-Speech output

## Prerequisites

- Python 3.8 or higher
- MongoDB installed and running locally
- OpenAI API key
- Working microphone and speakers

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-ai-poc
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Copy the `.env.example` file to `.env`
- Update the following variables in `.env`:
  - `MONGODB_URI`: Your MongoDB connection string
  - `OPENAI_API_KEY`: Your OpenAI API key

## Usage

1. Run the main script:
```bash
python main.py
```

2. Choose from the following options:
   - Register new user: Record a voice sample for speaker recognition
   - Process voice command: Speak a command and get a response
   - Exit: Close the application

## Testing

1. Register a new user:
   - Select option 1
   - Enter a unique user ID
   - Speak for 5 seconds when prompted

2. Test voice commands:
   - Select option 2
   - Speak your query when prompted
   - Wait for the assistant to process and respond

## MongoDB Setup

1. Create a database named `voice_assistant`
2. The following collections will be created automatically:
   - `voice_profiles`: Stores user voice embeddings
   - `information`: Stores queryable data

## Troubleshooting

- Ensure your microphone is properly connected and selected as the default input device
- Check MongoDB is running locally on the default port (27017)
- Verify your OpenAI API key is valid and has sufficient credits
- Make sure all required Python packages are installed correctly

## Error Handling

The application includes comprehensive error handling for:
- Audio capture issues
- Speech recognition failures
- Database connection problems
- API communication errors

Check the console output for specific error messages and troubleshooting guidance.