# STT_Local_Mac

A Python application that performs Speech-to-Text transcription from YouTube videos using the Whisper model, running locally on Mac M-series GPUs. The application also generates a summary of the transcribed text.

## Features

- Download audio from YouTube videos
- Transcribe audio using Whisper model (distil-large-v2)
- Generate summary using BART model
- Utilizes Mac M-series GPU for faster processing
- Outputs both full transcription and summary

## Prerequisites

- Mac with M-series chip (M1/M2/M3)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- FFmpeg (can be installed via Homebrew: `brew install ffmpeg`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/STT_Local_Mac.git
   cd STT_Local_Mac
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate whisper-env
   ```

## Usage

1. Run the script:
   ```bash
   python transcribe.py
   ```

2. When prompted, enter a YouTube URL:
   ```
   Enter YouTube URL: https://youtu.be/your-video-id
   ```

3. The script will:
   - Download the audio
   - Transcribe the content
   - Generate a summary
   - Save the outputs

## Output Files

After processing, you'll find two files in your project directory:

- `transcription.txt`: Contains the full transcription of the video
- `transcription_summary.md`: Contains a markdown-formatted summary of the content

## Troubleshooting

If you encounter any issues:

1. Ensure FFmpeg is installed:
   ```bash
   brew install ffmpeg
   ```

2. Verify your environment is activated:
   ```bash
   conda activate whisper-env
   ```

3. Check that you're using the correct Python version:
   ```bash
   python --version  # Should be Python 3.10.x
   ```

## Notes

- Processing time depends on the video length and your machine's capabilities
- The script requires an active internet connection to download YouTube videos
- Generated files are saved in the same directory as the script