import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import yt_dlp
import os

class AudioTranscriber:
    def __init__(self):
        # Initialize device - use MPS if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor for speech recognition
        print("Initializing speech recognition model...")
        model_id = "distil-whisper/distil-large-v2"
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch.float16,
            device=self.device,
        )

        # Initialize summarization pipeline
        print("Initializing summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=self.device
        )
        print("Models initialized successfully!")

    def download_audio(self, youtube_url, output_path="audio"):
        """Download audio from YouTube URL"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            audio_path = f"{output_path}/{info['title']}.mp3"
            return audio_path

    def _chunk_text(self, text, max_chunk_size=1000):
        """Split text into smaller chunks while maintaining sentence integrity"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks

    def summarize_text(self, text):
        """Generate a summary from text"""
        try:
            print("Generating summary...")
            chunks = self._chunk_text(text)
            summaries = []
            
            for i, chunk in enumerate(chunks, 1):
                print(f"Summarizing chunk {i}/{len(chunks)}...")
                summary = self.summarizer(
                    chunk,
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            
            return ' '.join(summaries)
            
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return None

    def transcribe(self, youtube_url):
        """Transcribe audio from YouTube URL and generate summary"""
        try:
            # Download audio
            print("Downloading audio...")
            audio_path = self.download_audio(youtube_url)
            
            # Transcribe
            print("Transcribing...")
            result = self.pipe(
                audio_path,
                generate_kwargs={"language": "en", "task": "transcribe"}
            )
            
            # Get transcription
            transcription = result["text"]
            
            # Generate summary
            summary = self.summarize_text(transcription)
            
            # Clean up audio file
            os.remove(audio_path)
            
            # Save transcription
            with open("transcription.txt", "w") as f:
                f.write(transcription)
            print("\nTranscription saved to: transcription.txt")
            
            # Save summary in markdown format
            with open("transcription_summary.md", "w") as f:
                f.write("# Video Summary\n\n")
                f.write(summary)
            print("Summary saved to: transcription_summary.md")
            
            return transcription, summary
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None, None

def main():
    try:
        print("Initializing transcriber...")
        transcriber = AudioTranscriber()
        
        youtube_url = input("\nEnter YouTube URL: ")
        transcription, summary = transcriber.transcribe(youtube_url)
        
        if transcription and summary:
            print("\nProcessing completed successfully!")
            print("\nPreview of transcription (first 150 characters):")
            print(transcription[:150] + "...")
            print("\nPreview of summary (first 150 characters):")
            print(summary[:150] + "...")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()