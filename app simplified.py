import streamlit as st
import os
import re
import asyncio
import edge_tts
import time
from datetime import datetime, timezone
from pathlib import Path
import traceback
import subprocess
import html

# App title and version
APP_TITLE = "SRT to Audio Converter"
VERSION = "4.0.0"

# Default frame rate now set to 30 FPS
DEFAULT_FRAME_RATE = "30"

# Define reliable voice options - Croatian first, then UK
CROATIAN_VOICES = [
    {"name": "hr-HR-SreckoNeural", "display_name": "Srećko (Male)"},
    {"name": "hr-HR-GabrijelaNeural", "display_name": "Gabrijela (Female)"}
]

UK_VOICES = [
    {"name": "en-GB-RyanNeural", "display_name": "Ryan (Male)"},
    {"name": "en-GB-SoniaNeural", "display_name": "Sonia (Female)"},
    {"name": "en-GB-LibbyNeural", "display_name": "Libby (Female)"}
]

# Combined voices list
AVAILABLE_VOICES = CROATIAN_VOICES + UK_VOICES

def setup_app():
    """Configure the Streamlit app settings"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎬",
        layout="wide"
    )
    
    # Initialize session state
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = DEFAULT_FRAME_RATE
    if 'selected_voice' not in st.session_state:
        # Default to Srećko voice
        st.session_state.selected_voice = CROATIAN_VOICES[0]["name"]
    if 'selected_voice_display' not in st.session_state:
        st.session_state.selected_voice_display = CROATIAN_VOICES[0]["display_name"]

def ensure_output_directory():
    """Ensure the audio_output directory exists"""
    output_dir = Path(os.path.join(os.getcwd(), "audio_output"))
    output_dir.mkdir(exist_ok=True)
    return output_dir

def clean_html_tags(text):
    """Remove HTML tags from text"""
    # First use the html module to unescape any HTML entities
    text = html.unescape(text)
    # Then remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text

def parse_srt_file(content):
    """Parse SRT file content into subtitle blocks"""
    if not content or not content.strip():
        return []
    
    # Split content into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                subtitle_num = int(lines[0].strip())
            except (ValueError, IndexError):
                continue
                
            # Extract timing
            timing_line = lines[1].strip()
            timing_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timing_line)
            if not timing_match:
                continue
                
            start_time = timing_match.group(1)
            end_time = timing_match.group(2)
            
            # Extract subtitle text and clean HTML tags
            subtitle_text = '\n'.join(lines[2:]).strip()
            clean_text = clean_html_tags(subtitle_text)
            
            if clean_text:
                subtitles.append({
                    'number': subtitle_num,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': clean_text,
                    'original_text': subtitle_text
                })
    
    return subtitles

class CyrillicTransliterator:
    """Convert Cyrillic characters to Latin equivalents"""
    
    def __init__(self):
        # Comprehensive Cyrillic to Latin mapping
        self.cyrillic_to_latin = {
            # Russian/Serbian Cyrillic to Latin
            'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
            'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e',
            'Ё': 'Yo', 'ё': 'yo', 'Ж': 'Zh', 'ж': 'zh', 'З': 'Z', 'з': 'z',
            'И': 'I', 'и': 'i', 'Й': 'Y', 'й': 'y', 'К': 'K', 'к': 'k',
            'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n',
            'О': 'O', 'о': 'o', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
            'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u',
            'Ф': 'F', 'ф': 'f', 'Х': 'Kh', 'х': 'kh', 'Ц': 'Ts', 'ц': 'ts',
            'Ч': 'Ch', 'ч': 'ch', 'Ш': 'Sh', 'ш': 'sh', 'Щ': 'Shch', 'щ': 'shch',
            'Ъ': '', 'ъ': '', 'Ы': 'Y', 'ы': 'y', 'Ь': '', 'ь': '',
            'Э': 'E', 'э': 'e', 'Ю': 'Yu', 'ю': 'yu', 'Я': 'Ya', 'я': 'ya',
            
            # Serbian specific
            'Ј': 'J', 'ј': 'j', 'Љ': 'Lj', 'љ': 'lj', 'Њ': 'Nj', 'њ': 'nj',
            'Ћ': 'C', 'ћ': 'c', 'Ђ': 'Dj', 'ђ': 'dj', 'Џ': 'Dz', 'џ': 'dz',
            
            # Macedonian specific
            'Ѓ': 'Gj', 'ѓ': 'gj', 'Ѕ': 'Dz', 'ѕ': 'dz', 'Ќ': 'Kj', 'ќ': 'kj',
            
            # Bulgarian specific
            'Ѝ': 'I', 'ѝ': 'i'
        }
    
    def transliterate(self, text):
        """Convert Cyrillic text to Latin transliteration"""
        if not text:
            return text
            
        result = ""
        for char in text:
            result += self.cyrillic_to_latin.get(char, char)
        
        return result
    
    def has_cyrillic(self, text):
        """Check if text contains Cyrillic characters"""
        if not text:
            return False
        cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')
        return bool(cyrillic_pattern.search(text))

async def get_voice_list():
    """Get all available voices from Edge TTS"""
    try:
        voices = await edge_tts.list_voices()
        return voices
    except Exception as e:
        st.error(f"Error fetching voices: {e}")
        return []

async def generate_audio(text, voice_name, rate=0, volume=0, pitch=0):
    """Generate audio from text using Edge TTS"""
    try:
        # Create audio parameters
        rate_str = f"{rate:+d}%" if rate != 0 else "+0%"
        volume_str = f"{volume:+d}%" if volume != 0 else "+0%"
        pitch_str = f"{pitch:+d}Hz" if pitch != 0 else "+0Hz"
        
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            st.warning(f"Empty text received for processing. Skipping.")
            return None
        
        # Initialize Edge TTS
        communicate = edge_tts.Communicate(
            text,
            voice_name,
            rate=rate_str,
            volume=volume_str,
            pitch=pitch_str
        )
        
        # Collect audio data
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        if not audio_data:
            st.error(f"No audio data received for voice '{voice_name}'")
            return None
         
        return audio_data
    
    except edge_tts.exceptions.NoAudioReceived:
        st.error(f"No audio received for voice '{voice_name}'. Please try another voice.")
        st.info("Available voices for Edge TTS may vary. Try a different voice if this one doesn't work.")
        return None
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

async def process_subtitle_to_audio(subtitle, voice_name, rate, volume, pitch, transliterator):
    """Process a single subtitle to audio"""
    try:
        text = subtitle['text']
        
        # Check for Cyrillic and transliterate if needed
        has_cyrillic = transliterator.has_cyrillic(text)
        if has_cyrillic:
            text = transliterator.transliterate(text)
        
        # Generate audio
        audio_data = await generate_audio(text, voice_name, rate, volume, pitch)
        
        return {
            'number': subtitle['number'],
            'start_time': subtitle['start_time'],
            'end_time': subtitle['end_time'],
            'text': text,
            'original_text': subtitle.get('original_text', subtitle['text']),
            'has_cyrillic': has_cyrillic,
            'audio_data': audio_data
        }
    except Exception as e:
        st.error(f"Error processing subtitle {subtitle['number']}: {str(e)}")
        return None

async def process_all_subtitles(subtitles, voice_name, rate, volume, pitch, progress_callback=None):
    """Process all subtitles to audio"""
    transliterator = CyrillicTransliterator()
    results = []
    total = len(subtitles) if subtitles else 0
    
    if total == 0:
        return []
    
    for i, subtitle in enumerate(subtitles):
        # Process subtitle
        result = await process_subtitle_to_audio(subtitle, voice_name, rate, volume, pitch, transliterator)
        
        # Update progress
        if progress_callback:
            progress_callback(i + 1, total, subtitle)
        
        # Only add successful results
        if result and result.get('audio_data'):
            results.append(result)
        
        # Small delay to prevent overloading the service
        await asyncio.sleep(0.1)
    
    return results

def save_audio_file(audio_results, subtitle_filename, frame_rate):
    """Save audio results to a file in the audio_output folder"""
    if not audio_results or len(audio_results) == 0:
        st.error("No audio generated. Cannot save file.")
        return None
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base filename from subtitle file
    if subtitle_filename:
        base_name = os.path.splitext(os.path.basename(subtitle_filename))[0]
    else:
        base_name = "audio"
    
    # Ensure output directory exists
    output_dir = ensure_output_directory()
    
    # Create output filepath with original name and timestamp
    output_filepath = os.path.join(output_dir, f"{base_name}_{frame_rate}fps_{timestamp}.wav")
    
    try:
        # Combine all audio data
        combined_audio = b""
        for result in audio_results:
            if result.get('audio_data'):
                combined_audio += result['audio_data']
        
        # Write to file
        with open(output_filepath, 'wb') as f:
            f.write(combined_audio)
        
        return output_filepath
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

def save_direct_audio(audio_data, filename_prefix="direct_audio"):
    """Save direct audio data to a file in the audio_output folder"""
    if not audio_data:
        st.error("No audio generated. Cannot save file.")
        return None
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    output_dir = ensure_output_directory()
    
    # Create output filepath
    output_filepath = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.wav")
    
    try:
        # Write to file
        with open(output_filepath, 'wb') as f:
            f.write(audio_data)
        
        return output_filepath
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

def get_subtitle_stats(subtitles):
    """Calculate statistics for subtitles"""
    if not subtitles:
        return {"count": 0, "words": 0, "chars": 0}
    
    total_text = " ".join(s["text"] for s in subtitles)
    return {
        "count": len(subtitles),
        "words": len(total_text.split()),
        "chars": len(total_text)
    }

def display_subtitle_preview(subtitles, max_display=5):
    """Display a preview of subtitles"""
    st.subheader("Subtitle Preview")
    
    if not subtitles:
        st.info("No subtitles to display")
        return
    
    # Display limited number of subtitles
    preview_subtitles = subtitles[:max_display]
    
    # Display HTML tag removal if detected
    any_html_tags = any("<" in s.get('original_text', '') and ">" in s.get('original_text', '') for s in subtitles)
    if any_html_tags:
        st.warning("HTML tags were detected and automatically removed from subtitles")
        with st.expander("Show example of HTML tag removal"):
            for s in subtitles:
                if "<" in s.get('original_text', '') and ">" in s.get('original_text', ''):
                    st.write("**Original with HTML:**", s.get('original_text'))
                    st.write("**Cleaned text:**", s['text'])
                    break
    
    # Create columns for display
    for subtitle in preview_subtitles:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**#{subtitle['number']}**")
            st.caption(f"{subtitle['start_time']} → {subtitle['end_time']}")
        with col2:
            st.write(subtitle['text'])
        st.markdown("---")
    
    if len(subtitles) > max_display:
        st.info(f"Showing {max_display} of {len(subtitles)} subtitles")

def get_current_utc_time():
    """Get current UTC time formatted as YYYY-MM-DD HH:MM:SS"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def srt_to_audio_tab():
    """SRT to Audio conversion tab"""
    st.header("🎬 SRT to Audio")
    st.write("Convert subtitle files to audio using text-to-speech")
    
    # Frame rate selector
    frame_rate = st.radio(
        "Frame Rate",
        ["25", "30"],
        index=1 if st.session_state.frame_rate == "30" else 0,
        horizontal=True,
        help="Select the frame rate for timing conversion (default: 30 FPS)"
    )
    
    # Update session state if changed
    if frame_rate != st.session_state.frame_rate:
        st.session_state.frame_rate = frame_rate
    
    # Voice selector - will display and automatically use the selected voice
    voice_options = [(voice["name"], voice["display_name"]) for voice in AVAILABLE_VOICES]
    
    # Create a dictionary for lookup
    voice_name_to_display = {name: display for name, display in voice_options}
    voice_display_to_name = {display: name for name, display in voice_options}
    
    # Find index of current selected voice
    default_index = 0
    for i, (name, display) in enumerate(voice_options):
        if name == st.session_state.selected_voice:
            default_index = i
            break
    
    # Voice selection dropdown
    selected_voice_display = st.selectbox(
        "Voice Selection",
        options=[display for _, display in voice_options],
        index=default_index,
        help="Srećko is selected by default, change if needed"
    )
    
    # Update selected voice in session state
    selected_voice_name = voice_display_to_name[selected_voice_display]
    if selected_voice_name != st.session_state.selected_voice:
        st.session_state.selected_voice = selected_voice_name
        st.session_state.selected_voice_display = selected_voice_display
    
    # Display current voice info
    st.info(f"Selected voice: {selected_voice_display} ({selected_voice_name})")
    
    # Audio settings
    st.subheader("Audio Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        rate = st.slider("Speed", -50, 50, 0, 5)
    with col2:
        volume = st.slider("Volume", -50, 50, 0, 5)
    with col3:
        pitch = st.slider("Pitch", -50, 50, 0, 5)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload SRT File",
        type=["srt"],
        help="Upload a subtitle file in SRT format"
    )
    
    if not uploaded_file:
        st.info("Please upload an SRT file to generate audio")
        return
    
    # Process uploaded file
    try:
        content = uploaded_file.read().decode('utf-8')
        subtitles = parse_srt_file(content)
        
        if not subtitles:
            st.error("No valid subtitles found in the file")
            return
        
        # Show subtitle statistics
        stats = get_subtitle_stats(subtitles)
        col1, col2, col3 = st.columns(3)
        col1.metric("Subtitles", stats["count"])
        col2.metric("Words", stats["words"])
        col3.metric("Characters", stats["chars"])
        
        # Display subtitle preview
        display_subtitle_preview(subtitles)
        
        # Check for Cyrillic text
        transliterator = CyrillicTransliterator()
        has_cyrillic = any(transliterator.has_cyrillic(s['text']) for s in subtitles)
        if has_cyrillic:
            st.warning("🔤 Cyrillic text detected - will be automatically transliterated to Latin")
        
        # Generate audio button
        if st.button("Generate Audio", type="primary"):
            # Set up progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(current, total, subtitle):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processing subtitle {current}/{total}: {subtitle['text'][:50]}...")
            
            # Run audio generation
            with st.spinner("Generating audio..."):
                try:
                    # Process all subtitles
                    audio_results = asyncio.run(
                        process_all_subtitles(
                            subtitles,
                            st.session_state.selected_voice,  # Use selected voice
                            rate,
                            volume,
                            pitch,
                            update_progress
                        )
                    )
                    
                    if not audio_results or len(audio_results) == 0:
                        st.error("No audio was generated. Please try a different voice.")
                        return
                    
                    # Save to file
                    output_file = save_audio_file(audio_results, uploaded_file.name, frame_rate)
                    
                    if output_file:
                        # Mark process as complete
                        progress_bar.progress(1.0)
                        status_text.text("Audio generation complete!")
                        
                        # Success message
                        st.success(f"✅ Audio file created: {os.path.basename(output_file)}")
                        
                        # Display audio player
                        with open(output_file, "rb") as f:
                            audio_bytes = f.read()
                            st.audio(audio_bytes)
                        
                        # Show file path
                        st.info(f"Audio saved to: {output_file}")
                    else:
                        st.error("Failed to save audio file")
                
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing SRT file: {str(e)}")

def simple_text_to_audio_tab():
    """Simple text to audio conversion tab"""
    st.header("🎤 Text to Audio")
    st.write("Convert text directly to audio using text-to-speech")
    
    # Voice selector - will display and automatically use the selected voice
    voice_options = [(voice["name"], voice["display_name"]) for voice in AVAILABLE_VOICES]
    
    # Create a dictionary for lookup
    voice_name_to_display = {name: display for name, display in voice_options}
    voice_display_to_name = {display: name for name, display in voice_options}
    
    # Find index of current selected voice
    default_index = 0
    for i, (name, display) in enumerate(voice_options):
        if name == st.session_state.selected_voice:
            default_index = i
            break
    
    # Voice selection dropdown
    selected_voice_display = st.selectbox(
        "Voice Selection",
        options=[display for _, display in voice_options],
        index=default_index,
        key="simple_voice_select",
        help="Srećko is selected by default, change if needed"
    )
    
    # Update selected voice in session state
    selected_voice_name = voice_display_to_name[selected_voice_display]
    if selected_voice_name != st.session_state.selected_voice:
        st.session_state.selected_voice = selected_voice_name
        st.session_state.selected_voice_display = selected_voice_display
    
    # Display current voice info
    st.info(f"Selected voice: {selected_voice_display} ({selected_voice_name})")
    
    # Audio settings
    st.subheader("Audio Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        rate = st.slider("Speed", -50, 50, 0, 5, key="simple_rate")
    with col2:
        volume = st.slider("Volume", -50, 50, 0, 5, key="simple_volume")
    with col3:
        pitch = st.slider("Pitch", -50, 50, 0, 5, key="simple_pitch")
    
    # Text input
    text_input = st.text_area(
        "Enter text to convert to audio",
        height=200,
        placeholder="Type or paste your text here..."
    )
    
    # Clean HTML tags if any
    if text_input and ("<" in text_input and ">" in text_input):
        cleaned_text = clean_html_tags(text_input)
        if cleaned_text != text_input:
            st.warning("HTML tags detected and will be removed")
            with st.expander("Show original vs cleaned text"):
                st.write("**Original:**", text_input[:200])
                st.write("**Cleaned:**", cleaned_text[:200])
            text_input = cleaned_text
    
    # Filename for saving
    custom_filename = st.text_input(
        "Custom filename (optional):",
        value=""
    )
    
    # Generate audio button
    if st.button("Generate Audio", key="gen_simple", type="primary"):
        if not text_input:
            st.warning("Please enter some text to convert to audio")
            return
        
        with st.spinner("Generating audio..."):
            try:
                # Check for Cyrillic
                transliterator = CyrillicTransliterator()
                has_cyrillic = transliterator.has_cyrillic(text_input)
                if has_cyrillic:
                    st.warning("🔤 Cyrillic text detected - automatically transliterating to Latin")
                    text_to_process = transliterator.transliterate(text_input)
                else:
                    text_to_process = text_input
                
                # Generate audio
                audio_data = asyncio.run(generate_audio(
                    text_to_process, 
                    st.session_state.selected_voice,
                    rate, 
                    volume, 
                    pitch
                ))
                
                if audio_data:
                    # Use custom filename if provided
                    filename_prefix = custom_filename if custom_filename else "text_audio"
                    
                    # Save to file
                    output_file = save_direct_audio(audio_data, filename_prefix)
                    
                    if output_file:
                        st.success(f"✅ Audio file created: {os.path.basename(output_file)}")
                        
                        # Display audio player
                        with open(output_file, "rb") as f:
                            audio_bytes = f.read()
                            st.audio(audio_bytes)
                        
                        # Show file path
                        st.info(f"Audio saved to: {output_file}")
                    else:
                        st.error("Failed to save audio file")
                else:
                    st.error("Failed to generate audio. Please try a different voice.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def main():
    """Main application function"""
    setup_app()
    
    st.title(f"🎬 {APP_TITLE} v{VERSION}")
    
    # Display UTC time in the specified format
    current_time_utc = get_current_utc_time()
    st.write(f"📆 Current Time (UTC): **{current_time_utc}**")
    
    # Display output directory
    output_dir = ensure_output_directory()
    st.write(f"📁 Output Directory: **{output_dir}**")
    
    # Create tabs
    tab1, tab2 = st.tabs(["SRT to Audio", "Text to Audio"])
    
    with tab1:
        srt_to_audio_tab()
    
    with tab2:
        simple_text_to_audio_tab()
    
    # Footer
    st.markdown("---")
    st.caption(f"{APP_TITLE} v{VERSION} | Audio files saved to audio_output folder")
    st.caption(f"Selected voice: {st.session_state.selected_voice_display}")

if __name__ == "__main__":
    main()