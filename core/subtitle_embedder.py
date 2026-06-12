import os
import subprocess
from utils.subtitle_styles import (
    TRACK2_STYLE, TRACK3_STYLE, OnomatopoeiaStyle,
    MicrophoneStyle, DesktopStyle,
)

def embed_subtitles(input_video, output_video, track2_srt, track3_srt, onomatopoeia_srt, onomatopoeia_events, log,
                    mic_margin_v=None, game_margin_v=None):
    """
    Main subtitle embedding function - dynamically handles any combination of subtitle files.

    ``mic_margin_v`` / ``game_margin_v`` override the vertical position of the
    mic / game (desktop) SRT subtitles. (Animated mic subtitles are .ass and
    carry their position in the file itself, set in convert_to_srt.)
    """
    mic_style = (MicrophoneStyle.get_style_string(mic_margin_v)
                 if mic_margin_v is not None else TRACK2_STYLE)
    desktop_style = (DesktopStyle.get_style_string(game_margin_v)
                     if game_margin_v is not None else TRACK3_STYLE)
    # Check what subtitle files we have
    sub_files = {
        "mic": track2_srt,
        "desktop": track3_srt,
        "onomatopoeia": onomatopoeia_srt
    }
    
    valid_subs = {name: path for name, path in sub_files.items() if path and os.path.exists(path) and os.path.getsize(path) > 0}
    
    log(f"Subtitle availability - Track2 (Mic): {'mic' in valid_subs}, Track3 (Desktop): {'desktop' in valid_subs}, Onomatopoeia: {'onomatopoeia' in valid_subs}")
    
    if not valid_subs:
        log("No subtitle tracks available - copying video without subtitles")
        import shutil
        shutil.copy2(input_video, output_video)
        log(f"Video copied without subtitles: {os.path.basename(output_video)}")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log(f"Created output directory: {out_dir}")

    # Convert MKV to MP4 first if necessary
    temp_mp4 = None
    if input_video.lower().endswith('.mkv'):
        temp_mp4 = f"{os.path.splitext(output_video)[0]}_temp.mp4"
        log("Input is MKV format. Converting to MP4 first...")
        cmd_convert = [
            'ffmpeg', '-y', '-i', input_video,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
            temp_mp4
        ]
        log(f"Running conversion command: {' '.join(cmd_convert)}")
        subprocess.run(cmd_convert, check=True)
        input_video = temp_mp4

    # Build the filter string dynamically
    filter_parts = []
    
    # Process ASS files first as they often have complex styling
    for name, path in valid_subs.items():
        if path.lower().endswith('.ass'):
            path_fmt = path.replace('\\', '/')
            if os.name == 'nt':
                path_fmt = path_fmt.replace(':', r'\\:')
            filter_parts.append(f"ass='{path_fmt}'")

    # Then process SRT files
    for name, path in valid_subs.items():
        if path.lower().endswith('.srt'):
            path_fmt = path.replace('\\', '/')
            if os.name == 'nt':
                path_fmt = path_fmt.replace(':', r'\\:')
            
            style = desktop_style # Default (desktop/game track)
            if name == "mic":
                style = mic_style
            elif name == "onomatopoeia":
                style = OnomatopoeiaStyle.get_simple_style()

            filter_parts.append(f"subtitles='{path_fmt}':force_style='{style}'")
            
    video_filter = ",".join(filter_parts)

    # Construct the final ffmpeg command
    # Use CRF 10 to maintain high quality before the final trim
    cmd_embed = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', video_filter,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '10',
        '-c:a', 'copy',
        output_video
    ]
    
    log(f"Running dynamic embedding command: {' '.join(cmd_embed)}")
    subprocess.run(cmd_embed, check=True)
    
    log(f"All available subtitles embedded into {output_video} successfully\n")
        
    # Clean up temp MP4 file if created
    if temp_mp4 and os.path.exists(temp_mp4):
        try:
            os.remove(temp_mp4)
            log(f"Deleted temporary file: {temp_mp4}")
        except Exception as e:
            log(f"WARNING: Failed to delete temporary file {temp_mp4}: {e}")