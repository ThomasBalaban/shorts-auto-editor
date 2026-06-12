"""
Enhanced subtitle style definitions for different tracks including comic book onomatopoeia.
Contains configuration settings for various subtitle styles
that can be applied when embedding subtitles into videos.
"""

import random

class MicrophoneStyle:
    """Style for Track 2 (Microphone) - Teal text with darker gray stroke, using BubbleGum font"""
    FONT_NAME = "BubbleGum"
    FONT_SIZE = 20
    PRIMARY_COLOR = "&H00d2ff00"  # Teal color
    OUTLINE_COLOR = "&H00171717"  # Dark gray
    BACKGROUND_COLOR = "&H00000000"  # Transparent
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1  # Outline
    OUTLINE = 3
    SHADOW = 1
    ALIGNMENT = 2  # Bottom-center alignment
    MARGIN_V = 100  # Higher value = higher position from bottom
    MARGIN_L = 40  # Left margin
    MARGIN_R = 40  # Right margin

    @classmethod
    def get_style_string(cls, margin_v=None):
        """Generate the FFmpeg style string for subtitles. ``margin_v`` overrides
        the default vertical position (static-mode mic subtitles)."""
        mv = cls.MARGIN_V if margin_v is None else int(margin_v)
        return (
            f"FontName={cls.FONT_NAME},FontSize={cls.FONT_SIZE},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={mv},MarginL={cls.MARGIN_L},MarginR={cls.MARGIN_R}"
        )

class AnimatedMicrophoneStyle:
    """Style for Track 2 (Microphone) - Teal text with darker gray stroke, using BubbleGum font"""
    STYLE_NAME = "AnimatedMic"
    FONT_NAME = "BubbleGum"
    FONT_SIZE = 100
    PRIMARY_COLOR = "&H00d2ff00"  # Teal color
    OUTLINE_COLOR = "&H00171717"  # Dark gray
    BACKGROUND_COLOR = "&H00000000"  # Transparent
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1  # Outline
    OUTLINE = 10
    SHADOW = 1
    ALIGNMENT = 2  # Bottom-center alignment
    MARGIN_V = 640
    MARGIN_L = 40
    MARGIN_R = 40
    ENCODING = 1

    @classmethod
    def get_ass_style_string(cls, margin_v=None):
        """Generate the ASS style string. ``margin_v`` overrides the default
        vertical position (distance from the bottom in a 1920-tall frame) so the
        operator can drag where mic subtitles sit."""
        mv = cls.MARGIN_V if margin_v is None else int(margin_v)
        return (
            f"Style: {cls.STYLE_NAME},{cls.FONT_NAME},{cls.FONT_SIZE},"
            f"{cls.PRIMARY_COLOR},&H00000000,{cls.OUTLINE_COLOR},{cls.BACKGROUND_COLOR},"
            f"{cls.BOLD},{cls.ITALIC},0,0,100,100,0,0,{cls.BORDER_STYLE},{cls.OUTLINE},"
            f"{cls.SHADOW},{cls.ALIGNMENT},{cls.MARGIN_L},{cls.MARGIN_R},{mv},{cls.ENCODING}"
        )


class DesktopStyle:
    """Style for Track 3 (Desktop) - positioned below mic subtitles"""
    FONT_NAME = "Bold Marker"
    FONT_SIZE = 16
    PRIMARY_COLOR = "&H00FFFFFF"
    OUTLINE_COLOR = "&H00171717"
    BACKGROUND_COLOR = "&HFFFFFFFF"
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1
    OUTLINE = 2
    SHADOW = 1
    ALIGNMENT = 2  # Bottom-center
    MARGIN_V = 80  # Lower value = lower position (below mic track)
    MARGIN_L = 40  # Left margin
    MARGIN_R = 40  # Right margin

    @classmethod
    def get_style_string(cls, margin_v=None):
        """Generate the FFmpeg style string for subtitles. ``margin_v`` overrides
        the default vertical position so the operator can drag where game
        subtitles sit."""
        mv = cls.MARGIN_V if margin_v is None else int(margin_v)
        return (
            f"FontName={cls.FONT_NAME},FontSize={cls.FONT_SIZE},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={mv},MarginL={cls.MARGIN_L},MarginR={cls.MARGIN_R}"
        )


class OnomatopoeiaStyle:
    """Style for comic book onomatopoeia - top 1/3 of screen with dynamic sizing and random positioning"""
    FONT_NAME = "Bold Marker"
    BASE_FONT_SIZE = 140   # Base size, will be scaled by energy
    MAX_FONT_SIZE = 32    # Maximum size for loudest sounds
    PRIMARY_COLOR = "&H00FFFFFF"  # Bright yellow
    OUTLINE_COLOR = "&H00000000"  # Black outline
    BACKGROUND_COLOR = "&H00000000"  # Transparent
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1  # Outline
    OUTLINE = 4  # Thick outline for comic effect
    SHADOW = 2
    ALIGNMENT = 2  # Center alignment instead of left
    
    # Position constraints (top 1/3 of screen) - FIXED VALUES
    MIN_MARGIN_V = 500  # Distance from bottom (higher number = higher on screen)
    MAX_MARGIN_V = 500  # Maximum distance from bottom 
    MIN_MARGIN_L = 540   # Minimum left margin
    MAX_MARGIN_L = 540
    
    @classmethod
    def get_style_string_with_energy(cls, energy_level=0.5, random_position=True):
        """
        Generate the FFmpeg style string for onomatopoeia with dynamic sizing and positioning.
        
        Args:
            energy_level (float): Energy level from 0.0 to 1.0 for size scaling
            random_position (bool): Whether to randomize position
            
        Returns:
            str: FFmpeg style string
        """
        # Calculate font size based on energy (logarithmic scaling for better visual effect)
        size_range = cls.MAX_FONT_SIZE - cls.BASE_FONT_SIZE
        # Use square root for more dramatic scaling on loud sounds
        energy_factor = energy_level ** 0.7  # Slight curve for better distribution
        font_size = int(cls.BASE_FONT_SIZE + (size_range * energy_factor))
        
        # Simple positioning for better reliability
        if random_position:
            margin_v = random.randint(cls.MIN_MARGIN_V, cls.MAX_MARGIN_V)
            margin_l = random.randint(cls.MIN_MARGIN_L, cls.MAX_MARGIN_L)
        else:
            # Default centered position in top area
            margin_v = 300  # Fixed high position
            margin_l = 100  # Fixed left position
        
        return (
            f"FontName={cls.FONT_NAME},FontSize={font_size},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={margin_v},MarginL={margin_l},MarginR=40"
        )
    
    @classmethod
    def get_simple_style(cls):
        """Get a simple, reliable style for testing using class properties"""
        return (
            f"FontName={cls.FONT_NAME},FontSize={cls.BASE_FONT_SIZE},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={cls.MIN_MARGIN_V},MarginL={cls.MIN_MARGIN_L},MarginR=40"
        )
    
    @classmethod
    def get_random_color(cls):
        """Get a random vibrant color for variety in onomatopoeia"""
        colors = [
            "&H0000FFFF",  # Bright yellow
            "&H0000FF00",  # Bright green  
            "&H00FF0000",  # Bright blue
            "&H00FF00FF",  # Bright magenta
            "&H0000CCFF",  # Orange
            "&H00FFFF00",  # Cyan
        ]
        return random.choice(colors)
    
    @classmethod
    def get_style_string_with_color_and_energy(cls, energy_level=0.5, random_position=True, random_color=False):
        """
        Generate style string with optional random color and energy scaling.
        
        Args:
            energy_level (float): Energy level for size scaling
            random_position (bool): Whether to randomize position
            random_color (bool): Whether to use random colors
            
        Returns:
            str: FFmpeg style string
        """
        # Get base style
        style = cls.get_style_string_with_energy(energy_level, random_position)
        
        # Replace color if random color is requested
        if random_color:
            new_color = cls.get_random_color()
            style = style.replace(cls.PRIMARY_COLOR, new_color)
        
        return style


def generate_onomatopoeia_style(events):
    """
    Generate style strings for a list of onomatopoeia events with collision avoidance.
    
    Args:
        events (list): List of onomatopoeia events with 'energy' field
        
    Returns:
        list: List of style strings corresponding to each event
    """
    if not events:
        return []
    
    styles = []
    used_positions = []  # Track used positions to avoid overlap
    
    for event in events:
        energy = event.get('energy', 0.5)
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            # Generate random position
            margin_v = random.randint(OnomatopoeiaStyle.MIN_MARGIN_V, OnomatopoeiaStyle.MAX_MARGIN_V)
            margin_l = random.randint(OnomatopoeiaStyle.MIN_MARGIN_L, OnomatopoeiaStyle.MAX_MARGIN_L)
            
            # Check for collision with existing positions
            collision = False
            for used_v, used_l in used_positions:
                # Check if positions are too close (within 100 pixels)
                if abs(margin_v - used_v) < 100 and abs(margin_l - used_l) < 150:
                    collision = True
                    break
            
            if not collision:
                used_positions.append((margin_v, margin_l))
                break
                
            attempts += 1
        
        # Generate style with the position (collision or not after max attempts)
        style = OnomatopoeiaStyle.get_style_string_with_color_and_energy(
            energy_level=energy,
            random_position=False,  # We've already calculated position
            random_color=True
        )
        
        # Override the margins in the style string
        style = style.replace(f"MarginV={OnomatopoeiaStyle.MIN_MARGIN_V}", f"MarginV={margin_v}")
        style = style.replace(f"MarginL={OnomatopoeiaStyle.MIN_MARGIN_L}", f"MarginL={margin_l}")
        
        styles.append(style)
    
    return styles


# For backward compatibility with existing code
TRACK2_STYLE = MicrophoneStyle.get_style_string()
TRACK3_STYLE = DesktopStyle.get_style_string()

# New onomatopoeia style - default medium energy
ONOMATOPOEIA_STYLE = OnomatopoeiaStyle.get_style_string_with_energy(0.6)