#!/usr/bin/env python3
"""
TRMNL Image Webhook Uploader
Automatically uploads images from a directory to your TRMNL display
"""

import os
import io
import time
import json
import random
import logging
import requests
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Read version from VERSION file
def get_version():
    """Read version from VERSION file"""
    try:
        version_file = Path(__file__).parent / 'VERSION'
        if version_file.exists():
            return version_file.read_text().strip()
    except Exception:
        pass
    return "unknown"


CURRENT_VERSION = get_version()
VERSION_CHECK_URL = "https://api.github.com/repos/ExcuseMi/trmnl-image-webhook/releases/latest"


def check_for_updates():
    """Check if a newer version is available"""
    try:
        response = requests.get(VERSION_CHECK_URL, timeout=5)
        if response.status_code == 200:
            latest = response.json()
            latest_version = latest.get('tag_name', '').lstrip('v')

            if latest_version and latest_version != CURRENT_VERSION:
                logger.warning("=" * 70)
                logger.warning(f"UPDATE AVAILABLE: v{latest_version} (you have v{CURRENT_VERSION})")
                logger.warning(f"Release notes: {latest.get('html_url', '')}")
                logger.warning("Update: docker pull ghcr.io/excusemi/trmnl-image-webhook:latest")
                logger.warning("=" * 70)
    except Exception:
        # Silently fail - don't block startup for version check
        pass


class ImageUploader:
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    STATE_FILE = '/data/state.json'
    ORIENTATION_CACHE_FILE = '/data/orientation_cache.json'

    def __init__(self,
                 webhook_url: str,
                 images_dir: str,
                 interval_minutes: int,
                 selection_mode: str = 'random',
                 include_subfolders: bool = True,
                 display_width: int = 800,
                 display_height: int = 480,
                 layout: str = 'auto',
                 orientation_filter: str = 'any'):
        self.webhook_url = webhook_url
        self.images_dir = Path(images_dir)
        self.interval_seconds = interval_minutes * 60
        self.selection_mode = selection_mode
        self.include_subfolders = include_subfolders
        self.layout = layout
        self.orientation_filter = orientation_filter

        # Adjust dimensions based on layout
        if layout == 'portrait':
            # Swap dimensions for portrait mode
            self.display_width = min(display_width, display_height)
            self.display_height = max(display_width, display_height)
        elif layout == 'landscape':
            # Ensure landscape (wider than tall)
            self.display_width = max(display_width, display_height)
            self.display_height = min(display_width, display_height)
        else:
            # Auto - use as configured
            self.display_width = display_width
            self.display_height = display_height

        self.state = self._load_state()

        logger.info(f"Initialized uploader:")
        logger.info(f"  Images directory: {self.images_dir}")
        logger.info(f"  Display size: {self.display_width}x{self.display_height}")
        logger.info(f"  Layout: {layout}")
        logger.info(f"  Orientation filter: {orientation_filter}")
        logger.info(f"  Upload interval: {interval_minutes} minutes")
        logger.info(f"  Selection mode: {selection_mode}")
        logger.info(f"  Include subfolders: {include_subfolders}")

    def _load_state(self) -> dict:
        """Load state from file or create new state"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")

        return {
            'last_image': None,
            'current_index': 0,
            'shuffle_order': [],
            'last_upload': None
        }

    def _save_state(self):
        """Save current state to file"""
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state file: {e}")

    def _load_orientation_cache(self) -> dict:
        """Load orientation cache from file"""
        try:
            if os.path.exists(self.ORIENTATION_CACHE_FILE):
                with open(self.ORIENTATION_CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load orientation cache: {e}")
        return {}

    def _save_orientation_cache(self, cache: dict):
        """Save orientation cache to file"""
        try:
            os.makedirs(os.path.dirname(self.ORIENTATION_CACHE_FILE), exist_ok=True)
            with open(self.ORIENTATION_CACHE_FILE, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Could not save orientation cache: {e}")

    def get_image_files(self) -> List[Path]:
        """Get all supported image files from directory"""
        images = []

        if self.include_subfolders:
            for ext in self.SUPPORTED_FORMATS:
                images.extend(self.images_dir.rglob(f'*{ext}'))
                images.extend(self.images_dir.rglob(f'*{ext.upper()}'))
        else:
            for ext in self.SUPPORTED_FORMATS:
                images.extend(self.images_dir.glob(f'*{ext}'))
                images.extend(self.images_dir.glob(f'*{ext.upper()}'))

        # Sort for consistent ordering
        images = sorted(images)

        logger.info(f"Found {len(images)} images")

        # Filter by orientation if specified
        if self.orientation_filter != 'any':
            logger.info(f"Filtering images by orientation: {self.orientation_filter}")

            # Load orientation cache
            cache = self._load_orientation_cache()
            filtered_images = []
            cache_updated = False
            total = len(images)
            checked = 0

            for img_path in images:
                try:
                    # Cache key is relative path + modification time
                    rel_path = str(img_path.relative_to(self.images_dir))
                    mtime = img_path.stat().st_mtime
                    cache_key = f"{rel_path}:{mtime}"

                    # Check cache first
                    if cache_key in cache:
                        orientation = cache[cache_key]
                    else:
                        # Not in cache, need to check
                        checked += 1
                        if checked % 100 == 0:
                            logger.info(f"  Checking new/modified images: {checked}...")

                        with Image.open(img_path) as img:
                            # Apply EXIF rotation to get actual dimensions
                            from PIL import ImageOps
                            img = ImageOps.exif_transpose(img)
                            if img is None:
                                img = Image.open(img_path)

                            width, height = img.size

                            # Determine orientation
                            if width > height:
                                orientation = 'landscape'
                            elif height > width:
                                orientation = 'portrait'
                            else:
                                orientation = 'square'

                        # Update cache
                        cache[cache_key] = orientation
                        cache_updated = True

                    # Filter based on orientation
                    if self.orientation_filter == 'landscape' and orientation == 'landscape':
                        filtered_images.append(img_path)
                    elif self.orientation_filter == 'portrait' and orientation == 'portrait':
                        filtered_images.append(img_path)

                except Exception as e:
                    logger.debug(f"Could not check orientation of {img_path.name}: {e}")
                    # Include image if we can't determine orientation
                    filtered_images.append(img_path)

            # Save cache if updated
            if cache_updated:
                self._save_orientation_cache(cache)
                logger.info(f"  Updated orientation cache ({checked} new/modified images)")

            logger.info(f"Filtered to {len(filtered_images)} {self.orientation_filter} images")
            images = filtered_images

        return images

    def select_next_image(self, images: List[Path]) -> Optional[Path]:
        """Select next image based on selection mode"""
        if not images:
            logger.warning("No images found in directory")
            return None

        if self.selection_mode == 'random':
            return random.choice(images)

        elif self.selection_mode == 'sequential':
            # Get current index, wrap around if needed
            index = self.state.get('current_index', 0)
            if index >= len(images):
                index = 0

            selected = images[index]
            self.state['current_index'] = (index + 1) % len(images)
            return selected

        elif self.selection_mode == 'shuffle':
            # Create new shuffle order if needed
            image_names = [str(img.relative_to(self.images_dir)) for img in images]

            if (not self.state.get('shuffle_order') or
                    set(self.state['shuffle_order']) != set(image_names)):
                # New images or first run - create shuffle order
                self.state['shuffle_order'] = image_names.copy()
                random.shuffle(self.state['shuffle_order'])
                self.state['current_index'] = 0
                logger.info("Created new shuffle order")

            # Get next image from shuffle order
            index = self.state.get('current_index', 0)
            if index >= len(self.state['shuffle_order']):
                # End of shuffle - reshuffle
                random.shuffle(self.state['shuffle_order'])
                index = 0
                logger.info("Reshuffled images")

            selected_name = self.state['shuffle_order'][index]
            selected = self.images_dir / selected_name

            self.state['current_index'] = index + 1
            return selected

        elif self.selection_mode == 'newest':
            # Sort by modification time, newest first
            return max(images, key=lambda x: x.stat().st_mtime)

        elif self.selection_mode == 'oldest':
            # Sort by modification time, oldest first
            return min(images, key=lambda x: x.stat().st_mtime)

        else:
            logger.error(f"Unknown selection mode: {self.selection_mode}")
            return random.choice(images)

    def process_image(self, image_path: Path) -> Tuple[bytes, str]:
        """
        Process image for TRMNL e-ink display
        Isolates dithering to the image content only to prevent margin bleed.
        """
        try:
            with Image.open(image_path) as img:
                # 1. Standardize Orientation/Color
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
                if img is None:
                    img = Image.open(image_path)

                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # 2. Resize Content (Isolate logic from _process_single_image)
                margin = int(os.getenv('MARGIN', '0'))
                available_w = self.display_width - (2 * margin)
                available_h = self.display_height - (2 * margin)

                img_ratio = img.width / img.height
                display_ratio = available_w / available_h

                if img_ratio > display_ratio:
                    new_w, new_h = available_w, int(available_w / img_ratio)
                else:
                    new_h, new_w = available_h, int(available_h * img_ratio)

                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # 3. Dither ONLY the resized image
                img_l = img_resized.convert('L')
                bit_depth = int(os.getenv('BIT_DEPTH', '2'))
                use_dither = os.getenv('USE_DITHERING', 'true').lower() == 'true'

                if use_dither:
                    logger.info(f"  Dithering image content (Isolation Mode)")
                    if bit_depth == 1:
                        img_processed = img_l.convert('1', dither=Image.Dither.FLOYDSTEINBERG).convert('L')
                    else:
                        img_processed = self._dither_to_4_levels(img_l)
                else:
                    img_processed = img_l

                # 4. Create Clean Canvas (Pure background)
                border_style = os.getenv('BORDER_STYLE', 'white').lower()
                bg_color = 0 if border_style == 'black' else 255
                canvas = Image.new('L', (self.display_width, self.display_height), bg_color)

                # 5. Paste Dithered Content onto Clean Canvas
                x_offset = (self.display_width - new_w) // 2
                y_offset = (self.display_height - new_h) // 2
                canvas.paste(img_processed, (x_offset, y_offset))

                # 6. Generate final bytes
                if bit_depth == 1:
                    final_output = canvas.convert('1', dither=Image.Dither.NONE)
                    output = io.BytesIO()
                    final_output.save(output, format='PNG', optimize=True)
                    image_bytes = output.getvalue()
                else:
                    image_bytes = self._save_2bit_png(canvas)

                # --- DEBUG SAVING (The part that was missing) ---
                # Clean up old original files
                for old_file in Path('/data').glob('last_original.*'):
                    old_file.unlink()

                # Save original
                ext = image_path.suffix.lower()
                original_path = Path(f'/data/last_original{ext}')
                with open(image_path, 'rb') as src:
                    with open(original_path, 'wb') as dst:
                        dst.write(src.read())

                # Save processed
                processed_path = Path('/data/last_processed.png')
                with open(processed_path, 'wb') as f:
                    f.write(image_bytes)

                logger.info(f"  Saved debug images to /data/")
                # ------------------------------------------------

                return image_bytes, 'image/png'

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            with open(image_path, 'rb') as f:
                return f.read(), 'image/jpeg'

    def _dither_to_4_levels(self, img: Image.Image) -> Image.Image:
        """
        Apply Floyd-Steinberg dithering to 4 gray levels with a
        threshold to protect solid backgrounds/margins.
        """
        if img.mode != 'L':
            img = img.convert('L')

        width, height = img.size
        pixels = list(img.getdata())
        img_array = [list(pixels[i * width:(i + 1) * width]) for i in range(height)]

        for y in range(height):
            for x in range(width):
                old_pixel = img_array[y][x]

                # THRESHOLD FIX: If pixel is nearly white or black,
                # snap it to pure and skip error distribution.
                if old_pixel >= 254:
                    img_array[y][x] = 255
                    continue
                if old_pixel <= 1:
                    img_array[y][x] = 0
                    continue

                # Find nearest level (0, 85, 170, 255)
                if old_pixel < 43:
                    new_pixel = 0
                elif old_pixel < 128:
                    new_pixel = 85
                elif old_pixel < 213:
                    new_pixel = 170
                else:
                    new_pixel = 255

                img_array[y][x] = new_pixel
                error = old_pixel - new_pixel

                # Distribute error to neighboring pixels
                if x + 1 < width:
                    img_array[y][x + 1] = max(0, min(255, img_array[y][x + 1] + error * 7 // 16))
                if y + 1 < height:
                    if x > 0:
                        img_array[y + 1][x - 1] = max(0, min(255, img_array[y + 1][x - 1] + error * 3 // 16))
                    img_array[y + 1][x] = max(0, min(255, img_array[y + 1][x] + error * 5 // 16))
                    if x + 1 < width:
                        img_array[y + 1][x + 1] = max(0, min(255, img_array[y + 1][x + 1] + error * 1 // 16))

        flat_pixels = [pixel for row in img_array for pixel in row]
        dithered = Image.new('L', (width, height))
        dithered.putdata(flat_pixels)
        return dithered
    def _save_2bit_png(self, img: Image.Image) -> bytes:
        """
        Save grayscale image as 2-bit PNG using pypng.

        Quantizes to 4 gray levels: 0 (black), 85 (dark), 170 (light), 255 (white)
        Creates native 2-bit grayscale PNG (color type 0, bit depth 2)
        """
        try:
            import png
        except ImportError:
            raise ImportError("pypng required for 2-bit PNG support. Run: pip install pypng")

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        width, height = img.size
        pixels = list(img.getdata())

        # Quantize to 4 levels (0, 1, 2, 3)
        rows = []
        for y in range(height):
            row = []
            for x in range(width):
                pixel = pixels[y * width + x]
                # Map 0-255 to 0-3
                if pixel < 64:
                    value = 0  # Black
                elif pixel < 128:
                    value = 1  # Dark gray
                elif pixel < 192:
                    value = 2  # Light gray
                else:
                    value = 3  # White
                row.append(value)
            rows.append(row)

        # Write 2-bit grayscale PNG with maximum compression
        output = io.BytesIO()
        writer = png.Writer(
            width=width,
            height=height,
            greyscale=True,
            bitdepth=2,
            compression=9  # Maximum compression to stay under TRMNL's ~50KB limit
        )
        writer.write(output, rows)

        return output.getvalue()

    def _add_label(self, img: Image.Image, image_path: Path, label_mode: str) -> Image.Image:
        """Add filename or path label to image"""
        from PIL import ImageDraw, ImageFont

        # Get label text
        if label_mode == 'filename':
            label_text = image_path.name
        elif label_mode == 'path':
            label_text = str(image_path.relative_to(self.images_dir))
        else:
            return img

        # Create a copy to draw on
        labeled = img.copy()
        draw = ImageDraw.Draw(labeled)

        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position at bottom with padding
        padding = 10
        x = padding
        y = self.display_height - text_height - padding - 5

        # Draw black background rectangle
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
            fill=0  # Black background
        )

        # Draw white text
        draw.text((x, y), label_text, fill=255, font=font)

        logger.info(f"  Added label: {label_text}")
        return labeled

    def _process_single_image(self, img: Image.Image) -> Image.Image:
        """
        Scale image to fit display while maintaining aspect ratio
        Centers with configurable border style and optional margin
        ALWAYS returns exactly display_width x display_height
        """
        # Get margin setting
        margin = int(os.getenv('MARGIN', '0'))
        margin = max(0, min(100, margin))  # Clamp to 0-100

        # Calculate available space after margin
        available_width = self.display_width - (2 * margin)
        available_height = self.display_height - (2 * margin)

        # Calculate scaling to fit within available space
        img_ratio = img.width / img.height
        display_ratio = available_width / available_height

        if img_ratio > display_ratio:
            # Image is wider - scale by width
            new_width = available_width
            new_height = int(available_width / img_ratio)
        else:
            # Image is taller - scale by height
            new_height = available_height
            new_width = int(available_height * img_ratio)

        # Resize image with high quality
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Get border style
        border_style = os.getenv('BORDER_STYLE', 'white').lower()

        # Create canvas based on border style
        if border_style == 'black':
            # Black borders for classic framing
            canvas = Image.new('RGB', (self.display_width, self.display_height), (0, 0, 0))
        elif border_style == 'blur':
            # Blurred background - scale original to fill, blur heavily
            canvas = img.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            from PIL import ImageFilter
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=20))
        else:
            # White borders (default, clean look)
            canvas = Image.new('RGB', (self.display_width, self.display_height), (255, 255, 255))

        # Center image on canvas (accounting for margin)
        x_offset = (self.display_width - new_width) // 2
        y_offset = (self.display_height - new_height) // 2
        canvas.paste(img_resized, (x_offset, y_offset))

        if margin > 0:
            logger.info(f"  Scaled: {img.size} → {new_width}x{new_height}, centered with {margin}px margin")
        else:
            logger.info(
                f"  Scaled: {img.size} → {new_width}x{new_height}, centered on {self.display_width}x{self.display_height} canvas")

        # Verify exact size
        assert canvas.size == (self.display_width, self.display_height), \
            f"Canvas size mismatch: {canvas.size} != {(self.display_width, self.display_height)}"

        return canvas

    def _process_fill_image(self, img: Image.Image) -> Image.Image:
        """
        Scale and crop image to fill entire display
        May crop parts of the image to maintain aspect ratio
        ALWAYS returns exactly display_width x display_height
        """
        # Calculate scaling to fill display
        img_ratio = img.width / img.height
        display_ratio = self.display_width / self.display_height

        if img_ratio > display_ratio:
            # Image is wider - scale by height and crop width
            new_height = self.display_height
            new_width = int(self.display_height * img_ratio)
        else:
            # Image is taller - scale by width and crop height
            new_width = self.display_width
            new_height = int(self.display_width / img_ratio)

        # Resize image with high quality
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to exact display size
        x_offset = (new_width - self.display_width) // 2
        y_offset = (new_height - self.display_height) // 2
        img_cropped = img_resized.crop((
            x_offset,
            y_offset,
            x_offset + self.display_width,
            y_offset + self.display_height
        ))

        logger.info(
            f"  Scaled: {img.size} → {new_width}x{new_height}, cropped to {self.display_width}x{self.display_height}")

        # Verify exact size
        assert img_cropped.size == (self.display_width, self.display_height), \
            f"Cropped size mismatch: {img_cropped.size} != {(self.display_width, self.display_height)}"

        return img_cropped

    def upload_image(self, image_path: Path) -> bool:
        """Upload image to TRMNL webhook (or skip if dry run)"""
        try:
            logger.info(f"Uploading: {image_path.name}")

            # Process image for optimal display
            image_data, content_type = self.process_image(image_path)

            # Check if dry run mode
            dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'

            if dry_run:
                logger.info(f"✓ DRY RUN: Skipped upload of {image_path.name}")
                logger.info(f"  Would upload: {len(image_data) / 1024:.1f}KB {content_type}")
                logger.info(f"  Check /data/last_processed.png to see result")
                return True

            # Validate image size
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > 5:
                logger.error(f"✗ Image too large: {size_mb:.2f}MB (max 5MB)")
                return False

            logger.info(f"  Sending {len(image_data) / 1024:.1f}KB {content_type.split('/')[-1].upper()} to TRMNL")

            # Upload to TRMNL
            response = requests.post(
                self.webhook_url,
                data=image_data,
                headers={'Content-Type': content_type},
                timeout=30
            )

            # Check response
            if response.status_code == 200:
                logger.info(f"✓ Successfully uploaded {image_path.name}")
                logger.info(f"  Response: {response.status_code}")
            elif response.status_code == 422:
                logger.error(f"✗ Upload rejected (422): Image format or size invalid")
                return False
            elif response.status_code == 429:
                logger.error(f"✗ Rate limited (429): Too many uploads (max 12/hour)")
                return False
            else:
                logger.error(f"✗ Upload failed with status {response.status_code}")
                return False

            response.raise_for_status()

            # Update state
            self.state['last_image'] = str(image_path.relative_to(self.images_dir))
            self.state['last_upload'] = datetime.now().isoformat()
            self._save_state()

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Upload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error uploading image: {e}")
            return False

    def run(self):
        """Main loop - upload images at specified interval"""
        logger.info("Starting image uploader...")
        logger.info(f"Next upload in {self.interval_seconds} seconds")

        # Upload immediately on start
        images = self.get_image_files()
        if images:
            next_image = self.select_next_image(images)
            if next_image:
                self.upload_image(next_image)

        # Continue with scheduled uploads
        while True:
            try:
                time.sleep(self.interval_seconds)

                # Refresh image list each time (in case new images added)
                images = self.get_image_files()

                if not images:
                    logger.warning("No images found, waiting for next interval...")
                    continue

                next_image = self.select_next_image(images)
                if next_image and next_image.exists():
                    self.upload_image(next_image)
                else:
                    logger.warning(f"Selected image not found: {next_image}")

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    # Check for updates
    check_for_updates()

    # Get configuration from environment variables
    webhook_url = os.getenv('WEBHOOK_URL')
    images_dir = os.getenv('IMAGES_DIR', '/images')
    interval_minutes = int(os.getenv('INTERVAL_MINUTES', '60'))
    selection_mode = os.getenv('SELECTION_MODE', 'random').lower()
    include_subfolders = os.getenv('INCLUDE_SUBFOLDERS', 'true').lower() == 'true'
    display_width = int(os.getenv('DISPLAY_WIDTH', '800'))
    display_height = int(os.getenv('DISPLAY_HEIGHT', '480'))
    layout = os.getenv('LAYOUT', 'auto').lower()
    orientation_filter = os.getenv('ORIENTATION_FILTER', 'any').lower()

    # Validate configuration
    if not webhook_url:
        logger.error("WEBHOOK_URL environment variable is required!")
        logger.error("Get your webhook URL from TRMNL plugin settings")
        exit(1)

    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        logger.error("Make sure to mount a directory to /images")
        exit(1)

    # Validate selection mode
    valid_modes = ['random', 'sequential', 'shuffle', 'newest', 'oldest']
    if selection_mode not in valid_modes:
        logger.error(f"Invalid SELECTION_MODE: {selection_mode}")
        logger.error(f"Valid modes: {', '.join(valid_modes)}")
        exit(1)

    # Validate layout
    valid_layouts = ['auto', 'landscape', 'portrait']
    if layout not in valid_layouts:
        logger.error(f"Invalid LAYOUT: {layout}")
        logger.error(f"Valid layouts: {', '.join(valid_layouts)}")
        exit(1)

    # Validate orientation filter
    valid_filters = ['any', 'landscape', 'portrait']
    if orientation_filter not in valid_filters:
        logger.error(f"Invalid ORIENTATION_FILTER: {orientation_filter}")
        logger.error(f"Valid filters: {', '.join(valid_filters)}")
        exit(1)

    # Create and run uploader
    uploader = ImageUploader(
        webhook_url=webhook_url,
        images_dir=images_dir,
        interval_minutes=interval_minutes,
        selection_mode=selection_mode,
        include_subfolders=include_subfolders,
        display_width=display_width,
        display_height=display_height,
        layout=layout,
        orientation_filter=orientation_filter
    )

    uploader.run()


if __name__ == '__main__':
    main()