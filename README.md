# TRMNL Image Webhook

Automatically upload images from your photo collection to your TRMNL e-ink display with proper 1-bit dithering for beautiful grayscale rendering.

## Features

- 📸 **Automatic Image Processing** - Scales and converts photos to 1-bit e-ink format
- 🎨 **Floyd-Steinberg Dithering** - Professional halftone effect for smooth gradients
- 🎲 **Multiple Selection Modes** - Random, sequential, shuffle, newest, or oldest
- 📁 **Subfolder Support** - Organize photos in folders and subfolders
- 🏷️ **Optional Labels** - Show filename or path on images
- 🐳 **Docker Ready** - Easy deployment with docker-compose
- 💾 **State Management** - Remembers position for sequential/shuffle modes
- 🔍 **Debug Mode** - Saves processed images for inspection
- 🧪 **Dry Run** - Test without uploading

## Quick Start

### 1. Get Your Webhook URL

1. Log into [TRMNL](https://usetrmnl.com)
2. Go to Plugins > Webhook Image
3. Click "Add to my plugins"
4. Copy your webhook URL

### 2. Set Up Configuration

```bash
# Clone or download this repository
cd trmnl-image-webhook

# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

Minimum required config:
```bash
WEBHOOK_URL=https://usetrmnl.com/api/plugin_settings/YOUR-UUID/image
IMAGES_PATH=/path/to/your/photos
```

### 3. Run with Docker

```bash
docker-compose up -d
```

That's it! Your TRMNL will start showing photos from your collection.

## Configuration

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `WEBHOOK_URL` | Your TRMNL webhook URL | `https://usetrmnl.com/api/...` |
| `IMAGES_PATH` | Path to your photos | `./images` or `/home/user/Photos` |

### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY_WIDTH` | `800` | Display width (OG: 800, Plus: 1280) |
| `DISPLAY_HEIGHT` | `480` | Display height (OG: 480, Plus: 800) |
| `INTERVAL_MINUTES` | `60` | Minutes between uploads |
| `SELECTION_MODE` | `random` | How to pick images (see below) |
| `INCLUDE_SUBFOLDERS` | `true` | Include images from subdirectories |
| `USE_DITHERING` | `true` | Apply Floyd-Steinberg dithering |
| `IMAGE_LABEL` | `none` | Show label (none/filename/path) |
| `DRY_RUN` | `false` | Test mode - don't upload |

### Selection Modes

- **random** - Pick any image randomly
- **sequential** - Go through images A-Z, remembers position
- **shuffle** - Random order, each image once before reshuffling
- **newest** - Always show most recently modified image
- **oldest** - Show oldest image first

### Image Labels

Add text overlay to images:

- **none** - No label (default)
- **filename** - Show just the filename
- **path** - Show relative path from images directory

Example with label:
```bash
IMAGE_LABEL=path
```

## Image Processing

### What Happens to Your Photos

1. **Scaling** - Resized to fit display (800x480 or 1280x800)
2. **Grayscale** - Converted to grayscale
3. **Dithering** - Floyd-Steinberg dithering applied for smooth gradients
4. **1-bit Conversion** - Pure black and white (2 colors)
5. **PNG Export** - Optimized 1-bit PNG (~20-40KB)

### Why Dithering?

TRMNL displays are 1-bit (pure black and white). Dithering creates the illusion of grayscale by using patterns of black and white dots - like newspaper photos. This makes photos look much better than simple thresholding.

**With dithering:**
- Smooth gradients in sky, skin tones, etc.
- Details visible in shadows and highlights
- Professional halftone appearance

**Without dithering:**
- Harsh black/white contrast
- Loss of detail
- Posterized look

## Examples

### Basic Setup

```bash
# .env
WEBHOOK_URL=https://usetrmnl.com/api/plugin_settings/abc-123/image
IMAGES_PATH=/home/user/Photos
INTERVAL_MINUTES=60
SELECTION_MODE=random
INCLUDE_SUBFOLDERS=true
USE_DITHERING=true
```

### Photo Album (Sequential)

```bash
SELECTION_MODE=sequential
INTERVAL_MINUTES=120
IMAGE_LABEL=filename
```

### Slideshow (Shuffle)

```bash
SELECTION_MODE=shuffle
INTERVAL_MINUTES=30
INCLUDE_SUBFOLDERS=true
```

### Latest Photo Display

```bash
SELECTION_MODE=newest
INTERVAL_MINUTES=15
IMAGE_LABEL=path
```

## Deployment

### Docker Compose (Recommended)

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f trmnl-image-webhook

# Stop
docker-compose down

# Restart after config changes
docker-compose restart
```

One-liner to update and restart the application after setting the .env
```bash
git pull && docker compose down && docker compose build && docker compose up -d
```



## Debugging

### Dry Run Mode

Test without uploading:

```bash
DRY_RUN=true
docker-compose restart
```

### Common Issues

**No images found**

- Check `IMAGES_PATH` is correct
- Set `INCLUDE_SUBFOLDERS=true` if images are in subdirectories
- Verify image formats (PNG, JPG, JPEG, BMP, GIF supported)

**Images not displaying on TRMNL**

- Check device WiFi connection
- Verify webhook URL is correct
- Try "Force Refresh" in TRMNL plugin settings
- Check `data/last_processed.png` looks correct

**Rate limited (429 error)**

- TRMNL allows max 12 uploads per hour
- Increase `INTERVAL_MINUTES` to 60 or higher

**Upload rejected (422 error)**

- Image may be corrupted
- File over 5MB limit (shouldn't happen with processing)
- Try different source image

## Technical Details

### Supported Image Formats

Input: PNG, JPEG, JPG, BMP, GIF
Output: 1-bit PNG


### Display Sizes

**TRMNL OG:**
```bash
DISPLAY_WIDTH=800
DISPLAY_HEIGHT=480
```


## Requirements

- Docker & Docker Compose
- Network access to TRMNL API
- Directory of images (local or mounted)

## License

MIT License - see LICENSE file
## Support

For issues with:
- **This tool**: Open a GitHub issue
- **TRMNL device/service**: Contact TRMNL support
- **Docker/deployment**: Check Docker logs first

