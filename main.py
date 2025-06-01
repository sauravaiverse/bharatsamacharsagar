import os
import random
import time
import requests
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv
import json

# --- IMPORTS FOR OAuth CLIENT ID (USER CREDENTIALS) ---
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
# --- END OAuth IMPORTS ---

# Original imports for Google Blogger API (build, HttpError)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import base64
import logging
from datetime import datetime
import unicodedata
import platform
import sys # Added for find_system_font more robustly

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blog_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# IMPORTANT: How to run this script:
# 1. Ensure you have a .env file with GNEWS_API_KEY, GEMINI_API_KEY, and TOGETHER_API_KEY (for local testing).
# 2. Before running main.py for the first time, run generate_token.py (from Part 2) to create token_blogger.json.
# 3. Install required libraries: pip install python-dotenv requests Pillow google-generativeai google-api-python-client google-auth-httplib2 google-auth-oauthlib
# 4. Run from terminal: python your_script_name.py

# Load environment variables (for local testing)
load_dotenv()

# --- CONFIGURATION ---
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY') # NEW: Together AI API Key

CATEGORIES = ["technology", "health", "sports", "business", "entertainment"]
NUM_SOURCE_ARTICLES_TO_AGGREGATE = 5
LANGUAGE = 'hi'  # Changed from 'en' to 'hi' for Hindi

# NEW: Together AI Model Configuration
TOGETHER_DIFFUSION_MODEL = "black-forest-labs/FLUX.1-schnell-Free" # Specified by user

BRANDING_LOGO_PATH = os.getenv('BRANDING_LOGO_PATH', None)
IMAGE_OUTPUT_FOLDER = "transformed_images"
BLOG_OUTPUT_FOLDER = "blog_drafts"

# Hindi category names for better SEO and user experience
HINDI_CATEGORIES = {
    "technology": "टेक्नोलॉजी",
    "health": "स्वास्थ्य",
    "sports": "खेल",
    "business": "व्यापार",
    "entertainment": "मनोरंजन"
}

# --- BLOGGER AUTHENTICATION CONFIGURATION ---
BLOGGER_BLOG_ID = os.getenv('BLOGGER_BLOG_ID', '8169847264446388236') # Apni blog ID yahan daalo, ya .env mein.

# These will hold the JSON strings directly from GitHub Secrets OR be read from local files
GOOGLE_CLIENT_SECRETS_JSON = os.getenv('GOOGLE_CLIENT_SECRETS_JSON')
GOOGLE_OAUTH_TOKEN_JSON = os.getenv('GOOGLE_OAUTH_TOKEN_JSON')

# Define Blogger Scopes
BLOGGER_SCOPES = ['https://www.googleapis.com/auth/blogger']

# --- LLM Retry Configuration ---
LLM_MAX_RETRIES = 5
LLM_INITIAL_RETRY_DELAY_SECONDS = 5

# --- Enhanced font configuration with fallbacks ---
FONT_PATHS = {
    'mac': [
        "/Library/Fonts/NotoSansDevanagari-Regular.ttf", # Preferred for Hindi
        "/System/Library/Fonts/Supplemental/Mangal.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFProText-Regular.ttf"
    ],
    'windows': [
        "C:/Windows/Fonts/Mangal.ttf", # Preferred for Hindi
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    ],
    'linux': [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", # Preferred for Hindi
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    ]
}

def find_system_font():
    """Find the best available font for the current system, prioritizing Devanagari for Hindi."""
    system = platform.system().lower()
    
    if 'darwin' in system:
        font_list = FONT_PATHS['mac']
    elif 'windows' in system:
        font_list = FONT_PATHS['windows']
    else: # Assume Linux/Unix-like (GitHub Actions uses Ubuntu which is Linux)
        font_list = FONT_PATHS['linux']

    for font_path in font_list:
        if os.path.exists(font_path):
            logger.info(f"Using font: {font_path}")
            return font_path

    logger.warning("No suitable system fonts found, using PIL default. Text quality on images may be low.")
    return None

DEFAULT_FONT_PATH = find_system_font()

# Create necessary directories
for folder in [IMAGE_OUTPUT_FOLDER, BLOG_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(folder, cat), exist_ok=True)

# Setup Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    RESEARCH_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
    CONTENT_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
else:
    logger.error("GEMINI_API_KEY not set. Gemini functions will not work.")
    RESEARCH_MODEL = None # Set to None if not available
    CONTENT_MODEL = None # Set to None if not available

# --- NEW/MODIFIED FUNCTION FOR OAuth CLIENT ID AUTHENTICATION ---
def get_blogger_oauth_credentials():
    """
    Obtains OAuth 2.0 credentials for Blogger API using client_secrets.json and token_blogger.json.
    Prioritizes environment variables (for CI/CD) then local files (for development).
    """
    creds = None
    CLIENT_SECRETS_FILE = 'client_secrets.json' # Local file name (for local testing)
    TOKEN_FILE = 'token_blogger.json' # Local file name (for local testing)

    # 1. Sabse pehle GitHub Secrets (Environment Variables) se load karne ki koshish karo
    if GOOGLE_OAUTH_TOKEN_JSON:
        try:
            token_info = json.loads(GOOGLE_OAUTH_TOKEN_JSON)
            creds = UserCredentials.from_authorized_user_info(token_info, BLOGGER_SCOPES)
            logger.info("INFO: Blogger OAuth token loaded from environment variable (GitHub Secret).")
        except json.JSONDecodeError as e:
            logger.error(f"ERROR: GOOGLE_OAUTH_TOKEN_JSON is not valid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"ERROR: Could not load Blogger OAuth token from env var: {e}")
            return None

    # Agar credentials valid hain lekin expired hain, toh refresh karne ki koshish karo
    if creds and creds.expired and creds.refresh_token:
        logger.info("INFO: Blogger OAuth token expired, attempting to refresh...")
        try:
            creds.refresh(Request())
            logger.info("INFO: Blogger OAuth token refreshed successfully.")
            # Important: If refreshed, the token_info might change (new expiry).
            # If running locally (not CI), you might want to save it back.
            if not os.getenv('CI'): # Check if not running in CI (e.g., GitHub Actions)
                with open(TOKEN_FILE, 'w') as token_file:
                    token_file.write(creds.to_json())
                logger.info(f"INFO: Refreshed Blogger OAuth token saved to '{TOKEN_FILE}'.")
        except Exception as e:
            logger.error(f"ERROR: Failed to refresh Blogger OAuth token: {e}. You might need to re-authenticate manually by deleting {TOKEN_FILE}.")
            creds = None

    # Agar koi valid credentials environment variable se nahi mile (ya refresh fail hua),
    # aur hum GitHub Actions (CI environment) mein nahi hain, toh local files aur interactive flow try karo.
    # 'CI' env var is usually 'true' in GitHub Actions. So, this block will be skipped in Actions.
    if not creds and not os.getenv('CI'):
        # Local token file se load karne ki koshish karo
        if os.path.exists(TOKEN_FILE):
            try:
                creds = UserCredentials.from_authorized_user_file(TOKEN_FILE, BLOGGER_SCOPES)
                logger.info(f"INFO: Blogger OAuth token loaded from local file '{TOKEN_FILE}'.")
            except Exception as e:
                logger.warning(f"WARNING: Could not load Blogger OAuth token from local file '{TOKEN_FILE}': {e}. Will re-authenticate.")
                creds = None

        # Agar ab bhi credentials nahi mile, toh local interactive OAuth flow initiate karo
        if not creds:
            # Client secrets ko environment variable se load karne ki koshish karo (agar .env mein set hain)
            # ya phir local client_secrets.json file se load karo.
            client_config_info = {}
            if GOOGLE_CLIENT_SECRETS_JSON: # Agar GOOGLE_CLIENT_SECRETS_JSON .env mein set hai
                try:
                    client_config_info = json.loads(GOOGLE_CLIENT_SECRETS_JSON)
                    logger.info("INFO: Client secrets loaded from environment variable (local .env).")
                except json.JSONDecodeError as e:
                    logger.critical(f"CRITICAL ERROR: GOOGLE_CLIENT_SECRETS_JSON in env is not valid JSON: {e}")
                    return None
            elif os.path.exists(CLIENT_SECRETS_FILE): # Agar .env mein nahi hai, toh local file try karo
                try:
                    with open(CLIENT_SECRETS_FILE, 'r') as f:
                        client_config_info = json.load(f)
                    logger.info(f"INFO: Client secrets loaded from local file '{CLIENT_SECRETS_FILE}'.")
                except Exception as e:
                    logger.critical(f"CRITICAL ERROR: Could not load client secrets from '{CLIENT_SECRETS_FILE}': {e}")
                    return None
            else:
                logger.critical(f"CRITICAL ERROR: No client secrets found (neither in GOOGLE_CLIENT_SECRETS_JSON env nor local '{CLIENT_SECRETS_FILE}'). Cannot perform OAuth flow.")
                return None


            logger.info(f"INFO: Initiating interactive OAuth flow for Blogger. Please follow browser instructions.")
            flow = InstalledFlow.from_client_config(client_config_info, BLOGGER_SCOPES)
            try:
                # Yeh browser kholega user interaction ke liye (GitHub Actions mein nahi chalega)
                creds = flow.run_local_server(port=0, prompt='consent', authorization_prompt_message='Please authorize this application to access your Blogger account.')
                with open(TOKEN_FILE, 'w') as token_file:
                    token_file.write(creds.to_json())
                logger.info(f"INFO: New token saved to '{TOKEN_FILE}'.")
            except Exception as e:
                logger.error(f"ERROR during local OAuth flow: {e}")
                return None

    if creds and creds.valid:
        logger.info("INFO: Valid Blogger OAuth credentials obtained successfully.")
    else:
        logger.error("ERROR: Could not obtain valid Blogger OAuth credentials. Posting to Blogger will likely fail.")
    return creds

def validate_environment():
    """Validate that all required environment variables and dependencies are set"""
    errors = []

    if not GNEWS_API_KEY:
        errors.append("GNEWS_API_KEY not found in environment variables.")

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY not found in environment variables. Gemini functions will be skipped.")
    
    if not TOGETHER_API_KEY: # NEW: Check for Together AI key
        errors.append("TOGETHER_API_KEY not found in environment variables. Together AI image generation will be skipped.")

    # Validate Blogger API credentials
    if not BLOGGER_BLOG_ID:
        errors.append("BLOGGER_BLOG_ID not set. Cannot post to Blogger.")

    try:
        import PIL
        import google.generativeai
        from google.api_core import exceptions
        import requests
        import googleapiclient.discovery
        import google_auth_oauthlib.flow
        import google.oauth2.credentials
    except ImportError as e:
        errors.append(f"Missing required package: {e}. Please run 'pip install python-dotenv requests Pillow google-generativeai google-api-python-client google-auth-httplib2 google-auth-oauthlib'.")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    logger.info("Environment validation passed.")
    return True

def sanitize_filename(filename):
    """Create a safe filename from any string"""
    # Normalize unicode characters to their closest ASCII equivalents
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Replace invalid characters with underscore, then strip extra underscores/dashes at ends
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in filename).strip()
    safe_title = re.sub(r'[_ -]+', '_', safe_title).lower() # Ensure single underscores and lowercase
    return safe_title[:100] # Truncate to a reasonable length for file paths

def fetch_gnews_articles(category, max_articles_to_fetch=10, max_retries=3):
    """Fetches articles from GNews API with retry logic"""
    url = f'https://gnews.io/api/v4/top-headlines'
    params = {
        'category': category,
        'lang': LANGUAGE,
        'token': GNEWS_API_KEY,
        'max': max_articles_to_fetch # Request up to this many articles
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching up to {max_articles_to_fetch} articles for {category} (attempt {attempt + 1})...")
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = resp.json()
            articles = data.get('articles', [])

            if not articles:
                logger.warning(f"No articles found for category {category} from GNews API.")
                return []

            # Select unique articles based on URL to avoid duplicates if API sends similar ones
            unique_articles = {article['url']: article for article in articles}.values()

            selected_articles = list(unique_articles)[:max_articles_to_fetch] # Cap at requested max
            logger.info(f"Successfully fetched {len(selected_articles)} articles for {category}.")
            return selected_articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching articles for {category} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch articles for {category} after {max_retries} attempts")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from GNews API: {e}")
            return []

def aggregate_articles(articles_list, category):
    """
    Aggregates data from multiple articles to create a consolidated view
    for a single, unique blog post.
    Note: Image URL finding removed as featured image will be AI generated.
    """
    if not articles_list:
        logger.warning(f"No articles provided for aggregation in category {category}.")
        return None

    consolidated_content = []
    consolidated_descriptions = []
    titles = []
    competitor_domains = set()
    primary_source_url_for_disclaimer = None # To link back to one source in the disclaimer

    # Sort articles by content length to prioritize more substantive sources for aggregation
    sorted_articles = sorted(articles_list, key=lambda x: len(x.get('content', '')), reverse=True)

    for i, article in enumerate(sorted_articles):
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        source_url = article.get('url', '')
        source_domain = article.get('source', {}).get('url', '').replace('https://', '').replace('http://', '').split('/')[0]

        if title: titles.append(title)
        if description: consolidated_descriptions.append(description)

        # Only add content if it's substantial and not a placeholder "[Removed]"
        if content and content.strip() != '[Removed]' and len(content.strip()) > 50:
            # Append content with a clear source identifier for the AI
            consolidated_content.append(f"### Source: {title}\n\n{content}")
            if not primary_source_url_for_disclaimer: # Take the first substantial article's URL for disclaimer
                primary_source_url_for_disclaimer = source_url

        if source_domain:
            competitor_domains.add(source_domain)

    # Formulate a consolidated topic based on primary titles
    consolidated_topic = titles[0] if titles else f"Recent Developments in {HINDI_CATEGORIES.get(category, category).capitalize()}"
    if len(titles) > 1:
        combined_titles_string = " ".join(titles[:min(3, len(titles))])
        consolidated_topic = f"व्यापक दृष्टिकोण: {combined_titles_string}"
        if len(consolidated_topic) > 150:
            consolidated_topic = consolidated_topic[:150] + "..."
        consolidated_topic = consolidated_topic.replace('...', '...').strip()

    # Use a dummy description if none are available
    if not consolidated_descriptions:
        consolidated_descriptions.append(f"Recent developments in {HINDI_CATEGORIES.get(category, category)}.")

    return {
        "consolidated_topic": consolidated_topic,
        "combined_content": "\n\n---\n\n".join(consolidated_content) if consolidated_content else "No substantial content found from sources. AI will generate based on topic.",
        "combined_description": " ".join(consolidated_descriptions)[:300].strip(),
        "competitors": list(competitor_domains),
        "primary_source_url": primary_source_url_for_disclaimer if primary_source_url_for_disclaimer else articles_list[0]['url'] if articles_list else 'https://news.example.com/source-unavailable'
    }

# --- NEW: Function to generate image using Together AI ---
def generate_image_from_together_ai(prompt_text, max_retries=3, initial_delay=5):
    """
    Generates an image using Together AI's FLUX model via its API.
    Returns raw image bytes.
    """
    if not TOGETHER_API_KEY:
        logger.error("TOGETHER_API_KEY is not set. Cannot generate image.")
        return None

    api_url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Format the prompt for FLUX model with aspect ratio
    formatted_prompt = f"{prompt_text} --ar 16:9"

    payload = {
        "model": TOGETHER_DIFFUSION_MODEL,
        "prompt": formatted_prompt,
        "n": 1,
        "size": "1024x576"  # 16:9 aspect ratio
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating image from Together AI model '{TOGETHER_DIFFUSION_MODEL}' for prompt: '{prompt_text[:70]}...' (attempt {attempt + 1})...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                image_url = result['data'][0]['url']

                # Download the image
                image_response = requests.get(image_url, timeout=30)
                image_response.raise_for_status()

                logger.info("Image successfully generated and downloaded by Together AI.")
                return image_response.content
            else:
                logger.error(f"Together AI API returned unexpected response format: {result}")
                raise ValueError("Together AI API did not return a valid image URL.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed during Together AI image generation: {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to generate image from Together AI after {max_retries} attempts.")
                return None
        except Exception as e:
            logger.error(f"Unexpected error during Together AI image generation: {e}", exc_info=True)
            return None


def enhance_image_quality(img):
    """Apply advanced image enhancement techniques."""
    if img.mode != 'RGB':
        img = img.convert('RGB')

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)

    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))

    return img

def create_text_with_shadow(draw, position, text, font, text_color, shadow_color, shadow_offset):
    """Draw text with shadow for better visibility."""
    x, y = position
    shadow_x, shadow_y = shadow_offset

    draw.text((x + shadow_x, y + shadow_y), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=text_color)

def find_content_bbox_and_trim(img, tolerance=20, border_colors_to_trim=((0,0,0), (255,255,255))):
    """
    Attempts to find the bounding box of non-border content pixels and trims the image.
    Considers black and white as potential uniform border colors.
    Increased tolerance for slight variations in border color.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size
    pixels = img.load()

    def is_similar(pixel1, pixel2, tol):
        return all(abs(c1 - c2) <= tol for c1, c2 in zip(pixel1, pixel2))

    def is_border_pixel_group(pixel):
        return any(is_similar(pixel, bc, tolerance) for bc in border_colors_to_trim)

    top = 0
    for y in range(height):
        if not all(is_border_pixel_group(pixels[x, y]) for x in range(width)):
            top = y
            break

    bottom = height
    for y in range(height - 1, top, -1):
        if not all(is_border_pixel_group(pixels[x, y]) for x in range(width)):
            bottom = y + 1
            break

    left = 0
    for x in range(width):
        if not all(is_border_pixel_group(pixels[x, y]) for y in range(height)):
            left = x
            break

    right = width
    for x in range(width - 1, left, -1):
        if not all(is_border_pixel_group(pixels[x, y]) for y in range(height)):
            right = x + 1
            break

    if (left, top, right, bottom) != (0, 0, width, height):
        min_content_ratio = 0.75
        trimmed_width = right - left
        trimmed_height = bottom - top
        if trimmed_width > (width * min_content_ratio) and \
           trimmed_height > (height * min_content_ratio):
            logger.info(f"Automatically trimmed detected uniform borders from original image. BBox: ({left}, {top}, {right}, {bottom})")
            return img.crop((left, top, right, bottom))
        else:
            logger.debug(f"Trimming borders would remove too much content ({trimmed_width}/{width} or {trimmed_height}/{height}). Skipping trim.")

    logger.debug("No significant uniform color borders detected in original image for trimming.")
    return img


def transform_image(image_input_bytes, title_text, category_text, output_category_folder, safe_filename):
    """
    Processes, and adds branding/text to an image provided as bytes.
    Saves the image to disk and returns its file path and Base64 encoded string.
    Returns (relative_file_path, base64_data_uri) or (None, None) on failure.
    """
    if not image_input_bytes:
        logger.info("No image bytes provided for transformation. Skipping image processing.")
        return None, None

    output_full_path = None
    base64_data_uri = None

    try:
        logger.info(f"Processing image bytes for title: {title_text[:70]}...")

        # Load image directly from bytes (CHANGED: no more downloading from URL)
        img = Image.open(BytesIO(image_input_bytes))

        if img.mode in ('RGBA', 'LA', 'P'):
            alpha = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
            background = Image.new('RGB', img.size, (255, 255, 255))
            if alpha:
                background.paste(img, mask=alpha)
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img = find_content_bbox_and_trim(img)

        target_content_width = 1200
        target_content_height = 675
        target_aspect = target_content_width / target_content_height

        original_width, original_height = img.size
        original_aspect = original_width / original_height

        if original_aspect > target_aspect:
            resize_height = target_content_height
            resize_width = int(target_content_height * original_aspect)
        else:
            resize_width = target_content_width
            resize_height = int(target_content_width / original_aspect)

        img = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

        left_crop = (resize_width - target_content_width) // 2
        top_crop = (resize_height - target_content_height) // 2
        img = img.crop((left_crop, top_crop, left_crop + target_content_width, top_crop + target_content_height))

        img = enhance_image_quality(img)
        img = img.convert('RGBA')

        extended_area_height = int(target_content_height * 0.25)
        final_canvas_height = target_content_height + extended_area_height
        final_canvas_width = target_content_width

        new_combined_img = Image.new('RGBA', (final_canvas_width, final_canvas_height), (0, 0, 0, 255))
        new_combined_img.paste(img, (0, 0))

        strip_from_original_height = int(target_content_height * 0.05)
        if strip_from_original_height > 0:
            bottom_strip_for_extension = img.crop((0, target_content_height - strip_from_original_height, target_content_width, target_content_height))
            stretched_strip = bottom_strip_for_extension.resize((target_content_width, extended_area_height), Image.Resampling.BICUBIC)
            new_combined_img.paste(stretched_strip, (0, target_content_height))
            logger.info("Extended bottom of image with stretched content for seamless look.")

        gradient_overlay_image = Image.new('RGBA', new_combined_img.size, (0, 0, 0, 0))
        draw_gradient = ImageDraw.Draw(gradient_overlay_image)
        gradient_top_y_on_canvas = target_content_height
        for y_relative_to_extended_area in range(extended_area_height):
            alpha = int(255 * (y_relative_to_extended_area / extended_area_height) * 0.95)
            absolute_y_on_canvas = gradient_top_y_on_canvas + y_relative_to_extended_area
            draw_gradient.line([(0, absolute_y_on_canvas), (final_canvas_width, absolute_y_on_canvas)], fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(new_combined_img, gradient_overlay_image)
        draw = ImageDraw.Draw(img)

        if BRANDING_LOGO_PATH and os.path.exists(BRANDING_LOGO_PATH):
            try:
                logo = Image.open(BRANDING_LOGO_PATH).convert("RGBA")
                logo_height = int(target_content_height * 0.08)
                logo_width = int(logo.width * (logo_height / logo.height))
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

                padding = int(target_content_width * 0.02)
                logo_x = target_content_width - logo_width - padding
                logo_y = padding

                logo_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                logo_overlay.paste(logo, (logo_x, logo_y), logo)
                img = Image.alpha_composite(img, logo_overlay)
                logger.info("Branding logo applied successfully.")
            except Exception as e:
                logger.error(f"Error applying branding logo: {e}")

        # Draw only the brand name in Hindi at the bottom right
        brand_text = "भारत समाचार सागर"
        brand_font_size = max(int(target_content_height * 0.045), 28)
        if DEFAULT_FONT_PATH:
            try:
                brand_font = ImageFont.truetype(DEFAULT_FONT_PATH, brand_font_size)
            except (IOError, OSError):
                brand_font = ImageFont.load_default()
        else:
            brand_font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        padding = int(target_content_width * 0.03)
        try:
            bbox = brand_font.getbbox(brand_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = brand_font.getsize(brand_text)
        x = target_content_width - text_width - padding
        y = final_canvas_height - text_height - padding
        create_text_with_shadow(draw, (x, y), brand_text, brand_font, (255,255,255,255), (0,0,0,180), (2,2))

        output_filename = f"{safe_filename}_{int(time.time())}.jpg"
        output_full_path = os.path.join(IMAGE_OUTPUT_FOLDER, output_category_folder, output_filename)

        final_img_for_save = img.convert('RGB')

        buffer = BytesIO()
        final_img_for_save.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes_output = buffer.getvalue()

        base64_encoded_image = base64.b64encode(image_bytes_output).decode('utf-8')
        base64_data_uri = f"data:image/jpeg;base64,{base64_encoded_image}"

        with open(output_full_path, 'wb') as f:
            f.write(image_bytes_output)

        logger.info(f"Transformed image saved to disk: {output_full_path}")
        logger.info(f"Transformed image Base64 encoded.")
        return output_full_path, base64_data_uri

    except IOError as e:
        logger.error(f"Error processing image bytes: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during image transformation: {e}", exc_info=True)
        return None, None

def _gemini_generate_content_with_retry(model, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_INITIAL_RETRY_DELAY_SECONDS):
    """
    Helper function to call Gemini's generate_content with retry logic for transient errors.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if not response.text or response.text.strip() == "":
                logger.warning(f"Attempt {attempt + 1}: Gemini returned empty response. Retrying...")
                raise ValueError("Empty response from Gemini model.")
            return response
        except (
            exceptions.InternalServerError,
            exceptions.ResourceExhausted,
            exceptions.DeadlineExceeded,
            requests.exceptions.RequestException,
            ValueError
        ) as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Gemini API call failed after {max_retries} attempts.")
                raise


def perform_research_agent(target_topic, competitors):
    """
    Acts as the 'Research Agent'. Uses Gemini to find SEO keywords and outline suggestions.
    Outputs a JSON string.
    """
    if not RESEARCH_MODEL:
        logger.error("Research model not initialized. Skipping research agent.")
        return None

    prompt = (
        f"आप एक विशेषज्ञ SEO कीवर्ड रिसर्च एजेंट हैं जो हिंदी में कंटेंट स्ट्रैटेजी में माहिर हैं। "
        f"आपका काम इस विषय के लिए व्यापक SEO कीवर्ड रिसर्च और आउटलाइन तैयार करना है: '{target_topic}'.\n\n"
        f"प्रतिस्पर्धियों (जैसे {', '.join(competitors[:5])}) के कंटेंट का विश्लेषण करके प्रासंगिक SEO कीवर्ड, कंटेंट गैप और संरचनात्मक अंतर्दृष्टि की पहचान करें।\n\n"
        f"**महत्वपूर्ण:** विषय, मूल स्रोत जानकारी और कीवर्ड रिसर्च के आधार पर, एक **अद्वितीय, आकर्षक और SEO-अनुकूलित ब्लॉग पोस्ट शीर्षक (H1)** तैयार करें जो पाठकों को आकर्षित करे और अच्छी रैंकिंग प्राप्त करे। यह शीर्षक मूल स्रोत शीर्षकों से अलग होना चाहिए और एक समेकित, गहन परिप्रेक्ष्य को दर्शाता हो।\n\n"
        "## प्रक्रिया प्रवाह:\n"
        "1.  **प्रारंभिक कीवर्ड खोज:** प्राथमिक (उच्च खोज मात्रा, उच्च प्रासंगिकता), द्वितीयक (लंबी पूंछ, विशिष्ट), और विविध कीवर्ड क्लस्टर की पहचान करें। विभिन्न उपयोगकर्ता इरादों (सूचनात्मक, वाणिज्यिक, नेविगेशनल) के बारे में सोचें।\n"
        "2.  **प्रतिस्पर्धी विश्लेषण:** विषय के संबंध में प्रतिस्पर्धी रणनीतियों और कंटेंट गैप के 2-3 प्रमुख अंतर्दृष्टि प्रदान करें।\n"
        "3.  **कीवर्ड मूल्यांकन:** पहचाने गए कीवर्ड के लिए खोज मात्रा और प्रतिस्पर्धा स्तर का आकलन करें। SEO अनुकूलन के लिए उच्च-मूल्य, प्रासंगिक कीवर्ड को प्राथमिकता दें। महत्वपूर्ण संबंधित इकाइयों और अवधारणाओं की पहचान करें।\n"
        "4.  **आउटलाइन निर्माण:** उच्च-मूल्य कीवर्ड को रणनीतिक रूप से शामिल करते हुए एक विस्तृत, पदानुक्रमित ब्लॉग पोस्ट आउटलाइन (मार्कडाउन शीर्षकों `##`, `###` का उपयोग करके) तैयार करें। सुनिश्चित करें कि आउटलाइन तार्किक रूप से प्रवाहित हो और विषय के व्यापक पहलुओं को कवर करे।\n\n"
        "## आउटपुट विनिर्देश:\n"
        "निम्नलिखित संरचना के साथ एक JSON ऑब्जेक्ट (स्ट्रिंग के रूप में) तैयार करें। सुनिश्चित करें कि `blog_outline` एक वैध मार्कडाउन स्ट्रिंग है।\n"
        "```json\n"
        "{{\n"
        "  \"suggested_blog_title\": \"आपका अद्वितीय और आकर्षक ब्लॉग पोस्ट शीर्षक यहां\",\n"
        "  \"primary_keywords\": [\"कीवर्ड1\", \"कीवर्ड2\", \"कीवर्ड3\"],\n"
        "  \"secondary_keywords\": {{\"उप_विषय1\": [\"कीवर्डA\", \"कीवर्डB\"], \"उप_विषय2\": [\"कीवर्डC\", \"कीवर्डD\"]}},\n"
        "  \"competitor_insights\": \"प्रतिस्पर्धी रणनीतियों और कंटेंट गैप का सारांश।\",\n"
        "  \"blog_outline\": \"## परिचय\\n\\n### हुक\\n\\n## मुख्य खंड 1: [खंड शीर्षक]\\n\\n### उप-खंड 1.1\\n\\n## निष्कर्ष\\n\"\n"
        "}}\n"
        "```\n"
        "**बाधाएं:** वाणिज्यिक रूप से प्रासंगिक शब्दों पर ध्यान केंद्रित करें। ब्रांडेड प्रतिस्पर्धी शब्दों को बाहर रखें। पूरा आउटपुट एक वैध JSON स्ट्रिंग होना चाहिए। `blog_outline` में कम से कम 8 अलग-अलग मार्कडाउन शीर्षक (H2 या H3) होने चाहिए और यह उपयोगकर्ता जुड़ाव और SEO के लिए संरचित होना चाहिए। `suggested_blog_title` संक्षिप्त, प्रभावशाली और आदर्श रूप से 70 अक्षरों से कम होना चाहिए। JSON ब्लॉक के बाहर कोई परिचयात्मक या समापन टिप्पणी शामिल न करें।"
    )
    try:
        logger.info(f"Generating research for: '{target_topic[:70]}...'")
        response = _gemini_generate_content_with_retry(RESEARCH_MODEL, prompt)

        json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            research_data = json.loads(json_str)
            logger.info("Research generation successful.")
            return research_data
        else:
            logger.warning(f"Could not find valid JSON in markdown block for '{target_topic}'. Attempting to parse raw response.")
            try:
                research_data = json.loads(response.text.strip())
                logger.info("Research generation successful (parsed raw response).")
                return research_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse research response as JSON for '{target_topic}'. Raw response:\n{response.text[:500]}...")
                return None

    except Exception as e:
        logger.error(f"Research Agent generation failed for '{target_topic}': {e}", exc_info=True)
        return None

def generate_content_agent(consolidated_article_data, research_output, transformed_image_filepath):
    """
    Acts as the 'Content Generator Agent'. Uses Gemini to write the blog post
    based on aggregated source data and research output.
    """
    if not CONTENT_MODEL:
        logger.error("Content model not initialized. Skipping content generation.")
        return None

    image_path_for_prompt = transformed_image_filepath if transformed_image_filepath else "None"

    primary_keywords_str = ', '.join(research_output.get('primary_keywords', []))
    secondary_keywords_str = ', '.join([kw for sub_list in research_output.get('secondary_keywords', {}).values() for kw in sub_list])

    new_blog_title = research_output.get('suggested_blog_title', consolidated_article_data.get('consolidated_topic', 'डिफ़ॉल्ट समेकित ब्लॉग शीर्षक'))

    logger.info(f"Using primary keywords for content generation: {primary_keywords_str}")
    logger.info(f"Using secondary keywords for content generation: {secondary_keywords_str}")

    category = consolidated_article_data.get('category', 'general')
    hindi_category = HINDI_CATEGORIES.get(category, category)
    primary_keywords = research_output.get('primary_keywords', [])
    secondary_keywords = []
    if research_output.get('secondary_keywords'):
        for sub_list in research_output.get('secondary_keywords', {}).values():
            secondary_keywords.extend(sub_list)
    
    logger.info("Preparing metadata for blog post:")
    logger.info(f"Title: {new_blog_title}")
    logger.info(f"Category: {hindi_category}")
    logger.info(f"Primary Keywords: {primary_keywords}")
    logger.info(f"Secondary Keywords: {secondary_keywords}")

    combined_content_for_prompt = consolidated_article_data.get('combined_content', '')
    if len(combined_content_for_prompt) > 4000:
        combined_content_for_prompt = combined_content_for_prompt[:4000] + "\n\n[...कंटेंट संक्षिप्ति के लिए छोटा किया गया...]"
        logger.info(f"Truncated combined_content for prompt: {len(consolidated_article_data['combined_content'])} -> {len(combined_content_for_prompt)} characters.")

    consolidated_article_data_for_prompt = consolidated_article_data.copy()
    consolidated_article_data_for_prompt['combined_content'] = combined_content_for_prompt

    raw_description_for_prompt = consolidated_article_data.get(
        'combined_description',
        'नवीनतम समाचारों और रुझानों पर एक व्यापक और अंतर्दृष्टिपूर्ण दृष्टिकोण।'
    )
    blog_description_for_prompt = raw_description_for_prompt.replace('"', '').replace('\n', ' ').replace('\r', ' ').strip()[:155]

    prompt = (
        f"आप एक विशेष ब्लॉग लेखन एजेंट हैं जो SEO रिसर्च और समेकित लेख डेटा को व्यापक, प्रकाशन-तैयार, SEO-अनुकूलित ब्लॉग पोस्ट में बदलता है। "
        f"आप कई स्रोतों से जानकारी को संश्लेषित करके गहन, प्राधिकृत कंटेंट बनाने में माहिर हैं, जबकि पाठक जुड़ाव और SEO सर्वोत्तम प्रथाओं को बनाए रखते हैं।\n\n"
        f"## इनपुट आवश्यकताएं:\n"
        f"1.  `aggregated_source_data`: {json.dumps(consolidated_article_data_for_prompt, indent=2)}\n"
        f"2.  `research_output`: {json.dumps(research_output, indent=2)}\n"
        f"3.  `transformed_image_path_info`: '{image_path_for_prompt}' (यह मुख्य फीचर्ड इमेज का फाइल पाथ है। कंटेंट बॉडी में इस इमेज को फिर से एम्बेड न करें। इसे HTML टेम्पलेट में अलग से हैंडल किया जाएगा।)\n\n"
        f"## कंटेंट विनिर्देश:\n"
        f"-   **शब्द गणना:** 2500-3000 शब्दों का लक्ष्य रखें। `aggregated_source_data['combined_content']` पर विचारपूर्वक विस्तार करें, गहराई, विशिष्ट (यहां तक कि काल्पनिक) विवरण, और अपने प्रशिक्षण डेटा से संबंधित जानकारी जोड़ें। इनपुट से कंटेंट को सीधे कॉपी-पेस्ट न करें। पुनर्लेखन और एकीकरण करें।\n"
        f"-   **शीर्षक संरचना:** प्रदान किए गए आउटलाइन (`research_output['blog_outline']`) का उपयोग करें। कम से कम 25 शीर्षक (`##` और `###` केवल, मुख्य H1 शीर्षक को छोड़कर) सुनिश्चित करें।\n"
        f"-   **पैराग्राफ लंबाई:** प्रत्येक पैराग्राफ में व्यापक कवरेज के लिए कम से कम 5 वाक्य होने चाहिए, जब तक कि यह छोटा इंट्रो/आउट्रो या बुलेट पॉइंट स्पष्टीकरण न हो।\n"
        f"-   **लेखन शैली:** पेशेवर लेकिन बातचीत जैसी, आकर्षक और मानव जैसी। जहां सरल शब्द पर्याप्त हों वहां जार्गन से बचें। यह न बताएं कि आप एक AI हैं या कंटेंट जनरेट किया है। एक स्पष्ट, प्राधिकृत और विश्वसनीय स्वर सुनिश्चित करें जो कंटेंट को अत्यधिक विश्वसनीय के रूप में स्थापित करे।\n"
        f"-   **लक्षित दर्शक:** निर्दिष्ट श्रेणी में रुचि रखने वाला व्यापक दर्शक।\n"
        f"-   **कीवर्ड एकीकरण:** `primary_keywords` ({primary_keywords_str}) और `secondary_keywords` ({secondary_keywords_str}) को कीवर्ड स्टफिंग के बिना पूरे टेक्स्ट में स्वाभाविक रूप से बुनें। उन्हें शीर्षकों, उपशीर्षकों और बॉडी पैराग्राफ में एकीकृत करें।\n"
        f"-   **कंटेंट विस्तार:** `aggregated_source_data['combined_content']` पर अपने व्यापक ज्ञान आधार से विशिष्ट विवरण, स्पष्टीकरण और संदर्भ जोड़कर काफी विस्तार करें। एक अद्वितीय और व्यापक लेख बनाने के लिए *सभी प्रदान किए गए स्रोतों* से जानकारी को संश्लेषित करने पर जोर दें।\n"
        f"-   **डेटा और उदाहरण:** प्रासंगिक डेटा, आंकड़े और वास्तविक दुनिया के उदाहरण शामिल करें (यहां तक कि मूल लेख में नहीं, तो भी प्रशंसनीय लोगों का निर्माण करें)। विश्वसनीयता बढ़ाने के लिए जहां उपयुक्त हो वहां यथार्थवादी संख्याओं, तिथियों, नामों और संगठनों का आविष्कार करें। सुनिश्चित करें कि ये विवरण समेकित कंटेंट से प्राप्त मुख्य बिंदुओं का *समर्थन* करते हैं।\n"
        f"-   **लिंकिंग:** जहां उपयुक्त हो वहां प्रासंगिक बाहरी लिंक जनरेट करें। **महत्वपूर्ण रूप से, सुनिश्चित करें कि ये विषय से संबंधित प्रतिष्ठित डोमेन से वास्तविक, प्रशंसनीय URL हैं।** इन URL को यथार्थवादी रूप से आविष्कार करें और उन्हें आसपास के वाक्यों में स्वाभाविक रूप से एम्बेड करें। लिंक या कच्चे URL से पहले `@` प्रतीक या किसी अन्य उपसर्ग का उपयोग न करें। `example.com` या इसी तरह के प्लेसहोल्डर डोमेन शामिल न करें।**\n"
        f"-   **इमेज समावेशन:** कंटेंट बॉडी के भीतर फीचर्ड इमेज के लिए किसी भी मार्कडाउन `![alt text](image_path)` सिंटैक्स को शामिल न करें। फीचर्ड इमेज को अलग से हैंडल किया जाता है।\n"
        f"## आउटपुट संरचना:\n"
        f"मार्कडाउन प्रारूप में पूरा ब्लॉग पोस्ट जनरेट करें। इसे मेटाडेटा ब्लॉक के बाद ब्लॉग कंटेंट के साथ शुरू होना चाहिए।\n\n"
        f"**मेटाडेटा ब्लॉक (सटीक की-वैल्यू पेयर्स, कोई --- डिलीमिटर्स नहीं, न्यूलाइन से अलग):**\n"
        f"title: {new_blog_title}\n"
        f"description: {blog_description_for_prompt}\n"
        f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"categories: [{hindi_category}, {', '.join(primary_keywords[:2])}]\n"
        f"tags: [{', '.join(primary_keywords + secondary_keywords[:5])}]\n"
        f"featuredImage: {transformed_image_filepath if transformed_image_filepath else 'None'}\n\n"
        f"**ब्लॉग कंटेंट (मेटाडेटा ब्लॉक के बाद):**\n"
        f"1.  **मुख्य शीर्षक (H1):** प्रदान किए गए `suggested_blog_title` के आधार पर H1 शीर्षक से शुरू करें। उदाहरण: `# {new_blog_title}`.\n"
        f"2.  **परिचय (2-3 पैराग्राफ):** पाठक को हुक करें। समस्या या विषय को स्पष्ट रूप से बताएं और अपने ब्लॉग का मूल्य प्रस्ताव।\n"
        f"3.  **मुख्य खंड:** `research_output` से `blog_outline` का पालन करें। प्रत्येक खंड (`##`) और उप-खंड (`###`) का विस्तार करें। सुनिश्चित करें कि प्रत्येक खंड पर्याप्त जानकारी प्रदान करता है।\n"
        f"4.  **FAQ खंड:** विषय से संबंधित 5-7 अक्सर पूछे जाने वाले प्रश्नों को विस्तृत, व्यापक उत्तरों के साथ शामिल करें और कीवर्ड को शामिल करें।\n"
        f"5.  **निष्कर्ष:** प्रमुख टेकअवे को संक्षेप में प्रस्तुत करें, एक आगे देखने वाला बयान प्रदान करें, और एक स्पष्ट कॉल-टू-एक्शन।\n"
        f"ब्लॉग कंटेंट के बाहर कोई परिचयात्मक या समापन टिप्पणी शामिल न करें (जैसे 'यहां आपका ब्लॉग पोस्ट है')। **आउटपुट मार्कडाउन के भीतर कोई ब्रैकेटेड निर्देश (जैसे `[इसे उल्लेख करें]`), प्लेसहोल्डर (जैसे `example.com`), या मेरे लिए इरादित कोई टिप्पणी शामिल न करें। पूरा आउटपुट पॉलिश किया हुआ, अंतिम कंटेंट होना चाहिए, प्रकाशन के लिए तैयार।**"
    )

    try:
        logger.info(f"Generating full blog content for: '{new_blog_title[:70]}...'")
        response = _gemini_generate_content_with_retry(CONTENT_MODEL, prompt)

        content = response.text.strip()

        logger.info(f"--- Raw AI-generated Markdown Content (first 500 chars): ---\n{content[:500]}\n--- End Raw AI Markdown ---")
        logger.info(f"Full raw AI-generated Markdown content length: {len(content)} characters.")

        if not re.search(r"title:\s*.*\n.*?tags:\s*\[.*\]", content, re.DOTALL):
            logger.warning("Generated content appears to be missing the required metadata block!")
            content = (
                f"title: {new_blog_title}\n"
                f"description: {blog_description_for_prompt}\n"
                f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"categories: [{hindi_category}, {', '.join(primary_keywords[:2])}]\n"
                f"tags: [{', '.join(primary_keywords + secondary_keywords[:5])}]\n"
                f"featuredImage: {transformed_image_filepath if transformed_image_filepath else 'None'}\n\n"
                f"{content}"
            )
            logger.info("Added missing metadata block to content")

        content = clean_ai_artifacts(content)

        logger.info(f"--- Cleaned AI-generated Markdown Content (first 500 chars): ---\n{content[:500]}\n--- End Cleaned AI Markdown ---")

        logger.info("Content generation successful.")
        return content

    except Exception as e:
        logger.error(f"Content Agent generation failed for '{new_blog_title}': {e}", exc_info=True)
        return None

def clean_ai_artifacts(content):
    """Enhanced cleaning of AI-generated artifacts and placeholders."""
    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'\s*@\S+', '', content)
    placeholder_domains = [
        'example.com', 'example.org', 'placeholder.com', 'yoursite.com',
        'website.com', 'domain.com', 'site.com', 'yourblogname.com', 'ai-generated.com'
    ]
    for domain in placeholder_domains:
        content = re.sub(rf'\[[^\]]*\]\(https?://(?:www\.)?{re.escape(domain)}[^\)]*\)', '', content, flags=re.IGNORECASE)
        content = re.sub(rf'https?://(?:www\.)?{re.escape(domain)}\S*', '', content, flags=re.IGNORECASE)
    ai_patterns = [
        r'(?i)note:.*?(?=\n|$)',
        r'(?i)important:.*?(?=\n|$)',
        r'(?i)remember to.*?(?=\n|$)',
        r'(?i)please.*?(?=\n|$)',
        r'(?i)you should.*?(?=\n|$)',
        r'<!--.*?-->',
        r'/\*.*?\*/',
    ]
    for pattern in ai_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = '\n'.join([line.strip() for line in content.split('\n')])
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return content.strip()

def parse_markdown_metadata(markdown_content):
    """
    Parses metadata from the top of a markdown string.
    """
    metadata = {}
    lines = markdown_content.split('\n')
    content_start_index = 0

    logger.debug("Starting metadata parsing...")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            content_start_index = i + 1
            logger.debug(f"Blank line found, metadata ends at line {i}. Content starts at {content_start_index}.")
            break
        if ':' in stripped_line:
            key, value = stripped_line.split(':', 1)
            metadata[key.strip()] = value.strip()
            logger.debug(f"Parsed metadata line: {key.strip()}: {value.strip()}")
        else:
            content_start_index = i
            logger.warning(f"Metadata block ended unexpectedly at line {i} with: '{stripped_line}'")
            break
    else:
        content_start_index = len(lines)
        logger.debug("No blank line found, assuming all content is metadata or empty after checking.")

    blog_content_only = '\n'.join(lines[content_start_index:]).strip()

    if blog_content_only.startswith('# '):
        h1_line_end = blog_content_only.find('\n')
        if h1_line_end != -1:
            h1_title = blog_content_only[2:h1_line_end].strip()
            if 'title' not in metadata:
                metadata['title'] = h1_title
            blog_content_only = blog_content_only[h1_line_end:].strip()
            logger.debug(f"Extracted H1 title: '{h1_title}'. Remaining content starts after H1.")
        else:
            h1_title = blog_content_only[2:].strip()
            if 'title' not in metadata:
                metadata['title'] = h1_title
            blog_content_only = ""
            logger.debug(f"Extracted H1 title (only line): '{h1_title}'. Content became empty.")

    logger.info(f"Final parsed metadata: {metadata}")
    logger.info(f"Blog content starts with: {blog_content_only[:100]}...")

    return metadata, blog_content_only

def markdown_to_html(markdown_text, main_featured_image_filepath=None, main_featured_image_b64_data_uri=None):
    """
    Converts a subset of Markdown to HTML.
    """
    html_text = markdown_text
    html_text = clean_ai_artifacts(html_text)

    html_text = re.sub(r'###\s*(.*)', r'<h3>\1</h3>', html_text)
    html_text = re.sub(r'##\s*(.*)', r'<h2>\1</h2>', html_text)
    html_text = re.sub(r'#\s*(.*)', r'<h1>\1</h1>', html_text)

    html_text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', html_text)
    html_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'_(.*?)_', r'<em>\1</em>', html_text)
    html_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_text)

    html_text = re.sub(r'^\s*([-*]|\d+\.)\s+(.*)$', r'<li>\2</li>', html_text, flags=re.MULTILINE)

    def wrap_lists(match):
        list_items_html = match.group(0)
        if re.search(r'<li>\s*\d+\.', list_items_html):
            return f'<ol>{list_items_html}</ol>'
        else:
            return f'<ul>{list_items_html}</ul>'

    html_text = re.sub(r'(<li>.*?</li>\s*)+', wrap_lists, html_text, flags=re.DOTALL)

    def image_replacer(match):
        alt_text = match.group(1)
        src_url = match.group(2)
        if main_featured_image_filepath and os.path.basename(src_url) == os.path.basename(main_featured_image_filepath):
            logger.info(f"Replacing markdown image link '{src_url}' with Base64 data URI for in-content display.")
            return f'<img src="{main_featured_image_b64_data_uri}" alt="{alt_text}" class="in-content-image">'
        else:
            escaped_alt_text = alt_text.replace('"', '"')
            return f'<img src="{src_url}" alt="{escaped_alt_text}" class="in-content-image">'

    html_text = re.sub(r'!\[(.*?)\]\((.*?)\)', image_replacer, html_text)

    html_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', html_text)

    lines = html_text.split('\n')
    parsed_lines = []
    current_paragraph_lines = []

    block_tags_re = re.compile(r'^\s*<(h\d|ul|ol|li|img|a|div|p|blockquote|pre|table|script|style|br)', re.IGNORECASE)

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if current_paragraph_lines:
                para_content = ' '.join(current_paragraph_lines).strip()
                if para_content:
                    parsed_lines.append(f"<p>{para_content}</p>")
                current_paragraph_lines = []
            parsed_lines.append('')
        elif block_tags_re.match(stripped_line):
            if current_paragraph_lines:
                para_content = ' '.join(current_paragraph_lines).strip()
                if para_content:
                    parsed_lines.append(f"<p>{para_content}</p>")
                current_paragraph_lines = []
            parsed_lines.append(line)
        else:
            current_paragraph_lines.append(line)

    if current_paragraph_lines:
        para_content = ' '.join(current_paragraph_lines).strip()
        if para_content:
            parsed_lines.append(f"<p>{para_content}</p>")

    final_html_content = '\n'.join(parsed_lines)

    final_html_content = re.sub(r'<p>\s*</p>', '', final_html_content)
    final_html_content = re.sub(r'<p><br\s*/?></p>', '', final_html_content)
    final_html_content = re.sub(r'<h1>(.*?)</h1>', r'<h2>\1</h2>', final_html_content) # Ensure no H1 from content

    return final_html_content

def generate_enhanced_html_template(title, description, keywords, image_url_for_seo,
                                  image_src_for_html_body, html_blog_content,
                                  category, article_url_for_disclaimer, published_date):
    """Generate enhanced HTML template with better styling and comprehensive SEO elements."""

    # Escape HTML special characters
    escaped_title_html = title.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&apos;')
    escaped_description_html = description.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&apos;')

    json_safe_title = json.dumps(title)[1:-1]
    json_safe_description = json.dumps(description)[1:-1]

    structured_data = f"""
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "NewsArticle",
      "headline": "{json_safe_title}",
      "image": {json.dumps([image_url_for_seo]) if image_url_for_seo else "[]"},
      "datePublished": "{published_date}T00:00:00Z",
      "dateModified": "{published_date}T00:00:00Z",
      "articleSection": "{category.capitalize()}",
      "author": {{
        "@type": "Organization",
        "name": "AI Content Creator"
      }},
      "publisher": {{
        "@type": "Organization",
        "name": "Your Publication Name",
        "logo": {{
          "@type": "ImageObject",
          "url": "{image_url_for_seo}"
        }}
      }},
      "mainEntityOfPage": {{
        "@type": "WebPage",
        "@id": "{article_url_for_disclaimer}"
      }},
      "description": "{json_safe_description}"
    }}
    </script>
    """

    html_styles = """
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --text-color: #333;
            --light-bg: #f5f7fa;
            --card-bg: #ffffff;
            --border-color: #e0e0e0;
            --shadow-light: 0 4px 15px rgba(0,0,0,0.08);
            --shadow-hover: 0 6px 20px rgba(0,0,0,0.12);
        }

        body {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: var(--text-color);
            background: var(--light-bg);
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 850px;
            margin: 30px auto;
            padding: 25px;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: var(--shadow-light);
            transition: all 0.3s ease-in-out;
        }
        .container:hover {
            box-shadow: var(--shadow-hover);
        }

        .article-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .category-tag {
            display: inline-block;
            background: var(--primary-color);
            color: white;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.8px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 2.2em;
            color: var(--secondary-color);
            margin-bottom: 15px;
            line-height: 1.3;
        }
        h2 {
            font-size: 1.7em;
            color: var(--secondary-color);
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px dashed var(--border-color);
        }
        h3 {
            font-size: 1.3em;
            color: var(--secondary-color);
            margin-top: 25px;
            margin-bottom: 10px;
        }

        p {
            margin-bottom: 1.2em;
        }

        .featured-image {
            width: 100%;
            height: auto;
            max-height: 843.75px;
            object-fit: cover;
            border-radius: 8px;
            margin-top: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .in-content-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2em auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        a {
            color: var(--primary-color);
            text-decoration: none;
            transition: color 0.2s ease-in-out;
        }
        a:hover {
            color: #1a5e8c;
            text-decoration: underline;
        }

        ul, ol {
            margin-left: 25px;
            margin-bottom: 1.5em;
        }
        li {
            margin-bottom: 0.6em;
        }

        .source-link {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.95em;
            text-align: center;
            color: #666;
        }

        @media (max-width: 768px) {
            .container {
                margin: 15px;
                padding: 15px;
            }
            h1 { font-size: 1.8em; }
            h2 { font-size: 1.5em; }
            h3 { font-size: 1.2em; }
            .category-tag { font-size: 0.8em; padding: 6px 14px; }
        }
    </style>
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escaped_title_html}</title>
    <meta name="description" content="{escaped_description_html}">
    <meta name="keywords" content="{keywords}">
    <meta name="robots" content="index, follow">
    <meta name="author" content="AI Content Creator">

    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="article">
    <meta property="og:url" content="{article_url_for_disclaimer}">
    <meta property="og:title" content="{escaped_title_html}">
    <meta property="og:description" content="{escaped_description_html}">
    <meta property="og:image" content="{image_url_for_seo}">

    <!-- Twitter -->
    <meta property="twitter:card" content="summary_large_image">
    <meta property="twitter:url" content="{article_url_for_disclaimer}">
    <meta property="twitter:title" content="{escaped_title_html}">
    <meta property="twitter:description" content="{escaped_description_html}">
    <meta property="twitter:image" content="{image_url_for_seo}">

    {structured_data}
    {html_styles}
</head>
<body>
    <div class="container">
        <div class="article-header">
            <span class="category-tag">{category.upper()}</span>
            <h1>{title}</h1>
            {f'<img src="{image_src_for_html_body}" alt="{escaped_title_html}" class="featured-image">' if image_src_for_html_body else ''}
        </div>
        <div class="article-content">
            {html_blog_content}
        </div>
        <div class="source-link">
            <p><strong>Disclaimer:</strong> This article was generated by an AI content creation system, synthesizing information from multiple sources. It may contain fictional details and external links for illustrative purposes.</p>
            <p>A primary source contributing to this content can be found here: <a href="{article_url_for_disclaimer}" target="_blank" rel="noopener noreferrer">{article_url_for_disclaimer}</a></p>
        </div>
    </div>
</body>
</html>"""

def post_to_blogger(html_file_path, blog_id, blogger_user_credentials):
    """
    Posts a generated HTML blog to Blogger.
    """
    if not blogger_user_credentials or not blogger_user_credentials.valid:
        logger.error("Blogger User Credentials are not valid. Cannot post to Blogger.")
        return False

    try:
        blogger_service = build('blogger', 'v3', credentials=blogger_user_credentials)

        with open(html_file_path, 'r', encoding='utf-8') as f:
            full_html_content = f.read()

        logger.info("Starting metadata parsing from HTML file for Blogger post...")
        
        metadata_match = re.search(r"title:\s*(.*?)\n.*?tags:\s*\[(.*?)\]", full_html_content, re.DOTALL | re.IGNORECASE)
        post_title = "Generated Blog Post"
        post_labels = []

        if metadata_match:
            post_title = metadata_match.group(1).strip()
            tags_str = metadata_match.group(2).strip()
            post_labels = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            logger.info(f"Found title: {post_title}, Tags: {post_labels}")
            
            categories_match = re.search(r"categories:\s*\[(.*?)\]", full_html_content, re.DOTALL | re.IGNORECASE)
            if categories_match:
                categories_str = categories_match.group(1).strip()
                parsed_categories = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
                post_labels.extend(parsed_categories)
                logger.info(f"Categories found: {parsed_categories}")
                logger.info(f"Combined labels after adding categories: {post_labels}")
        else:
            logger.warning("Could not find metadata block in HTML content. Attempting to extract from H1...")
            h1_match = re.search(r'<h1>(.*?)</h1>', full_html_content, re.IGNORECASE | re.DOTALL)
            if h1_match:
                post_title = h1_match.group(1).strip()
                logger.info(f"Extracted title from H1: {post_title}")
            else:
                logger.warning("Could not find H1 tag either. Using default title.")

        post_labels = list(set([label.strip().lower() for label in post_labels if label.strip()]))
        if not post_labels:
            logger.warning("No labels found in metadata. Adding default labels based on title...")
            default_labels = [word.lower() for word in re.findall(r'\w+', post_title) if len(word) > 3]
            post_labels.extend(default_labels[:5])
        
        logger.info(f"Final cleaned labels to send to Blogger: {post_labels}")

        post_body = {
            'kind': 'blogger#post',
            'blog': {'id': blog_id},
            'title': post_title,
            'content': full_html_content,
            'labels': post_labels,
            'status': 'LIVE'
        }
        logger.info(f"Preparing Blogger post with:")
        logger.info(f"Title: {post_title}")
        logger.info(f"Labels: {post_labels}")
        logger.info(f"Content length: {len(full_html_content)} characters")

        logger.info(f"Attempting to insert blog post to Blogger: '{post_title}' with labels: {post_labels}...")
        request = blogger_service.posts().insert(blogId=blog_id, body=post_body)
        response = request.execute()

        logger.info(f"✅ Successfully posted '{post_title}' to Blogger! Post ID: {response.get('id')}")
        logger.info(f"View live at: {response.get('url')}")
        response_labels = response.get('labels', [])
        logger.info(f"Blogger API Response labels: {response_labels}")
        
        if not response_labels:
            logger.warning("No labels found in Blogger API response. Labels may not have been set correctly.")
        elif set(response_labels) != set(post_labels):
            logger.warning(f"Labels mismatch! Sent: {post_labels}, Received: {response_labels}")
        
        return True

    except HttpError as e:
        error_content = e.content.decode('utf-8')
        logger.error(f"Failed to post to Blogger due to API error: {e}")
        logger.error(f"Error details: {error_content}")
        if "rateLimitExceeded" in error_content:
            logger.error("Blogger API rate limit exceeded. Consider reducing posting frequency.")
        elif "User lacks permission" in str(e) or "insufficient permission" in str(e).lower():
            logger.error("Blogger: User lacks permission to post. Ensure the authenticated Google account has Author/Admin rights on the target blog.")
        return False
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Blogger posting: {e}", exc_info=True)
        return False


def save_blog_post(consolidated_topic_for_fallback, generated_markdown_content, category, transformed_image_filepath, transformed_image_b64, primary_source_url):
    """
    Saves the generated blog post in an HTML file with SEO elements.
    Accepts the *transformed_image_filepath* for SEO metadata and *transformed_image_b64* for inline HTML.
    `primary_source_url` is used for the disclaimer link.
    Returns the file path of the saved HTML blog.
    """
    metadata, blog_content_only_markdown = parse_markdown_metadata(generated_markdown_content)

    title = metadata.get('title', consolidated_topic_for_fallback)

    description_fallback = f"A comprehensive look at the latest news in {category} related to '{title}'."
    description = metadata.get('description', description_fallback).replace('&', '&amp;').replace('"', '&quot;').replace("'", '&apos;')[:155]

    keywords_from_meta = metadata.get('tags', '').replace(', ', ',').replace(' ', '_')
    if not keywords_from_meta:
        keywords = ','.join([category, 'news', 'latest', sanitize_filename(title)[:30]])
    else:
        keywords = keywords_from_meta.lower()

    image_src_for_html_body = transformed_image_b64 if transformed_image_b64 else ''

    # image_url_for_seo will be empty as we don't host images publicly from this script
    image_url_for_seo = ''
    if image_src_for_html_body:
        logger.warning("For optimal SEO (og:image, twitter:image, JSON-LD), 'image_url_for_seo' should be a publicly accessible URL. It is currently left blank or points to Base64 URI as the script doesn't upload images to a public host.")
        logger.warning("You may need to manually update the og:image, twitter:image, and JSON-LD image URL in Blogger after publishing if you want external image SEO.")

    published_date = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))

    html_blog_content = markdown_to_html(
        blog_content_only_markdown,
        main_featured_image_filepath=transformed_image_filepath,
        main_featured_image_b64_data_uri=transformed_image_b64
    )

    safe_title_for_file = sanitize_filename(title)

    folder = os.path.join(BLOG_OUTPUT_FOLDER, category)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{safe_title_for_file}.html")

    final_html_output = generate_enhanced_html_template(
        title, description, keywords, image_url_for_seo,
        image_src_for_html_body, html_blog_content,
        HINDI_CATEGORIES.get(category, category), # Use Hindi category name in template
        primary_source_url, published_date
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_html_output)
    logger.info(f"✅ Saved blog post: {file_path}")
    return file_path


# --- MODIFIED main function ---
def main():
    blogger_oauth_credentials = None

    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return

    if BLOGGER_BLOG_ID:
        logger.info("\n--- Authenticating with Blogger using OAuth 2.0 ---")
        blogger_oauth_credentials = get_blogger_oauth_credentials()
        if not blogger_oauth_credentials:
            logger.critical("CRITICAL: Failed to obtain Blogger OAuth credentials. Cannot post to Blogger. Exiting.")
            return
        logger.info("--- Blogger OAuth Authentication Successful ---\n")
    else:
        logger.warning("INFO: BLOGGER_BLOG_ID not configured. Blogger posting will be skipped.")

    global_competitors = [
        "forbes.com", "reuters.com", "bloomberg.com", "theverge.com",
        "techcrunch.com", "healthline.com", "webmd.com", "espn.com",
        "investopedia.com", "zdnet.com", "cnet.com", "medicalnewstoday.com",
        "bbc.com/news", "cnn.com", "nytimes.com"
    ]

    for category in CATEGORIES:
        logger.info(f"\n--- Starting processing for category: [{HINDI_CATEGORIES.get(category, category).upper()}] ---")

        raw_articles = fetch_gnews_articles(category, max_articles_to_fetch=NUM_SOURCE_ARTICLES_TO_AGGREGATE)

        if not raw_articles:
            logger.info(f"No raw articles fetched for {category}. Skipping category.")
            continue

        consolidated_data = aggregate_articles(raw_articles, category)

        if not consolidated_data:
            logger.error(f"Failed to aggregate articles for {category}. Skipping blog generation.")
            continue

        consolidated_topic = consolidated_data['consolidated_topic']
        consolidated_description = consolidated_data['combined_description']
        consolidated_content_for_ai = consolidated_data['combined_content']
        primary_source_url_for_disclaimer = consolidated_data['primary_source_url']

        effective_competitors = list(set(global_competitors + consolidated_data['competitors']))

        logger.info(f"\n  Starting workflow for consolidated topic: '{consolidated_topic[:70]}...'")

        transformed_image_filepath = None
        transformed_image_b64 = None

        # --- NEW IMAGE GENERATION FLOW ---
        # 1. Generate prompt for Together AI based on the consolidated topic
        image_gen_prompt = f"A visually striking and relevant image depicting the essence of the news topic: '{consolidated_topic}'. " \
                           f"Focus on professional, high-resolution, and compelling visual storytelling suitable for a news article. " \
                           f"Style: dynamic, modern digital art with subtle motion blur or futuristic elements."

        if TOGETHER_API_KEY:
            together_image_bytes = generate_image_from_together_ai(image_gen_prompt)

            if together_image_bytes:
                safe_image_filename = sanitize_filename(consolidated_topic)
                transformed_image_filepath, transformed_image_b64 = transform_image(
                    together_image_bytes,
                    consolidated_topic,
                    category,
                    category,
                    safe_image_filename
                )
            else:
                logger.warning(f"  Together AI image generation failed for topic '{consolidated_topic}'. Proceeding without a featured image.")
        else:
            logger.warning("  TOGETHER_API_KEY not set. Skipping Together AI image generation.")
        # --- END NEW IMAGE GENERATION FLOW ---

        try:
            consolidated_article_data_for_ai = {
                "consolidated_topic": consolidated_topic,
                "combined_description": consolidated_description,
                "combined_content": consolidated_content_for_ai,
                "category": category,
                "original_image_url_selected": "AI Generated" # Indicate source of image
            }

            if GEMINI_API_KEY and RESEARCH_MODEL and CONTENT_MODEL:
                research_output = perform_research_agent(consolidated_topic, effective_competitors)
                if not research_output:
                    logger.error(f"Failed to get research output for: '{consolidated_topic}'. Skipping content generation.")
                    continue
                logger.info(f"  Research successful. Suggested Title: '{research_output.get('suggested_blog_title', 'N/A')}'")
                logger.info(f"  Primary Keywords: {research_output.get('primary_keywords', [])}")

                generated_blog_markdown = generate_content_agent(
                    consolidated_article_data_for_ai,
                    research_output,
                    transformed_image_filepath
                )

                if not generated_blog_markdown:
                    logger.error(f"Failed to generate blog content for: '{consolidated_topic}'. Skipping save.")
                    continue
            else:
                logger.warning("GEMINI_API_KEY or RESEARCH_MODEL/CONTENT_MODEL is not initialized. Skipping AI content generation.")
                processed_description = consolidated_description.replace('"', '&quot;').replace('\n', ' ').strip()[:155].replace("'", '&apos;')
                generated_blog_markdown = (
                    f"title: {consolidated_topic}\n"
                    f"description: {processed_description}\n"
                    f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
                    f"categories: [{category}]\n"
                    f"tags: [{category}, news]\n"
                    f"featuredImage: {transformed_image_filepath or 'None'}\n\n"
                    f"# {consolidated_topic}\n\n"
                    f"<p>This is a placeholder blog post because AI generation was skipped due to missing API key.</p>\n"
                    f"<p>Original aggregated content details (first 500 chars): {consolidated_content_for_ai[:500]}...</p>"
                )
                research_output = {"primary_keywords": [], "secondary_keywords": {}, "competitor_insights": "", "blog_outline": "", "suggested_blog_title": consolidated_topic}

            saved_html_file_path = save_blog_post(
                consolidated_topic,
                generated_blog_markdown,
                category,
                transformed_image_filepath,
                transformed_image_b64,
                primary_source_url_for_disclaimer
            )

            if saved_html_file_path and blogger_oauth_credentials and BLOGGER_BLOG_ID:
                post_to_blogger(
                    saved_html_file_path,
                    BLOGGER_BLOG_ID,
                    blogger_oauth_credentials
                )
            else:
                logger.warning("Skipping Blogger post due to missing HTML file or Blogger credentials/ID.")

        except Exception as e:
            logger.critical(f"An unexpected error occurred during blog generation workflow for '{consolidated_topic}': {e}", exc_info=True)
        finally:
            time.sleep(30)

if __name__ == '__main__':
    main()
