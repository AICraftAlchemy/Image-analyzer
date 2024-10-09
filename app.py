import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from datetime import datetime
import logging
from PIL import Image
import base64
from io import BytesIO

# Set up logging to display only in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the API key from the .env file
load_dotenv()
API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client with API key
client = Groq(api_key=API_KEY)

def log_activity(user_name: str, activity: str, status: bool):
    status_str = "success" if status else "failed"
    logger.info(f"User '{user_name}' - {activity}: {status_str}")

def encode_image_to_base64(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    buffered = BytesIO()
    try:
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding image for user: {str(e)}")
        return None

# Function to call the Groq API
def analyze_image_and_text(user_name: str, user_text: str, image):
    try:
        image_base64 = encode_image_to_base64(image)

        if image_base64 is None:
            raise Exception("Failed to encode image")

        log_activity(user_name, "API call", True)

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        response_content = completion.choices[0].message.content
        log_activity(user_name, "Image analysis", True)
        return response_content
    except Exception as e:
        log_activity(user_name, "Image analysis", False)
        return None

# Streamlit App Interface
def main():
    # Set page configuration to hide Streamlit's hamburger menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Session state to track if user has entered name
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None

    # Welcome screen
    if st.session_state.user_name is None:
        st.title("Welcome to Image Analyzer! 👋")
        user_name = st.text_input("Please enter your name:")
        if st.button("Enter"):
            if user_name:
                st.session_state.user_name = user_name
                log_activity(user_name, "Login", True)
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
                log_activity("anonymous", "Login", False)
        return

    # Main application interface
    st.title(f"Hello, {st.session_state.user_name}! 🎉")
    st.subheader("Image Analyzer")

    # User input for text and image upload
    user_text = st.text_input("Enter text")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    # Use default text if user doesn't provide any
    if not user_text:
        user_text = "Describe the image"

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            log_activity(st.session_state.user_name, "Image upload", True)

            if st.button("Analyze"):
                with st.spinner("Analyzing the image..."):
                    response = analyze_image_and_text(st.session_state.user_name, user_text, image)
                    if response:
                        st.success("Analysis complete!")
                        st.write(response)
                    else:
                        st.error("Sorry, there was an error analyzing the image. Please try again later or try another image.")
        except Exception as e:
            st.error("Error processing image. Please try another image.")
            log_activity(st.session_state.user_name, "Image processing", False)
    else:
        st.info("Please upload an image to analyze.")

    # Logout option
    if st.sidebar.button("Logout"):
        log_activity(st.session_state.user_name, "Logout", True)
        st.session_state.user_name = None
        st.rerun()
        
    st.markdown("""
    <div class='footer'>
    Developed by <a href='https://aicraftalchemy.github.io'>Ai Craft Alchemy</a><br>
    Connect with us: <a href='tel:+917661081043'>+91 7661081043</a>
    </div>
    """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(
        page_title="Image Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    main()