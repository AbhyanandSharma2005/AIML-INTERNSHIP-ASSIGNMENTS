from waitress import serve
from app import app
import logging

if __name__ == '__main__':
    # Set up basic logging for production
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.INFO)
    
    port = 8080
    print(f"Starting WeatherVision Production Server on http://0.0.0.0:{port}")
    print("Press Ctrl+C to quit.")
    
    # Serve the app using Waitress (Production WSGI Server for Windows)
    serve(app, host='0.0.0.0', port=port, threads=4)
