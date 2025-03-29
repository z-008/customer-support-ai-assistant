"""
Main entry point for the Customer Support AI Assistant
"""

import uvicorn
from customer_support_ai.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
