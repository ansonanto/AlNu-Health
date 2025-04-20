#!/usr/bin/env python3
"""
Reset script for ChromaDB - completely removes and recreates the database directory
"""
import os
import sys
import shutil
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reset_db")

# Get the ChromaDB path from config
try:
    from config import CHROMA_PATH
    logger.info(f"Using ChromaDB path from config: {CHROMA_PATH}")
except ImportError:
    CHROMA_PATH = "./chroma_db"
    logger.info(f"Config not found, using default ChromaDB path: {CHROMA_PATH}")

def reset_chroma_db():
    """Completely reset the ChromaDB directory"""
    try:
        # Check if the directory exists
        if os.path.exists(CHROMA_PATH):
            # Create a backup
            backup_dir = f"{CHROMA_PATH}_backup_{int(time.time())}"
            logger.info(f"Creating backup at {backup_dir}")
            try:
                shutil.copytree(CHROMA_PATH, backup_dir)
                logger.info("Backup created successfully")
            except Exception as e:
                logger.warning(f"Could not create backup: {str(e)}")
            
            # Remove the directory
            logger.info(f"Removing ChromaDB directory: {CHROMA_PATH}")
            try:
                shutil.rmtree(CHROMA_PATH)
                logger.info("ChromaDB directory removed successfully")
            except Exception as e:
                logger.error(f"Error removing ChromaDB directory: {str(e)}")
                logger.info("Trying alternative removal method...")
                
                # Try alternative removal method
                for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e2:
                            logger.error(f"Could not remove file {name}: {str(e2)}")
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e2:
                            logger.error(f"Could not remove directory {name}: {str(e2)}")
                
                # Try to remove the main directory again
                try:
                    os.rmdir(CHROMA_PATH)
                    logger.info("ChromaDB directory removed with alternative method")
                except Exception as e2:
                    logger.error(f"Could not remove main directory: {str(e2)}")
                    return False
        
        # Create a fresh directory
        logger.info(f"Creating fresh ChromaDB directory at {CHROMA_PATH}")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Create an empty .gitkeep file to ensure the directory is tracked in git
        with open(os.path.join(CHROMA_PATH, ".gitkeep"), "w") as f:
            f.write("# This file ensures the directory is tracked in git\n")
        
        logger.info("ChromaDB reset completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error during reset: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting ChromaDB reset process")
    success = reset_chroma_db()
    if success:
        logger.info("ChromaDB reset completed successfully")
        sys.exit(0)
    else:
        logger.error("ChromaDB reset failed")
        sys.exit(1)
