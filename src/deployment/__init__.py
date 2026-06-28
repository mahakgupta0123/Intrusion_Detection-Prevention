"""
Deployment Module

This module provides deployment capabilities for IDS/IPS systems:
- FastAPI server for prevention
- Web dashboard
- Real-time API endpoints

Functions:
    create_app() - Create FastAPI application
    run_server() - Start FastAPI server

Usage:
    from src.deployment import create_app
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host='0.0.0.0', port=8000)
"""

from .prevention_api import create_app, run_server

__all__ = [
    'create_app',
    'run_server'
]
