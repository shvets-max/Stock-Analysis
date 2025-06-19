# Stock Analysis Project

## Overview

This project performs stock screening, static scoring, dynamic scoring, 
and risk management for stock selection based on financial data.

## Pipeline

1.  **Screener**: Filters stocks based on fundamental criteria.
2.  **Static Scoring**: Assigns scores based on static financial metrics.
3.  **Dynamic Scoring**: Adjusts scores based on dynamic of some financial metrics.
4.  **Risk Management**: Find optimal portfolio weights and assess risk.

## Structure

-   `data/`: Contains raw and processed data.
-   `notebooks/`: Jupyter notebooks for exploration and analysis.
-   `scoring/`: Python package for stock screening and scoring.
    -   `constants.py`: Defines constants used in the project.
    -   `screen_stocks.py`: Main script for screening stocks.
    -   `static_scoring.py`: Script for static scoring.
    -   `dynamic_scoring.py`: Script for dynamic scoring.
    -   `utils/`: Utility functions.
-   `risk_management/`: Python package for risk assessment and management.
    -   `risk_manager.py`: Main script for risk management.
    -   `models.py`: Models for risk assessment.
-   `tests/`: Unit tests.

## Usage

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run `screen_stocks.py` to screen stocks.
3.  Run `static_scoring.py` to perform static scoring.
4.  Run `dynamic_scoring.py` to perform dynamic scoring.
5.  Run `risk_management/risk_manager.py` for risk assessment and management.