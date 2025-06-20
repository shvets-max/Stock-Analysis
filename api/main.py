import io
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from scoring.utils.static import load_and_normalize_percentages_from_file

app = FastAPI()


@app.get(
    "/",
    response_class=RedirectResponse,
    status_code=302,
    include_in_schema=False,
)
def redirect_docs():
    """Redirect to SwaggerUI"""
    return "docs"


class Limits(BaseModel):
    column_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@app.post("/numeric_columns/")
async def get_numeric_columns(
    file: UploadFile = File(..., description="CSV file containing stock data")
):
    """
    Endpoint to identify numeric columns in the uploaded CSV file.

    Args:
        file (UploadFile): The uploaded CSV file.

    Returns:
        List[str]: A list of column names identified as numeric columns.
    """
    try:
        # Load and normalize the file
        data = await load_and_normalize_percentages_from_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    # Validate the DataFrame
    if data.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    if not data.columns.any():
        raise HTTPException(
            status_code=400, detail="The uploaded CSV file has no columns."
        )

    # Identify numeric columns
    numeric_columns = [
        col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])
    ]

    return {"numeric_columns": numeric_columns}


@app.post("/screen_stocks_with_limits/")
async def run_screen_stocks_with_limits(
    file: UploadFile = File(..., description="CSV file containing stock data"),
    limits: List[Limits] = None,
):
    """
    Endpoint to screen stocks based on specified limits.

    Args:
        file (UploadFile): The uploaded CSV file.
        limits (List[Limits]): A list of filtering criteria.

    Returns:
        StreamingResponse: A CSV file containing the filtered stock data.
    """
    try:
        # Load and normalize the file
        data = await load_and_normalize_percentages_from_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    if limits is None:
        limits = []

    # Apply filtering based on limits
    for limit in limits:
        if limit.column_name in data.columns:
            if limit.min_value is not None:
                data = data[
                    (data[limit.column_name] >= limit.min_value)
                    | (pd.isna(data[limit.column_name]))
                ]
            if limit.max_value is not None:
                data = data[
                    (data[limit.column_name] <= limit.max_value)
                    | (pd.isna(data[limit.column_name]))
                ]

    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Return CSV file as a StreamingResponse
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment;filename=filtered_stocks.csv"},
    )
