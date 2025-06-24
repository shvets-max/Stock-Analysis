import io
import json
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from scoring.general_info import GeneralInfo
from scoring.metrics import Forecasts, Fundamentals, Growth, Performance, Valuation
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


class Limit(BaseModel):
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
    limits: str = Form("[]", description="JSON string of filtering criteria"),
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

    try:
        parsed_limits = json.loads(limits)
        parsed_limits = (
            [parsed_limits] if isinstance(parsed_limits, dict) else parsed_limits
        )
        limits_list = [Limit(**limit) for limit in parsed_limits]
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for limits.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid limits format: {e}")

    # Apply filtering based on limits
    for limit in limits_list:
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


async def process_metric_data(
    metric_file: UploadFile,
    metric_class,
    general_data: pd.DataFrame,
    by: str = None,
):
    """
    Load, merge, process, and calculate scores for a given metric.

    Args:
        metric_file (UploadFile): The uploaded CSV file for the metric.
        metric_class: The class representing the metric (e.g., Growth, Valuation).
        general_data (pd.DataFrame): The preloaded general data.
        by (str, optional): The column to group by for normalization (e.g., "Sector").
        Defaults to None.

    Returns:
        pd.Series: The calculated scores for the metric, or None if no file is provided.
    """
    if not metric_file:
        return None

    metric = metric_class()
    metric_data = await load_and_normalize_percentages_from_file(metric_file)
    metric_data = metric_data.merge(general_data, on="Symbol", how="left")
    metric.data = metric_data
    if by:
        metric.normalize_metrics(by=by)
    else:
        metric.normalize_metrics()
    metric.initialize_weights()
    metric.calculate_scores()
    metric_score = (metric.normalized_data * metric.weights).sum(
        axis=1
    ) / metric.weights.sum(axis=1)
    return metric_score


@app.post("/static_scoring/", response_class=StreamingResponse)
async def static_scoring(
    general_file: UploadFile = File(..., description="CSV file for general data"),
    growth_file: UploadFile = File(None, description="CSV file for growth data"),
    valuation_file: UploadFile = File(None, description="CSV file for valuation data"),
    forecasts_file: UploadFile = File(None, description="CSV file for forecasts data"),
    performance_file: UploadFile = File(
        None, description="CSV file for performance data"
    ),
    fundamentals_file: UploadFile = File(
        None, description="CSV file for fundamentals data"
    ),
):
    """
    Run static scoring pipeline with data loaded from CSV files,
    merging with general data, and return CSV of scores.
    """
    # Load general data
    general_info = GeneralInfo()
    general_data = await load_and_normalize_percentages_from_file(general_file)
    general_info.data = general_data

    # Process each metric
    growth_score = await process_metric_data(growth_file, Growth, general_info.data)
    valuation_score = await process_metric_data(
        valuation_file, Valuation, general_info.data, by="Sector"
    )
    forecasts_score = await process_metric_data(
        forecasts_file, Forecasts, general_info.data
    )
    performance_score = await process_metric_data(
        performance_file, Performance, general_info.data
    )
    fundamentals_score = await process_metric_data(
        fundamentals_file, Fundamentals, general_info.data, by="Sector"
    )

    # Create a dictionary to hold the scores
    scores = {}
    if growth_score is not None:
        scores["growth_score"] = growth_score
    if valuation_score is not None:
        scores["valuation_score"] = valuation_score
    if forecasts_score is not None:
        scores["forecasts_score"] = forecasts_score
    if performance_score is not None:
        scores["performance_score"] = performance_score
    if fundamentals_score is not None:
        scores["fundamentals_score"] = fundamentals_score

    df = pd.DataFrame(scores)

    # Stream CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment;filename=static_scores.csv"},
    )
