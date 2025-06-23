from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    DONUT = "donut"
    GAUGE = "gauge"
    TABLE = "table"
    METRIC = "metric"

class DataPoint(BaseModel):
    """Single data point in a chart"""
    label: str = Field(description="Label or category name")
    value: Union[float, int, str] = Field(description="Numeric value or text")
    percentage: Optional[float] = Field(None, description="Percentage if applicable")
    color: Optional[str] = Field(None, description="Color if visible")

class Chart(BaseModel):
    """Individual chart or widget in dashboard"""
    title: Optional[str] = Field(None, description="Chart title if visible")
    type: ChartType = Field(description="Type of chart/visualization")
    data_points: List[DataPoint] = Field(description="All data points in the chart")
    x_axis_label: Optional[str] = Field(None, description="X-axis label")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label")
    units: Optional[str] = Field(None, description="Units of measurement (%, $, etc.)")

class MetricWidget(BaseModel):
    """Key metric or KPI widget"""
    label: str = Field(description="Metric name or description")
    value: Union[float, int, str] = Field(description="Main metric value")
    units: Optional[str] = Field(None, description="Units (%, $, K, M, etc.)")
    trend: Optional[str] = Field(None, description="Trend indicator if visible (up, down, stable)")
    secondary_value: Optional[str] = Field(None, description="Secondary or comparison value")

class TimeSeriesData(BaseModel):
    """Time series data point"""
    period: str = Field(description="Time period (year, month, date)")
    value: Union[float, int] = Field(description="Value for that period")

class TimeSeries(BaseModel):
    """Time series chart data"""
    title: Optional[str] = Field(None, description="Chart title")
    series_name: Optional[str] = Field(None, description="Data series name")
    data: List[TimeSeriesData] = Field(description="Time series data points")

class DashboardData(BaseModel):
    """Complete dashboard data extraction"""
    dashboard_title: Optional[str] = Field(None, description="Main dashboard title")
    charts: List[Chart] = Field(default_factory=list, description="All charts and visualizations")
    metrics: List[MetricWidget] = Field(default_factory=list, description="Key metrics and KPIs")
    time_series: List[TimeSeries] = Field(default_factory=list, description="Time series charts")
    text_content: List[str] = Field(default_factory=list, description="Other text content not in charts")
    watermarks: List[str] = Field(default_factory=list, description="Watermarks or credits to ignore")
    
    
class QualityAssessment(BaseModel):
    """LLM assessment of extraction quality"""
    completeness_score: float = Field(ge=0, le=10, description="How complete is the extraction (0-10)")
    accuracy_score: float = Field(ge=0, le=10, description="How accurate are the extracted values (0-10)")
    structure_score: float = Field(ge=0, le=10, description="How well structured is the data (0-10)")
    missing_elements: List[str] = Field(default_factory=list, description="Elements that appear missing")
    potential_errors: List[str] = Field(default_factory=list, description="Potential extraction errors")
    confidence_level: str = Field(description="Overall confidence: high, medium, low")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    

# Prompt templates for structured extraction
DASHBOARD_EXTRACTION_PROMPT = """
Analyze this dashboard/analytics image and extract ALL visible data into structured JSON format.

FOCUS ON:
- All numeric values (numbers, percentages, currencies)
- Chart data points and their labels
- Time series data with periods and values
- Key metrics and KPIs
- Chart titles and axis labels
- Any text content

IGNORE:
- Watermarks (iStock, Getty Images, etc.)
- Copyright notices
- Stock photo identifiers

Extract everything systematically and precisely. If a value is partially obscured or unclear, make your best estimate and note uncertainty.

Return the data in the exact JSON schema format specified.
"""

QUALITY_ASSESSMENT_PROMPT = """
You are an expert data analyst. Evaluate the quality of this JSON extraction from a dashboard image.

ORIGINAL IMAGE CONTEXT: {image_description}

EXTRACTED JSON: {extracted_json}

Assess the extraction quality across these dimensions:
1. COMPLETENESS: Did we capture all visible data elements?
2. ACCURACY: Are the numeric values and labels correct?
3. STRUCTURE: Is the data properly organized and categorized?

Look for:
- Missing charts or metrics that should be there
- Incorrect numeric values or percentages
- Misclassified chart types
- Poor data organization
- Watermarks incorrectly extracted as data

Provide specific, actionable feedback for improvement.
"""