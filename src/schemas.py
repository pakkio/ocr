from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class GradioCompatibleModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True
    )

    @classmethod
    def model_json_schema(cls, by_alias=True, ref_template='#/$defs/{model}'):
        schema = super().model_json_schema(by_alias=by_alias, ref_template=ref_template)
        def force_strict_schema(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "properties" in obj:
                    obj["additionalProperties"] = False
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        force_strict_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    force_strict_schema(item)
        force_strict_schema(schema)
        return schema

class ChartType(str, Enum):
    LINE = "line"
    LINE_CHART = "Line Chart"
    BAR = "bar"
    BAR_CHART = "Bar Chart"
    PIE = "pie"
    PIE_CHART = "Pie Chart"
    AREA = "area"
    AREA_CHART = "Area Chart"
    DONUT = "donut"
    DONUT_CHART = "Donut Chart"
    GAUGE = "gauge"
    GAUGE_CHART = "Gauge Chart"
    TABLE = "table"
    METRIC = "metric"

class DataPoint(GradioCompatibleModel):
    """Single data point in a chart"""
    label: str = Field(description="Label or category name")
    value: Union[float, int, str] = Field(description="Numeric value or text")
    percentage: Optional[float] = Field(None, description="Percentage if applicable")
    color: Optional[str] = Field(None, description="Color if visible")

class Chart(GradioCompatibleModel):
    """Individual chart or widget in dashboard"""
    title: Optional[str] = Field(None, description="Chart title if visible")
    type: ChartType = Field(description="Type of chart/visualization")
    data_points: List[DataPoint] = Field(default_factory=list, description="All data points in the chart")
    x_axis_label: Optional[str] = Field(None, description="X-axis label")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label")
    units: Optional[str] = Field(None, description="Units of measurement (%, $, etc.)")
    # We are removing the fragile field_validator as the prompt now handles this.

class MetricWidget(GradioCompatibleModel):
    """Key metric or KPI widget"""
    label: str = Field(description="Metric name or description")
    value: Union[float, int, str] = Field(description="Main metric value")
    units: Optional[str] = Field(None, description="Units (%, $, K, M, etc.)")
    trend: Optional[str] = Field(None, description="Trend indicator if visible (up, down, stable)")
    secondary_value: Optional[str] = Field(None, description="Secondary or comparison value")

class TimeSeriesData(GradioCompatibleModel):
    """Time series data point"""
    period: str = Field(description="Time period (year, month, date)")
    value: Union[float, int] = Field(description="Value for that period")

class TimeSeries(GradioCompatibleModel):
    """Time series chart data"""
    title: Optional[str] = Field(None, description="Chart title")
    series_name: Optional[str] = Field(None, description="Data series name")
    data: List[TimeSeriesData] = Field(description="Time series data points")

class DashboardData(GradioCompatibleModel):
    """Complete dashboard data extraction"""
    dashboard_title: Optional[str] = Field(None, description="Main dashboard title")
    charts: List[Chart] = Field(default_factory=list, description="All charts and visualizations")
    metrics: List[MetricWidget] = Field(default_factory=list, description="Key metrics and KPIs")
    time_series: List[TimeSeries] = Field(default_factory=list, description="Time series charts")
    text_content: List[str] = Field(default_factory=list, description="Other text content not in charts")
    watermarks: List[str] = Field(default_factory=list, description="Watermarks or credits to ignore")

class QualityAssessment(GradioCompatibleModel):
    """LLM assessment of extraction quality"""
    completeness_score: float = Field(ge=0, le=10, description="How complete is the extraction (0-10)")
    accuracy_score: float = Field(ge=0, le=10, description="How accurate are the extracted values (0-10)")
    structure_score: float = Field(ge=0, le=10, description="How well structured is the data (0-10)")
    missing_elements: List[str] = Field(description="Elements that appear missing")
    potential_errors: List[str] = Field(description="Potential extraction errors")
    confidence_level: str = Field(description="Overall confidence: high, medium, low")
    recommendations: List[str] = Field(description="Recommendations for improvement")

# --- THIS IS THE NEW, MORE ROBUST PROMPT ---
DASHBOARD_EXTRACTION_PROMPT = """
Analyze this dashboard/analytics image and extract ALL visible data into structured JSON format.

CHART TYPE NORMALIZATION:
When you identify a chart, you MUST set its 'type' field to one of the following exact string values:
["line", "Line Chart", "bar", "Bar Chart", "pie", "Pie Chart", "area", "Area Chart", "donut", "Donut Chart", "gauge", "Gauge Chart", "table", "metric"]
- If you see a 'progress ring' or 'percentage circle', use 'donut'.
- If you see a 'column chart', use 'bar'.
- If you see a 'KPI' or 'stat card', use 'metric'.

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

Extract everything systematically and precisely. If a value is partially obscured or unclear, make your best estimate.
Return the data in the exact JSON schema format specified.
"""

QUALITY_ASSESSMENT_PROMPT = """
You are an expert data analyst. Evaluate the quality of this JSON extraction from a dashboard image.
ORIGINAL IMAGE CONTEXT: {image_description}
EXTRACTED JSON: {extracted_json}
Assess the extraction quality across these dimensions ON A SCALE OF 0-10 (where 10 is perfect):
1. COMPLETENESS: Did we capture all visible data elements? (Score: 0-10)
2. ACCURACY: Are the numeric values and labels correct? (Score: 0-10)
3. STRUCTURE: Is the data properly organized and categorized? (Score: 0-10)
Look for:
- Missing charts or metrics that should be there
- Incorrect numeric values or percentages
- Misclassified chart types
- Poor data organization
- Watermarks incorrectly extracted as data
IMPORTANT: 
- All scores must be numbers between 0 and 10 (inclusive)
- Use decimal points for precision (e.g., 8.5, 9.2)
- Confidence level must be one of: "high", "medium", "low"
Provide specific, actionable feedback for improvement in the recommendations array.
"""
