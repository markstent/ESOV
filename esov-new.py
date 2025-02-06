# Part 1 - Imports and Setup
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
from fpdf import FPDF
from datetime import datetime
import streamlit.components.v1 as components
import webbrowser
import tempfile
import os

# Configure page settings
st.set_page_config(
    page_title="Advanced ESOV Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
    .insight-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Scenario class
# Scenario class
class ScenarioAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def calculate_response_rate(self, data, brand, market):
        """
        Calculate advertising response rate with controls and fallbacks.
        """
        # Get brand-market specific data
        market_data = data[
            (data['Brand'] == brand) &
            (data['Market'] == market)
        ].copy()
        
        # Sort by year to ensure proper calculations
        market_data = market_data.sort_values('Year')
        years_of_data = len(market_data['Year'].unique())
        
        # Check data sufficiency
        if years_of_data < 2:
            # Fallback to market average
            return self.calculate_market_response_rate(data, market)
        
        # Calculate year-over-year changes
        market_data['ESOV_change'] = market_data['ESOV'].diff()
        market_data['Share_change'] = market_data['Share of Market'].diff()
        
        # Remove invalid data points
        valid_data = market_data[
            (market_data['ESOV_change'].notna()) &
            (market_data['ESOV_change'] != 0) &
            (abs(market_data['ESOV_change']) <= market_data['ESOV_change'].std() * 2)
        ]
        
        if len(valid_data) < 2:
            # Fallback to market average if no valid changes
            return self.calculate_market_response_rate(data, market)
        
        # Calculate response rates
        valid_data['Response_Rate'] = valid_data['Share_change'] / valid_data['ESOV_change']
        
        # Apply controls for reasonable bounds
        reasonable_rates = valid_data[
            (valid_data['Response_Rate'] > 0) &
            (valid_data['Response_Rate'] < 0.5)
        ]
        
        if len(reasonable_rates) < 2:
            return {'response_rate': 0.15, 'confidence': 'low'}  # Conservative default
        
        response_rate = reasonable_rates['Response_Rate'].median()
        confidence = 'high' if years_of_data >= 4 else 'medium'
        
        return {
            'response_rate': response_rate,
            'confidence': confidence
        }

    def calculate_market_response_rate(self, data, market):
        """Calculate market-level response rate across all brands."""
        market_data = data[data['Market'] == market].copy()
        
        yearly_data = market_data.groupby(['Year', 'Brand']).agg({
            'ESOV': 'mean',
            'Share of Market': 'mean'
        }).reset_index()
        
        yearly_data['ESOV_change'] = yearly_data.groupby('Brand')['ESOV'].diff()
        yearly_data['Share_change'] = yearly_data.groupby('Brand')['Share of Market'].diff()
        
        valid_data = yearly_data[
            (yearly_data['ESOV_change'].notna()) &
            (yearly_data['ESOV_change'] != 0) &
            (abs(yearly_data['ESOV_change']) <= yearly_data['ESOV_change'].std() * 2)
        ]
        
        if len(valid_data) < 2:
            return {'response_rate': 0.15, 'confidence': 'low'}
        
        valid_data['Response_Rate'] = valid_data['Share_change'] / valid_data['ESOV_change']
        reasonable_rates = valid_data[
            (valid_data['Response_Rate'] > 0) &
            (valid_data['Response_Rate'] < 0.5)
        ]
        
        response_rate = reasonable_rates['Response_Rate'].median()
        return {
            'response_rate': response_rate,
            'confidence': 'medium'
        }
        
    def calculate_incrementality(self, brand, market, base_spend, incremental_spend):
        """
        Calculates expected incremental gains from additional spend.
        
        Parameters:
        - brand: Target brand
        - market: Target market
        - base_spend: Current spending level
        - incremental_spend: Additional spending to analyze
        
        Returns dictionary with:
        - Base metrics
        - New metrics with incremental spend
        - Expected gains from ESOV
        """
        # Check for sufficient historical data
        brand_market_data = self.data[
            (self.data['Brand'] == brand) &
            (self.data['Market'] == market)
        ]
        if len(brand_market_data['Year'].unique()) < 2:
            st.warning("Insufficient historical data for incrementality analysis. At least 2 years of data are required.")
            return None
            
        # Get latest year data
        latest_year = self.data['Year'].max()
        market_data = self.data[
            (self.data['Year'] == latest_year) &
            (self.data['Market'] == market)
        ]
        
        brand_data = market_data[market_data['Brand'] == brand]
        if brand_data.empty:
            raise ValueError("No data available for selected brand/market")
        
        # Calculate base metrics
        total_market_spend = market_data['Spend'].sum()
        current_sov = (base_spend / total_market_spend) * 100
        current_som = brand_data['Share of Market'].mean()
        current_esov = current_sov - current_som
        
        # Calculate new metrics with incremental spend
        new_total_spend = total_market_spend + incremental_spend
        new_brand_spend = base_spend + incremental_spend
        new_sov = (new_brand_spend / new_total_spend) * 100
        new_esov = new_sov - current_som
        
        # Calculate data-driven response rate
        response_metrics = self.calculate_response_rate(self.data, brand, market)
        base_response = response_metrics['response_rate']
        
        # Apply Headroom factor
        market_saturation = 1 - (current_som / 100)  # Harder to grow with higher share
        
        # Calculate incremental gain
        esov_improvement = new_esov - current_esov
        expected_share_gain = esov_improvement * base_response * market_saturation
        
        return {
            'current_metrics': {
                'spend': base_spend,
                'sov': current_sov,
                'som': current_som,
                'esov': current_esov
            },
            'new_metrics': {
                'spend': new_brand_spend,
                'sov': new_sov,
                'som': current_som + expected_share_gain,
                'esov': new_esov
            },
            'incrementality': {
                'spend_increase': incremental_spend,
                'esov_improvement': esov_improvement,
                'expected_share_gain': expected_share_gain,
                'response_rate': base_response,
                'response_confidence': response_metrics['confidence'],
                'market_saturation': market_saturation
            }
        }

    def optimise_market_budget(self, budget, brand, markets, filtered_data=None):
        """
        optimises budget allocation across markets using data-driven attractiveness scoring.
        """
        data_to_use = filtered_data if filtered_data is not None else self.data
        
        # Filter for selected brand and markets
        brand_data = data_to_use[
            (data_to_use['Brand'] == brand) & 
            (data_to_use['Market'].isin(markets))
        ]
        
        if brand_data.empty:
            raise ValueError("No data available for selected brand and markets")
        
        # Get latest year for allocation planning
        latest_year = brand_data['Year'].max()
        analysis_data = brand_data[brand_data['Year'] == latest_year]
        
        # Calculate market-level metrics
        market_metrics = []
        
        for market in markets:
            market_data = analysis_data[analysis_data['Market'] == market]
            if market_data.empty:
                continue
                
            # Calculate current spends
            current_brand_spend = market_data['Spend'].sum()
            total_market_spend = data_to_use[
                (data_to_use['Market'] == market) & 
                (data_to_use['Year'] == latest_year)
            ]['Spend'].sum()
            
            # Calculate share metrics
            current_sov = (current_brand_spend / total_market_spend) * 100
            current_som = market_data['Share of Market'].mean()
            current_esov = current_sov - current_som
            
            # Calculate market size index
            all_market_spends = data_to_use[
                data_to_use['Year'] == latest_year
            ].groupby('Market')['Spend'].sum()
            market_size_index = total_market_spend / all_market_spends.max()
            
            # Calculate attractiveness using smooth function
            # Add small constant to avoid division by zero
            attractiveness = (1 / (1 + abs(current_esov))) * market_size_index
                
            market_metrics.append({
                'market': market,
                'current_brand_spend': current_brand_spend,
                'total_market_spend': total_market_spend,
                'current_som': current_som,
                'current_sov': current_sov,
                'current_esov': current_esov,
                'market_size_index': market_size_index,
                'attractiveness_score': attractiveness
            })
        
        # Calculate allocation weights
        total_attractiveness = sum(m['attractiveness_score'] for m in market_metrics)
        allocations = []
        
        for mm in market_metrics:
            # Calculate budget allocation
            allocation_weight = mm['attractiveness_score'] / total_attractiveness
            allocated_budget = budget * allocation_weight
            
            # Calculate expected new metrics
            new_brand_spend = mm['current_brand_spend'] + allocated_budget
            new_sov = (new_brand_spend / mm['total_market_spend']) * 100
            new_esov = new_sov - mm['current_som']
            
            # Determine investment strategy based on ESOV changes
            if mm['current_esov'] < 0:
                if new_esov > 0:
                    strategy = 'Aggressive Growth'
                else:
                    strategy = 'Market Entry'
            else:
                if new_esov >= mm['current_esov']:
                    strategy = 'Market Leadership'
                else:
                    strategy = 'Maintain Position'
            
            allocations.append({
                'market': mm['market'],
                'market_size_index': round(mm['market_size_index'], 2),
                'current_esov': round(mm['current_esov'], 1),
                'current_spend': round(mm['current_brand_spend'], 0),
                'current_sov': round(mm['current_sov'], 1),
                'current_som': round(mm['current_som'], 1),
                'allocated_budget': round(allocated_budget, 0),
                'percentage_of_total': round(allocation_weight * 100, 1),
                'new_sov': round(new_sov, 1),
                'new_esov': round(new_esov, 1),
                'strategy': strategy,
                'attractiveness_score': round(mm['attractiveness_score'], 3)
            })
        
        return allocations
    
# Analyzer Class
class ESOVAnalyzer:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        
    def create_html_chart(self, fig, title):
        """Create HTML file for individual chart"""
        fig.update_layout(
            title=title,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            height=800,
            width=1200
        )
        
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, f"{title.lower().replace(' ', '_')}.html")
        fig.write_html(path)
        return path
        
    def load_data(self, file):
        try:
            if file.name.endswith('.csv'):
                self.data = pd.read_csv(file)
            else:
                self.data = pd.read_excel(file)
                
            required_columns = ['Year', 'Brand', 'Share of Market', 'Spend', 'Market']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
            self._preprocess_data()
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _preprocess_data(self):
        """Comprehensive data preprocessing"""
        # Convert numeric columns
        numeric_cols = ['Share of Market', 'Spend', 'Year']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Calculate Share of Voice from Spend
        self.data['Share of Voice'] = (self.data['Spend'] / self.data.groupby(['Year', 'Market'])['Spend'].transform('sum')) * 100
        
        # Calculate key metrics
        self.data['ESOV'] = self.data['Share of Voice'] - self.data['Share of Market']
        self.data['ESOV_Efficiency'] = (self.data['ESOV'] / self.data['Spend']) * 100
        
        self.data['Market_Position'] = self.data.groupby('Market')['Share of Market'].transform(lambda x: x.rank(pct=True, method='max'))
        
        # Calculate advanced metrics
        self.data['YOY_ESOV_Change'] = self.data.groupby(['Brand', 'Market'])['ESOV'].pct_change()
        self.data['ESOV_Rolling_Avg'] = self.data.groupby(['Brand', 'Market'])['ESOV'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean()
        )
        
        # Sort data
        self.data = self.data.sort_values(['Year', 'Market', 'Brand'])

    def create_trend_analysis(self, filtered_data, focus_brand='None', style_config=None):
        if style_config is None:
            style_config = {
                'primary_color': '#ffffff',
                'text_color': '#ffffff',
                'background_color': '#111111',
                'accent_color': '#e74c3c',
                'grid_color': '#333333'
            }

        brands = sorted(filtered_data['Brand'].unique())
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        colors = {brand: default_colors[i % len(default_colors)] for i, brand in enumerate(brands)}

        # Define styling functions
        marker_style = lambda is_focus, brand: dict(
            size=20 if is_focus else 12,
            color=style_config['accent_color'] if is_focus else colors[brand],
            line=dict(width=2 if is_focus else 0, color='white')
        )

        line_style = lambda is_focus, brand: dict(
            color=style_config['accent_color'] if is_focus else colors[brand],
            width=3 if is_focus else 1
        )
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'ESOV Trends Over Time', 'Share of Voice vs Share of Market',
                'ESOV Efficiency by Brand (ESOV/Spend)', 'Market Position Analysis',
                'ESOV Heatmap', 'Brand Performance Matrix'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        fig.update_layout(
            height=1200,
            template="plotly_dark",
            plot_bgcolor=style_config['background_color'],
            paper_bgcolor=style_config['background_color'],
            font=dict(family="Roboto", size=11, color=style_config['text_color']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(t=150)
        )
        filtered_data['Year_str'] = filtered_data['Year'].astype(int).astype(str)
        yearly_brand_data = filtered_data.groupby(['Brand', 'Year_str'])['ESOV'].mean().reset_index()

        # 1. ESOV Trends
        for brand in brands:
            is_focus = brand == focus_brand
            brand_data = yearly_brand_data[yearly_brand_data['Brand'] == brand].sort_values('Year_str')
            fig.add_trace(
                go.Scatter(
                    x=brand_data['Year_str'],
                    y=brand_data['ESOV'],
                    name=brand,
                    mode='lines+markers',
                    line=line_style(is_focus, brand),
                    marker=marker_style(is_focus, brand),
                    opacity=1 if is_focus else 0.7
                ),
                row=1, col=1
            )

        # 2. SOV vs SOM
        brand_avg = filtered_data.groupby('Brand').agg({
            'Share of Market': 'mean',
            'Share of Voice': 'mean'
        }).reset_index()
        
        max_value = max(filtered_data['Share of Voice'].max(), filtered_data['Share of Market'].max())
        min_value = min(filtered_data['Share of Voice'].min(), filtered_data['Share of Market'].min())

        fig.add_trace(
            go.Scatter(
                x=[min_value, max_value],
                y=[min_value, max_value],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='SOV = SOM',
                showlegend=True
            ),
            row=1, col=2
        )

        for brand in brands:
            is_focus = brand == focus_brand
            brand_data = brand_avg[brand_avg['Brand'] == brand]
            fig.add_trace(
                go.Scatter(
                    x=brand_data['Share of Market'],
                    y=brand_data['Share of Voice'],
                    mode='markers+text' if is_focus else 'markers',
                    marker=marker_style(is_focus, brand),
                    text=brand,
                    textposition='top center' if is_focus else None,
                    showlegend=False
                ),
                row=1, col=2  
            )

        # 3. ESOV Efficiency
        efficiency_data = filtered_data.groupby('Brand')['ESOV_Efficiency'].mean()
        bar_colors = [
            style_config['accent_color'] if brand == focus_brand 
            else colors[brand] 
            for brand in efficiency_data.index
        ]
        
        fig.add_trace(
            go.Bar(
                x=efficiency_data.index,
                y=efficiency_data.values,
                marker=dict(color=bar_colors),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Market Position
        brand_position = filtered_data.groupby('Brand').agg({
            'Market_Position': 'mean',
            'ESOV': 'mean'
        }).reset_index()
        
        for brand in brands:
            is_focus = brand == focus_brand
            brand_data = brand_position[brand_position['Brand'] == brand]
            fig.add_trace(
                go.Scatter(
                    x=brand_data['Market_Position'],
                    y=brand_data['ESOV'],
                    mode='markers+text' if is_focus else 'markers',
                    marker=marker_style(is_focus, brand),
                    text=brand,
                    textposition='top center' if is_focus else None,
                    showlegend=False
                ),
                row=2, col=2
            )

        # 5. ESOV Heatmap
        pivot_data = filtered_data.pivot_table(
            values='ESOV',
            index='Brand',
            columns='Year_str',
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdBu',
                text=np.round(pivot_data.values, 1),
                texttemplate="%{text}%",
                colorbar=dict(
                    title="ESOV %",
                    thickness=15,
                    len=0.3,
                    x=0.415,
                    y=0.125,
                    yanchor='middle'
                )
            ),
            row=3, col=1
        )

        # 6. Brand Performance Matrix
        for brand in brands:
            is_focus = brand == focus_brand
            brand_data = filtered_data[filtered_data['Brand'] == brand]
            fig.add_trace(
                go.Scatter(
                    x=[brand_data['Share of Market'].mean()],
                    y=[brand_data['ESOV'].mean()],
                    mode='markers+text' if is_focus else 'markers',
                    marker=marker_style(is_focus, brand),
                    text=brand,
                    textposition='top center' if is_focus else None,
                    showlegend=False
                ),
                row=3, col=2
            )

        axes_updates = {
            (1,1): ("Year", "ESOV (%)"),
            (1,2): ("Share of Market (%)", "Share of Voice (%)"),
            (2,1): ("Brand", "ESOV Efficiency (%)"),
            (2,2): ("Market Position", "ESOV (%)"),
            (3,1): ("Year", "Brand"),
            (3,2): ("Share of Market (%)", "ESOV (%)")
        }

        for (row, col), (xlabel, ylabel) in axes_updates.items():
            fig.update_xaxes(title_text=xlabel, row=row, col=col, gridcolor=style_config['grid_color'])
            fig.update_yaxes(title_text=ylabel, row=row, col=col, gridcolor=style_config['grid_color'])

        return fig
                
    def generate_insights(self, filtered_data):
        """Generate actionable insights"""
        insights = []
        
        # Market overview
        market_stats = {
            'avg_esov': filtered_data['ESOV'].mean(),
            'top_brand': filtered_data.loc[filtered_data['Share of Market'].idxmax()]['Brand'],
            'market_concentration': filtered_data.groupby('Brand')['Share of Market'].mean().std()
        }
        
        market_text = "All Markets" if len(filtered_data['Market'].unique()) > 1 else filtered_data['Market'].iloc[0]
        
        insights.append({
            'title': f'Market Overview - {market_text}',
            'content': f"""
            - Average ESOV: {market_stats['avg_esov']:.2f}%
            - Market Leader: {market_stats['top_brand']}
            - Market Concentration (std dev of market share): {'High (>10)' if market_stats['market_concentration'] > 10 else 'Moderate (5-10)' if market_stats['market_concentration'] > 5 else 'Low (<5)'}
            """
        })
        
        # Brand-specific insights
        for brand in filtered_data['Brand'].unique():
            brand_data = filtered_data[filtered_data['Brand'] == brand]
            brand_insights = self._analyze_brand(brand_data)
            insights.append({
                'title': f'{brand} Analysis',
                'content': brand_insights
            })
            
        return insights
    @staticmethod
    def format_percentage(value, decimals=2):
        """Format percentage values, handling zero case"""
        if abs(value) < 10**(-decimals):  # Handle values very close to zero
            return f"{0:.{decimals}f}%"
        return f"{value:.{decimals}f}%"

    def generate_recommendations(self, metrics):
        """Generate recommendations based on metrics analysis"""
        recommendations = []
        if metrics['avg_esov'] < 0:
            recommendations.append("Increase marketing investment to improve market position")
        if metrics['efficiency'] < self.data['ESOV_Efficiency'].mean():
            recommendations.append("Review and optimize spend allocation for better efficiency")
        if metrics['market_position'] > 0.7:
            recommendations.append("Defend market position through strategic investment")
        return recommendations if recommendations else ["Maintain current strategy"]

    def _analyze_brand(self, brand_data):
        metrics = {
            'avg_esov': brand_data['ESOV'].mean(),
            'trend': brand_data['YOY_ESOV_Change'].mean(),
            'efficiency': brand_data['ESOV_Efficiency'].mean(),
            'market_position': brand_data['Market_Position'].mean()
        }
        
        recommendations = self.generate_recommendations(metrics)
        
        metrics_output = (
            f"- Average ESOV: {self.format_percentage(metrics['avg_esov'])}\n"
            f"- YOY Trend: {self.format_percentage(metrics['trend'])}\n"
            f"- ESOV Efficiency: {self.format_percentage(metrics['efficiency'])}"
        )
        
        recommendations_output = "\n".join(f"- {r}" for r in recommendations)
        
        return f"Performance Metrics:\n{metrics_output}\n\nRecommendations:\n{recommendations_output}"
        
    @staticmethod
    def generate_pdf_report_to_file(filtered_data, selected_years, selected_market, selected_brands, report_name, file_path, insights, analyzer):
        """Generate a comprehensive PDF report including insights, recommendations, and visualizations."""
        try:
            # Validate inputs
            if filtered_data is None or filtered_data.empty:
                raise ValueError("No data available for report generation")

            # Create PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Try to use Arial which has better Unicode support than Helvetica
            try:
                pdf.set_font('Arial', '')
                font_family = 'Arial'
            except:
                # Fall back to Helvetica if Arial is not available
                pdf.set_font('Helvetica', '')
                font_family = 'Helvetica'
            
            # Add cover page
            pdf.add_page()
            pdf.set_font(font_family, 'B', 24)
            pdf.ln(60)
            pdf.cell(0, 10, report_name, ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Helvetica", '', 12)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", ln=True, align='C')
            pdf.cell(0, 10, f"Market Analysis: {selected_market}", ln=True, align='C')
            
            # Add table of contents
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "Table of Contents", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", '', 12)
            
            sections = [
                "1. Executive Summary",
                "2. Market Overview",
                "3. Brand Performance Analysis",
                "4. ESOV Analysis",
                "5. Competitive Landscape",
                "6. Recommendations",
                "7. Methodology & Definitions"
            ]
            
            for section in sections:
                pdf.cell(0, 8, section, ln=True)
            
            # 1. Executive Summary
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "1. Executive Summary", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", '', 12)
            
            summary_text = f"""
    This report analyzes the ESOV (Excess Share of Voice) performance across {len(selected_brands)} brands in {selected_market if selected_market != 'All' else 'all markets'} from {min(selected_years)} to {max(selected_years)}.

    Key Findings:
    * Average market ESOV: {filtered_data['ESOV'].mean():.1f}%
    * Leading brand: {filtered_data.loc[filtered_data['ESOV'].idxmax(), 'Brand']}
    * Number of brands showing positive ESOV: {len(filtered_data[filtered_data['ESOV'] > 0]['Brand'].unique())}
            """
            pdf.multi_cell(0, 6, summary_text)
            
            # 2. Market Overview
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "2. Market Overview", ln=True)
            pdf.ln(5)
            
            # Define style configuration
            style_config = {
                'background_color': '#ffffff',
                'text_color': '#000000',
                'grid_color': '#e0e0e0',
            }
            
            # Color palette for brands
            brands = sorted(filtered_data['Brand'].unique())
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            colors = {brand: default_colors[i % len(default_colors)] for i, brand in enumerate(brands)}

            # Create and save market overview charts
            market_fig = go.Figure()
            
            # Market Share Distribution
            shares = filtered_data.groupby('Brand')['Share of Market'].mean().sort_values(ascending=True)
            
            market_fig.add_trace(go.Bar(
                x=shares.values,
                y=shares.index,
                orientation='h',
                name='Share of Market',
                marker_color=[colors[brand] for brand in shares.index]
            ))
            
            market_fig.update_layout(
                title='Market Share Distribution',
                xaxis_title='Share of Market (%)',
                yaxis_title='Brand',
                height=400,
                plot_bgcolor=style_config['background_color'],
                paper_bgcolor=style_config['background_color'],
                font=dict(color=style_config['text_color']),
                showlegend=True
            )
            
            market_fig.update_xaxes(gridcolor=style_config['grid_color'], zeroline=False)
            market_fig.update_yaxes(gridcolor=style_config['grid_color'], zeroline=False)
            
            # Save the figure as a temporary image
            market_chart_path = "market_overview.png"
            market_fig.write_image(market_chart_path)
            
            # Add the chart to PDF
            pdf.image(market_chart_path, x=10, y=None, w=190)
            pdf.ln(140)  # Space for the image
            
            # 3. Brand Performance Analysis
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "3. Brand Performance Analysis", ln=True)
            pdf.ln(5)
            
            # Create brand performance matrix
            perf_fig = go.Figure()
            
            # Calculate average metrics for each brand
            brand_metrics = filtered_data.groupby('Brand').agg({
                'Share of Market': 'mean',
                'ESOV': 'mean'
            }).reset_index()
            
            for brand in brand_metrics['Brand']:
                brand_data = brand_metrics[brand_metrics['Brand'] == brand]
                perf_fig.add_trace(go.Scatter(
                    x=brand_data['Share of Market'],
                    y=brand_data['ESOV'],
                    mode='markers+text',
                    text=brand,
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=colors[brand],
                        line=dict(width=1, color='rgba(0,0,0,0.2)')
                    ),
                    name=brand
                ))
            
            perf_fig.update_layout(
                title='Brand Performance Matrix',
                xaxis_title='Share of Market (%)',
                yaxis_title='ESOV (%)',
                height=400,
                plot_bgcolor=style_config['background_color'],
                paper_bgcolor=style_config['background_color'],
                font=dict(color=style_config['text_color']),
                showlegend=True
            )
            
            # Add zero line
            perf_fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)")
            
            perf_fig.update_xaxes(gridcolor=style_config['grid_color'], zeroline=False)
            perf_fig.update_yaxes(gridcolor=style_config['grid_color'], zeroline=False)
            
            # Add zero line
            perf_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Save the performance matrix
            perf_chart_path = "performance_matrix.png"
            perf_fig.write_image(perf_chart_path)
            
            # Add the chart to PDF
            pdf.image(perf_chart_path, x=10, y=None, w=190)
            pdf.ln(140)  # Space for the image
            
            # 4. ESOV Analysis
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "4. ESOV Analysis", ln=True)
            pdf.ln(5)
            
            # Create ESOV trends chart
            esov_fig = go.Figure()
            
            for brand in selected_brands:
                brand_data = filtered_data[filtered_data['Brand'] == brand]
                yearly_esov = brand_data.groupby('Year')['ESOV'].mean()
                
                esov_fig.add_trace(go.Scatter(
                    x=yearly_esov.index,
                    y=yearly_esov.values,
                    name=brand,
                    mode='lines+markers',
                    line=dict(
                        color=colors[brand],
                        width=2
                    ),
                    marker=dict(
                        size=8,
                        color=colors[brand],
                        line=dict(width=1, color='rgba(0,0,0,0.2)')
                    )
                ))
            
            esov_fig.update_layout(
                title='ESOV Trends Over Time',
                xaxis_title='Year',
                yaxis_title='ESOV (%)',
                height=400,
                plot_bgcolor=style_config['background_color'],
                paper_bgcolor=style_config['background_color'],
                font=dict(color=style_config['text_color']),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            esov_fig.update_xaxes(gridcolor=style_config['grid_color'], zeroline=False)
            esov_fig.update_yaxes(gridcolor=style_config['grid_color'], zeroline=False)
            
            # Save the ESOV trends chart
            esov_chart_path = "esov_trends.png"
            esov_fig.write_image(esov_chart_path)
            
            # Add the chart to PDF
            pdf.image(esov_chart_path, x=10, y=None, w=190)
            pdf.ln(140)  # Space for the image
            
            # 5. Competitive Landscape
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "5. Competitive Landscape", ln=True)
            pdf.ln(5)
            
            # Create detailed performance table
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(50, 10, "Brand", border=1)
            pdf.cell(35, 10, "Avg SOM%", border=1)
            pdf.cell(35, 10, "Avg SOV%", border=1)
            pdf.cell(35, 10, "ESOV%", border=1)
            pdf.cell(35, 10, "Efficiency", border=1)
            pdf.ln()
            
            pdf.set_font("Helvetica", '', 10)
            for brand in selected_brands:
                brand_data = filtered_data[filtered_data['Brand'] == brand]
                pdf.cell(50, 8, brand, border=1)
                pdf.cell(35, 8, f"{brand_data['Share of Market'].mean():.1f}%", border=1)
                pdf.cell(35, 8, f"{brand_data['Share of Voice'].mean():.1f}%", border=1)
                pdf.cell(35, 8, f"{brand_data['ESOV'].mean():.1f}%", border=1)
                pdf.cell(35, 8, f"{brand_data['ESOV_Efficiency'].mean():.1f}", border=1)
                pdf.ln()
            
            # 6. Recommendations
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "6. Recommendations", ln=True)
            pdf.ln(5)
            
            # Add insights and recommendations
            for insight in insights:
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 8, insight['title'], ln=True)
                pdf.set_font("Helvetica", '', 11)
                pdf.multi_cell(0, 6, insight['content'])
                pdf.ln(5)
            
            # 7. Methodology & Definitions
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "7. Methodology & Definitions", ln=True)
            pdf.ln(5)
            
            methodology_text = """
    Key Metrics Definitions:

    ESOV (Excess Share of Voice):
    The difference between a brand's share of voice and share of market. A positive ESOV typically indicates potential for market share growth.

    Share of Voice (SOV):
    The percentage of category advertising spending attributed to a specific brand.

    Share of Market (SOM):
    The percentage of total category sales or market share held by a brand.

    ESOV Efficiency:
    Measures how effectively advertising spending translates into market share gains, calculated as ESOV divided by total spend.

    Methodology Notes:
    * All market share data is based on reported figures
    * Share of Voice calculations include all measured media spending
    * ESOV calculations follow industry-standard methodologies
    * Efficiency metrics account for market-specific factors and competitive intensity
    """
            pdf.set_font("Helvetica", '', 11)
            pdf.multi_cell(0, 6, methodology_text)
            
            # Clean up temporary files
            import os
            for temp_file in [market_chart_path, perf_chart_path, esov_chart_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Save the complete PDF
            pdf.output(file_path)
            return True
                    
        except Exception as e:
            return f"Error occurred: {str(e)}"

def main():
    st.markdown("""
        <style>
        .css-1d391kg {background-color: #111111;}
        .stMarkdown, .stHeader {color: #ffffff;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Advanced ESOV Analysis Dashboard")
    tab1, tab2, tab4, tab3 = st.tabs(["Overview", "Analysis", "Scenarios", "Data & Export"])

    with tab1:
        
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", 
            type=["csv", "xlsx"], 
            key="data_uploader_tab1"
        )
        
        if not uploaded_file:
            st.markdown("### Required Data Format")
            st.markdown("Your data should include the following columns:")
            
            # Create sample data
            sample_data = pd.DataFrame({
                'Year': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
                'Brand': ['Brand A', 'Brand B', 'Brand A', 'Brand B', 'Brand A', 'Brand B', 'Brand A', 'Brand B'],
                'Market': ['UK', 'UK', 'US', 'US', 'UK', 'UK', 'US', 'US'],
                'Share of Market (%)': [35.0, 25.0, 30.0, 28.0, 36.0, 26.0, 31.0, 29.0],
                'Spend': [1000000, 800000, 900000, 850000, 1100000, 850000, 950000, 900000]
            })
            
            # Display sample data
            st.dataframe(
                sample_data.style.format({
                    'Share of Market': '{:.1f}',
                    'Spend': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            st.markdown("""
            **Required Columns:**
            - `Year`: Numeric year (e.g., 2023)
            - `Brand`: Brand name/identifier
            - `Market`: Market name/identifier
            - `Share of Market`: Market share percentage
            - `Spend`: Advertising spend in monetary units
            """)
            
            st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)

    if uploaded_file:
        try:
            analyzer = ESOVAnalyzer()
            data = analyzer.load_data(uploaded_file)
            scenario_analyzer = ScenarioAnalyzer(data)
            
            with st.sidebar:
                st.header("Analysis Filters")
                selected_market = st.selectbox("Market", ['All'] + sorted(data['Market'].unique().tolist()),
                                           key="sidebar_market_selection")
                selected_years = st.multiselect("Years", options=sorted(data['Year'].unique()), 
                                             default=sorted(data['Year'].unique()),
                                             key="sidebar_years_multiselect")
                selected_brands = st.multiselect("Brands", options=sorted(data['Brand'].unique()), 
                                              default=sorted(data['Brand'].unique()),
                                              key="sidebar_brands_multiselect")
                focus_brand = st.selectbox("Focus Brand", ['None'] + sorted(data['Brand'].unique()),
                                       key="sidebar_focus_brand")
                
                st.header("Style Settings")
                primary_color = st.color_picker("Primary Color", "#ffffff", key="style_primary_color_picker")
                text_color = st.color_picker("Text Color", "#ffffff", key="style_text_color_picker")
                background_color = st.color_picker("Background Color", "#111111", key="style_background_color_picker")
                accent_color = st.color_picker("Accent Color", "#e74c3c", key="style_accent_color_picker")

            style_config = {
                'primary_color': primary_color,
                'text_color': text_color,
                'background_color': background_color,
                'accent_color': accent_color,
                'grid_color': '#333333'
            }

            filtered_data = data[
                (data['Year'].isin(selected_years)) &
                (data['Brand'].isin(selected_brands))
            ]
            if selected_market != 'All':
                filtered_data = filtered_data[filtered_data['Market'] == selected_market]

            # Tab 1: Overview
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Markets", len(data['Market'].unique()))
                with col2:
                    st.metric("Total Brands", len(data['Brand'].unique()))
                with col3:
                    st.metric("Average ESOV", f"{data['ESOV'].mean():.2f}%")
                with col4:
                    st.metric("Date Range", f"{data['Year'].min()} - {data['Year'].max()}")

            # Tab 2: Analysis
            with tab2:
                st.markdown("""
                    <style>
                    .stButton button {
                        font-size: 12px;
                        padding: 2px 6px;
                        margin: 0;
                        min-height: 0;
                    }
                    </style>
                """, unsafe_allow_html=True)
                fig = analyzer.create_trend_analysis(filtered_data, focus_brand, style_config)
                st.plotly_chart(fig, use_container_width=True)
                
                insights = analyzer.generate_insights(filtered_data)
                for insight in insights:
                    if 'Overview' in insight['title']:
                        with st.expander(insight['title']):
                            st.markdown(insight['content'])
                    elif focus_brand != 'None' and focus_brand in insight['title']:
                        with st.expander(insight['title']):
                            st.markdown(insight['content'])
                            
                with st.expander("Understanding the Dashboard Visualizations"):
                    st.markdown("""
                    ### 1. ESOV Trends Over Time
                    Shows each brand's Excess Share of Voice (ESOV) trajectory across years. Upward trends suggest growing market presence relative to share, while downward trends may indicate underinvestment. The focus brand appears highlighted in accent color.
                    
                    ### 2. Share of Voice vs Share of Market
                    Plots each brand's advertising presence against market share. Points above the diagonal line indicate higher advertising share than market share (positive ESOV), suggesting growth potential. Points below show underinvestment relative to current position.
                    
                    ### 3. ESOV Efficiency by Brand
                    Measures the efficiency of advertising investment using the formula: `ESOV_Efficiency = (ESOV / Spend) × 100`. Higher bars indicate better return on ad spend - more ESOV points generated per unit of spend. A brand with 2% efficiency generates 2 ESOV percentage points for every 100 units of spend. This metric helps identify which brands are most cost-effective at converting advertising investment into market presence.
                    
                    ### 4. Market Position Analysis
                    Shows the relationship between market position (x-axis) and ESOV (y-axis). Brands in the top right are market leaders with strong advertising support. Bottom left indicates challenger brands, while top left shows aggressive newcomers.
                    
                    ### 5. ESOV Heatmap
                    Visualizes ESOV patterns across brands and years. Blue indicates positive ESOV (overinvestment), red shows negative ESOV (underinvestment). Intensity shows magnitude. This helps spot temporal patterns and competitive dynamics.
                    
                    ### 6. Brand Performance Matrix
                    Maps brands by market share (x-axis) versus ESOV (y-axis). This reveals competitive positioning and potential strategic moves. Quadrants indicate different strategic situations: defend (top right), grow (top left), maintain (bottom right), or invest (bottom left).
                    """)
                    
                st.subheader("Print-Friendly Charts")
                cols = st.columns(3)
                charts = [
                    ("ESOV Trends", 1, 1),
                    ("Share Analysis", 1, 2),
                    ("Efficiency Analysis", 2, 1),
                    ("Market Position", 2, 2),
                    ("ESOV Heatmap", 3, 1),
                    ("Performance Matrix", 3, 2)
                ]

                st.markdown("<div style='display:flex; gap:5px; flex-wrap:wrap'>", unsafe_allow_html=True)

                charts = [
                    ("ESOV Trends", 1, 1),
                    ("Share Analysis", 1, 2),
                    ("Efficiency Analysis", 2, 1),
                    ("Market Position", 2, 2), 
                    ("ESOV Heatmap", 3, 1),
                    ("Performance Matrix", 3, 2)
                ]

                for idx, (title, row, col) in enumerate(charts):
                    individual_fig = go.Figure()
                    for trace in fig.select_traces(row=row, col=col):
                        individual_fig.add_trace(trace)
                    
                    individual_fig.update_layout(
                        title=title,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='black'),
                        height=800,
                        width=1200,
                        xaxis=dict(
                            tickmode='array',
                            ticktext=filtered_data['Year'].unique().astype(int).astype(str),
                            tickvals=filtered_data['Year'].unique()
                        ) if title == "ESOV Trends" else {}
                    )
                    
                    if st.button(title, key=f"print_{title}", type="secondary", use_container_width=False):
                        temp_dir = tempfile.mkdtemp()
                        path = os.path.join(temp_dir, f"{title.lower().replace(' ', '_')}.html")
                        individual_fig.write_html(path)
                        webbrowser.open(f'file://{path}')

                st.markdown("</div>", unsafe_allow_html=True)

            # Tab 3: Data & Export
            with tab3:
                display_columns = ['Year', 'Brand', 'Market', 'Share of Voice', 'Share of Market', 'ESOV', 'ESOV_Efficiency']
                table_data = filtered_data[display_columns].copy()
                
                format_columns = ['Share of Voice', 'Share of Market', 'ESOV', 'ESOV_Efficiency']
                for col in format_columns:
                    table_data[col] = table_data[col].round(2).astype(str) + '%'
                
                table_data = table_data.sort_values(['Year', 'Market', 'Brand'])
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    csv = table_data.to_csv(index=False)
                    st.download_button(
                        "Download Table Data", 
                        csv, 
                        "esov_table_data.csv", 
                        "text/csv",
                        key="tab3_download_table_data"
                    )
                
                st.dataframe(table_data, use_container_width=True, hide_index=True)
                
                report_name = st.text_input("Report name:", "ESOV Analysis Report", key="report_name_tab3")
                current_date = datetime.now().strftime("%Y-%m-%d")
                file_name = f"{report_name} ({current_date}).pdf"
                
                if st.button("Generate PDF Report", key="tab3_generate_pdf_button"):
                    result = analyzer.generate_pdf_report_to_file(
                        filtered_data, selected_years, selected_market, selected_brands, 
                        report_name, file_name, insights, analyzer
                    )
                    if isinstance(result, bool) and result:
                        with open(file_name, "rb") as pdf_file:
                            pdf_content = pdf_file.read()
                            st.download_button(
                                "Download Report",
                                pdf_content,
                                file_name=report_name + ".pdf",
                                mime="application/pdf",
                                key="tab3_download_pdf_report"
                            )
                    else:
                        st.error(result)
# Tab 4: Scenarios
            with tab4:
                st.header("Scenario Analysis")
                scenario = st.radio(
                    "Select Scenario:",
                    ["Budget Allocation Optimisation", "Incrementality Analysis"],
                    key="scenario_selector_tab4"
                )

                if scenario == "Budget Allocation Optimisation":
                    col1, col2 = st.columns(2)
                    with col1:
                        budget = st.number_input(
                            "Available Budget",
                            min_value=0,
                            max_value=1000000000,
                            value=1000000,
                            key="tab4_budget_input"
                        )
                    with col2:
                        target_brand = st.selectbox(
                            "Select Brand to optimise",
                            sorted(data['Brand'].unique()),
                            key="tab4_target_brand_select2"
                        )
                    
                    target_markets = st.multiselect(
                        "Select Markets to optimise",
                        sorted(data[data['Brand'] == target_brand]['Market'].unique()),
                        default=sorted(data[data['Brand'] == target_brand]['Market'].unique()),
                        key="tab4_target_markets_multiselect"
                    )

                    if st.button("Get Optimal Market Allocation", key="tab4_allocation_button"):
                        if not target_markets:
                            st.warning("Please select at least one market.")
                            return
                        try:
                            allocations = scenario_analyzer.optimise_market_budget(budget, target_brand, target_markets, data)
                            
                            # Display summary metrics
                            st.subheader("Budget Allocation Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Budget", f"{budget:,.0f}")
                            with col2:
                                markets_improving = sum(1 for alloc in allocations if alloc['new_esov'] > alloc['current_esov'])
                                st.metric("Markets Improving ESOV", f"{markets_improving} of {len(allocations)}")
                            with col3:
                                avg_improvement = np.mean([alloc['new_esov'] - alloc['current_esov'] for alloc in allocations])
                                st.metric("Average ESOV Improvement", f"{avg_improvement:.1f}%")

                            # Create DataFrame for display
                            allocation_df = pd.DataFrame(allocations)
                            
                            # Display current state
                            st.subheader("Current Market Positions")
                            current_metrics = allocation_df[['market', 'market_size_index', 'current_spend', 
                                                        'current_sov', 'current_som',  'current_esov']]
                            current_metrics.columns = ['Market', 'Market Size Index', 'Current Spend', 
                                                    'Share of Voice %', 'Share of Market %',  'ESOV %']
                            st.dataframe(
                                current_metrics.style
                                .format({
                                    'Market Size Index': '{:.2f}',
                                    'Current Spend': '{:,.0f}',
                                    'Share of Voice %': '{:.1f}%',
                                    'Share of Market %': '{:.1f}%',
                                    'ESOV %': '{:.1f}%'
                                })
                                .background_gradient(subset=['ESOV %'], cmap='RdYlGn')
                            )

                            # Display allocation results
                            st.subheader("Recommended Allocations")
                            allocation_metrics = allocation_df[['market', 'allocated_budget', 'percentage_of_total', 'attractiveness_score',
                                                            'new_sov', 'new_esov', 'strategy']]
                            allocation_metrics.columns = ['Market', 'Allocated Budget', 'Budget Share %', 'Attractiveness Score %', 
                                                        'New SOV %', 'New ESOV %', 'Strategy']
                            st.dataframe(
                                allocation_metrics.style
                                .format({
                                    'Allocated Budget': '{:,.0f}',
                                    'Budget Share %': '{:.1f}%',
                                    'Attractiveness Score %': '{:.1f}%',
                                    'New SOV %': '{:.1f}%',
                                    'New ESOV %': '{:.1f}%'
                                })
                                .background_gradient(subset=['New ESOV %'], cmap='RdYlGn')
                            )

                            # Display allocation explanation
                            with st.expander("Understanding the Allocation Logic"):
                                st.markdown("""
                                ### Market Budget Allocation Methodology
                                
                                The allocation model uses the following metrics and logic:

                                #### 1. Market Size Index
                                - Calculated relative to largest market
                                - Helps prioritize larger revenue opportunities
                                - Index range: 0 to 1 (1 = largest market)

                                #### 2. Current ESOV Position
                                - ESOV = Share of Voice - Share of Market
                                - Negative ESOV indicates underinvestment
                                - Positive ESOV indicates overinvestment

                                #### 3. Attractiveness Score
                                - Score = (1 / (1 + |ESOV|)) × Market Size Index
                                - Higher scores for markets closer to equilibrium
                                - Scaled by market size for final allocation
                                - Creates natural diminishing returns as |ESOV| increases
                                - Prevents overallocation to any single market
                                - Smoothly penalizes markets with extreme ESOV values
                                
                                Examples (before market size scaling):
                                - ESOV = 0%:   Score = 1.000
                                - ESOV = ±5%:  Score = 0.167
                                - ESOV = ±10%: Score = 0.091
                                - ESOV = ±20%: Score = 0.048

                                #### 4. Budget Allocation
                                - Weight = Market Attractiveness / Total Attractiveness
                                - Budget = Weight × Total Available Budget
                                
                                #### 5. Strategy Classification
                                - Market Entry: Negative ESOV, staying negative
                                - Aggressive Growth: Negative to positive ESOV
                                - Market Leadership: Maintaining positive ESOV
                                - Maintain Position: Reducing positive ESOV
                                """)
        

                                # Show allocation summary
                                st.subheader("Allocation Overview")
                                summary_metrics = pd.DataFrame({
                                    'Metric': [
                                        'Total Markets',
                                        'Average Market Size Index',
                                        'Markets with Negative ESOV',
                                        'Average ESOV Improvement',
                                        'Largest Allocation',
                                        'Smallest Allocation'
                                    ],
                                    'Value': [
                                        f"{len(allocation_df)}",
                                        f"{allocation_df['market_size_index'].mean():.2f}",
                                        f"{sum(allocation_df['current_esov'] < 0)}",
                                        f"{avg_improvement:.1f}%",
                                        f"{allocation_df['allocated_budget'].max():,.0f} ({allocation_df['percentage_of_total'].max():.1f}%)",
                                        f"{allocation_df['allocated_budget'].min():,.0f} ({allocation_df['percentage_of_total'].min():.1f}%)"
                                    ]
                                })
                                st.table(summary_metrics)
                        except Exception as e:
                            st.error(f"Error in budget allocation: {str(e)}")
                else:  # Incrementality Analysis
                                    st.subheader("Incrementality Analysis")
                                    
                                    # Check data requirements first
                                    year_count = len(data['Year'].unique())
                                    if year_count < 2:
                                        st.warning("""
                                            ⚠️ Incrementality analysis requires at least 2 years of historical data to make predictions.
                                            
                                            Current data only contains {year_count} year(s). Please provide additional historical data 
                                            to use this feature.
                                        """.format(year_count=year_count))
                                        st.stop()
                                    
                                    # Setup UI inputs
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        inc_brand = st.selectbox(
                                            "Select Brand",
                                            sorted(data['Brand'].unique()),
                                            key="incrementality_brand"
                                        )
                                    with col2:
                                        inc_market = st.selectbox(
                                            "Select Market",
                                            sorted(data[data['Brand'] == inc_brand]['Market'].unique()),
                                            key="incrementality_market"
                                        )
                                    
                                    # Check brand-market specific history
                                    brand_market_years = len(data[
                                        (data['Brand'] == inc_brand) & 
                                        (data['Market'] == inc_market)
                                    ]['Year'].unique())
                                    
                                    if brand_market_years < 2:
                                        st.warning(f"""
                                            ⚠️ Selected brand-market combination has only {brand_market_years} year(s) of data.
                                            
                                            Please select a different brand-market combination with at least 2 years of 
                                            historical data for incrementality analysis.
                                        """)
                                        st.stop()
                                    
                                    with col3:
                                        inc_spend = st.number_input(
                                            "Additional Investment",
                                            min_value=0,
                                            max_value=10000000,
                                            value=100000,
                                            step=10000,
                                            key="incrementality_spend"
                                        )

                                    if st.button("Calculate Incrementality", key="calc_incrementality"):
                                        try:
                                            # Get current spend
                                            current_spend = data[
                                                (data['Brand'] == inc_brand) &
                                                (data['Market'] == inc_market)
                                            ]['Spend'].iloc[-1]
                                            
                                            # Calculate incrementality
                                            inc_results = scenario_analyzer.calculate_incrementality(
                                                inc_brand, inc_market, current_spend, inc_spend
                                            )
                                            
                                            # Display results
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.subheader("Current Position")
                                                metrics_df = pd.DataFrame({
                                                    'Metric': ['Spend', 'Share of Voice', 'Share of Market', 'ESOV'],
                                                    'Value': [
                                                        f"{inc_results['current_metrics']['spend']:,.0f}",
                                                        f"{inc_results['current_metrics']['sov']:.1f}%",
                                                        f"{inc_results['current_metrics']['som']:.1f}%",
                                                        f"{inc_results['current_metrics']['esov']:.1f}%"
                                                    ]
                                                })
                                                st.table(metrics_df)
                                            
                                            with col2:
                                                st.subheader("Expected Position")
                                                new_metrics_df = pd.DataFrame({
                                                    'Metric': ['Spend', 'Share of Voice', 'Share of Market', 'ESOV'],
                                                    'Value': [
                                                        f"{inc_results['new_metrics']['spend']:,.0f}",
                                                        f"{inc_results['new_metrics']['sov']:.1f}%",
                                                        f"{inc_results['new_metrics']['som']:.1f}%",
                                                        f"{inc_results['new_metrics']['esov']:.1f}%"
                                                    ]
                                                })
                                                st.table(new_metrics_df)
                                            
                                            # Show incrementality details
                                            st.subheader("Expected Gains")
                                            inc_metrics = pd.DataFrame({
                                                'Metric': [
                                                    'Share of Market Gain',
                                                    'ESOV Improvement',
                                                    'Response Rate',
                                                    'Headroom'
                                                ],
                                                'Value': [
                                                    f"{inc_results['incrementality']['expected_share_gain']:.2f}%",
                                                    f"{inc_results['incrementality']['esov_improvement']:.1f}%",
                                                    f"{inc_results['incrementality']['response_rate']:.2f}%",
                                                    f"{inc_results['incrementality']['market_saturation']:.2f}"
                                                ]
                                            })
                                            st.table(inc_metrics)
                                            
                                            # Add explanation of metrics
                                            with st.expander("Understanding the Analysis"):
                                                st.markdown("""
                                                    ### How the Incrementality Analysis Works

                                                    This analysis uses historical data to predict the impact of additional investment. The predictions are based on:

                                                    1. **Response Rate**: 
                                                    - Calculated from historical relationship between ESOV changes and market share changes
                                                    - (∆ market share / ∆ ESOV) = how much brand gains for each unit increase in ESOV
                                                    - Uses at least 2 years of data to establish patterns
                                                    - Falls back to market average if brand-specific data is insufficient
                                                    - Confidence level indicates reliability of the prediction:
                                                        * High: 4+ years of consistent data
                                                        * Medium: 2-3 years of data
                                                        * Low: Using market average or conservative default

                                                    2. **Market Context**:
                                                    - Headroom: How much room for growth exists
                                                        * Decreases as current share increases
                                                        * Calculated as 1 - (current share / 100)
                                                        * Recognizes that growth becomes harder with higher market share

                                                    3. **Investment Impact**:
                                                    - SOV Change: How additional spend affects Share of Voice
                                                        * New SOV = (Current Spend + Additional Spend) / Total Market Spend
                                                    - ESOV Change: The gap between SOV and current market share
                                                        * ESOV = Share of Voice - Share of Market
                                                    - Expected Share Gain:
                                                        * Calculated as: ESOV improvement × Response Rate × Headroom
                                                        * Considers both historical effectiveness and current market position

                                                    The analysis requires at least 2 years of historical data to establish response patterns and make meaningful predictions. All predictions are based on actual market behavior rather than assumptions.
                                                                                                    """)
                                        except ValueError as e:
                                            st.error(str(e))
                                        except Exception as e:
                                            st.error(f"An error occurred during analysis: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()