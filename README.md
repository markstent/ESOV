# ESOV Analysis Dashboard

## Overview
The ESOV (Excess Share of Voice) Analysis Dashboard is a comprehensive analytical tool designed to help businesses measure and optimize their marketing and advertising investments. By analyzing Share of Voice (SOV) and Share of Market (SOM), the dashboard provides insights into competitive positioning, efficiency of advertising spend, and the potential impact of budget adjustments.

## What is ESOV?
**ESOV (Excess Share of Voice)** is a key marketing metric that represents the difference between a brand’s Share of Voice (SOV) and its Share of Market (SOM):

\[ ESOV = SOV - SOM \]

- **Share of Voice (SOV)**: The percentage of total category advertising spending a brand controls.
- **Share of Market (SOM)**: The percentage of total category sales a brand owns.

A **positive ESOV** (SOV > SOM) typically indicates that a brand is investing more in advertising relative to its market share, which may lead to future growth. Conversely, a **negative ESOV** (SOV < SOM) suggests that a brand is underinvesting compared to its market position, potentially risking a decline in market share.

## Why is ESOV Important?
1. **Predicts Market Share Growth**: Research suggests that higher ESOV often correlates with future market share gains.
2. **Guides Strategic Investment**: Identifies whether a brand is over-investing, under-investing, or optimally spending on advertising.
3. **Enhances Budget Allocation**: Helps brands allocate marketing budgets more effectively across different markets and brands.
4. **Benchmarks Competitive Positioning**: Provides insights into how a brand compares to competitors in terms of advertising presence.

## Features of the ESOV Analysis Dashboard
### 1. **Data Upload and Processing**
- Accepts CSV and Excel files with advertising and market share data.
- Automatically processes and calculates key metrics (ESOV, SOV, SOM, ESOV Efficiency, etc.).

### 2. **Market and Brand Analysis**
- **ESOV Trends Over Time**: Tracks the evolution of ESOV for different brands and markets.
- **SOV vs. SOM Analysis**: Compares advertising investment with market position.
- **ESOV Efficiency Metrics**: Measures how effectively advertising spend translates into market share gains.
- **Competitive Benchmarking**: Identifies leading and underperforming brands in the market.

### 3. **Scenario Analysis and Budget Optimization**
- **Incrementality Analysis**: Predicts the impact of additional marketing investment on market share growth.
- **Budget Allocation Optimization**: Provides data-driven recommendations for distributing marketing budgets across different markets for maximum impact.

### 4. **Comprehensive Insights and Recommendations**
- Generates automated insights and recommendations based on data analysis.
- Helps marketers identify growth opportunities and refine investment strategies.

### 5. **Data Export and Reporting**
- Provides downloadable data tables and visualization reports.
- Generates a professional PDF report summarizing findings, insights, and recommended actions.

## Built with Streamlit
This dashboard is developed using **Streamlit**, a powerful Python framework for building interactive web applications for data analysis and visualization. Streamlit makes it easy to create an intuitive and responsive interface for users to explore their ESOV data in real time.

## How to Use the Dashboard
1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the Application**:
   ```sh
   streamlit run app.py
   ```
3. **Upload Your Data**: Ensure your dataset includes columns for Year, Brand, Market, Spend, and Share of Market.
4. **Filter Your Analysis**: Select specific markets, years, and brands to focus on.
5. **Explore Insights**: View ESOV trends, efficiency metrics, and market positioning.
6. **Run Scenarios**: Use budget allocation and incrementality tools to refine marketing strategies.
7. **Export Reports**: Download insights as CSV files or generate a PDF report for presentations.

## Dataset Format
Your dataset should be in CSV or Excel format with the following required columns:

| Column Name       | Description                                        | Example Value |
|------------------|------------------------------------------------|---------------|
| Year            | The year of data entry                         | 2023          |
| Brand          | The brand name or identifier                   | Brand A       |
| Market         | The market region                               | UK            |
| Share of Market | The percentage of market share                | 25.5%         |
| Spend          | The total advertising spend for the brand      | 1000000       |

A sample dataset can be downloaded [here](https://example.com/sample-dataset.csv).

## Who Can Benefit from This Application?
- **Marketing Teams**: Optimize advertising budgets and maximize return on investment (ROI).
- **Brand Managers**: Understand brand positioning and future growth opportunities.
- **Executives & Decision Makers**: Make informed strategic decisions based on market insights.
- **Agencies & Consultants**: Provide clients with data-driven recommendations and reports.

## Conclusion
The ESOV Analysis Dashboard is a powerful tool for businesses looking to make data-driven marketing decisions. By understanding and optimizing ESOV, brands can ensure they are investing the right amount in advertising to drive sustainable market share growth. This application simplifies the complex task of analyzing market spend data, allowing brands to stay competitive and strategically allocate their marketing resources.
