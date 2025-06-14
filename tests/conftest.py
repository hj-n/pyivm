import pytest
import sys
import os
import json
from collections import defaultdict

# Add the parent directory to the path so we can import pyivm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def pytest_html_report_title(report):
    report.title = "Clustering Metrics Performance Report"

def pytest_html_results_table_header(cells):
    cells.insert(2, '<th class="sortable" col="score">Score Info</th>')

def pytest_html_results_table_row(report, cells):
    # Add score info to each test row
    score_info = ""
    if hasattr(report, 'sections'):
        for section in report.sections:
            if section[0] == "score_info":
                try:
                    data = json.loads(section[1])
                    score_info = f"Dataset: {data['dataset']}<br>Metric: {data['metric']}<br>Score: {data['score']:.6f}"
                except:
                    score_info = "N/A"
                break
    
    cells.insert(2, f'<td class="col-score">{score_info}</td>')

def pytest_html_results_summary(prefix, summary, postfix):
    # Collect all score results from the session
    if hasattr(pytest, '_score_results_global'):
        results = pytest._score_results_global
        
        # Create summary table
        score_table = create_scores_table(results)
        
        prefix.extend([
            '<h2>Metric Scores Summary</h2>',
            score_table
        ])

def create_scores_table(results):
    if not results:
        return '<p>No score results available</p>'
    
    # Organize results by dataset and metric
    data_matrix = defaultdict(dict)
    datasets = set()
    metrics = set()
    
    for result in results:
        dataset = result['dataset']
        metric = result['metric']
        score = result['score']
        
        datasets.add(dataset)
        metrics.add(metric)
        data_matrix[dataset][metric] = score
    
    datasets = sorted(datasets)
    metrics = sorted(metrics)
    
    # Create HTML table
    html = ['<table class="results-table" style="border-collapse: collapse; width: 100%; margin: 20px 0;">']
    
    # Header row
    html.append('<tr style="background-color: #f0f0f0;">')
    html.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Dataset</th>')
    for metric in metrics:
        html.append(f'<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">{metric}</th>')
    html.append('</tr>')
    
    # Data rows
    for dataset in datasets:
        html.append('<tr>')
        html.append(f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{dataset}</td>')
        for metric in metrics:
            score = data_matrix[dataset].get(metric, 'N/A')
            if isinstance(score, float):
                score_str = f'{score:.6f}'
            else:
                score_str = str(score)
            html.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{score_str}</td>')
        html.append('</tr>')
    
    html.append('</table>')
    
    return '\n'.join(html)

@pytest.fixture(scope="session", autouse=True)
def collect_results(request):
    # Initialize global score results collection
    if not hasattr(pytest, '_score_results_global'):
        pytest._score_results_global = []
    
    # Initialize per-config collection
    if not hasattr(request.config, '_score_results'):
        request.config._score_results = pytest._score_results_global
    
    yield