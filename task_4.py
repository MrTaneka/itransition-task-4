import base64
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import networkx as nx
from jinja2 import Template

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

DATA_DIRS = ['DATA1', 'DATA2', 'DATA3']
FILES = {
    'users': 'users.csv',
    'orders': 'orders.parquet',
    'books': 'books.yaml'
}
OUTPUT_FILE = 'index.html'

plt.style.use('ggplot')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Task 4 Analytics</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; }
        .tabs { display: flex; border-bottom: 2px solid #ecf0f1; margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; cursor: pointer; background: none; border: none; font-size: 16px; color: #7f8c8d; font-weight: bold; transition: 0.3s; }
        .tab-btn:hover { color: #3498db; }
        .tab-btn.active { color: #3498db; border-bottom: 3px solid #3498db; }
        .content { display: none; animation: fadeIn 0.5s; }
        .content.active { display: block; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; }
        .card h3 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; color: #95a5a6; }
        .card p, .card ul { font-size: 18px; font-weight: bold; color: #2c3e50; margin: 0; padding: 0; }
        ul { list-style: none; }
        li { margin-bottom: 5px; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; background: white; }
        .red-border { border-left-color: #e74c3c; }
        .ids-text { font-size: 14px; word-break: break-all; font-family: monospace; color: #555; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sales Analytics Dashboard</h1>

        <div class="tabs">
            {% for res in results %}
            <button class="tab-btn {{ 'active' if loop.first }}" onclick="openTab('{{ res.folder }}', this)">
                {{ res.folder }}
            </button>
            {% endfor %}
        </div>

        {% for res in results %}
        <div id="{{ res.folder }}" class="content {{ 'active' if loop.first }}">
            <div class="grid">
                <div class="card"><h3>Unique Users</h3><p>{{ res.uniq_u }}</p></div>
                <div class="card"><h3>Unique Author Sets</h3><p>{{ res.uniq_a }}</p></div>
                <div class="card" style="grid-column: span 2"><h3>Most Popular Author</h3><p>{{ res.pop_a }}</p></div>
                <div class="card">
                    <h3>Top 5 Days (Revenue)</h3>
                    <ul>
                        {% for day in res.top_5 %}<li>{{ day }}</li>{% endfor %}
                    </ul>
                </div>
                <div class="card">
                    <h3>Best Buyer (Aliases IDs)</h3>
                    <p class="ids-text">{{ res.best_ids_str }}</p>
                </div>
            </div>
            <div class="card red-border" style="margin-top: 20px;">
                <h3>Revenue Chart</h3>
                <img src="data:image/png;base64,{{ res.chart }}" alt="Chart not generated">
            </div>
        </div>
        {% endfor %}
    </div>

    <script>
        function openTab(tabId, btn) {
            document.querySelectorAll('.content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }
    </script>
</body>
</html>
"""


def clean_price(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    s_val = str(value).strip()
    multiplier = 1.2 if '€' in s_val else 1.0
    clean_val = re.sub(r'[^\d.]', '', s_val)
    return float(clean_val) * multiplier if clean_val else 0.0


def clean_date_string(date_str: Any) -> str:
    if pd.isna(date_str):
        return str(date_str)
    return str(date_str).replace("A.M.", "AM").replace("P.M.", "PM").replace(".M.", "M").strip()


def generate_plot(series: pd.Series, title: str) -> str:
    if series.empty:
        return ""

    plt.figure(figsize=(10, 4))
    series.sort_index().plot(kind='line', color='#e74c3c', linewidth=2)
    plt.title(title)
    plt.ylabel("USD")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    data = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return data


def reconciliate_users(users_df: pd.DataFrame) -> Tuple[int, Dict, Dict]:
    G = nx.Graph()
    valid_users = users_df.dropna(subset=['id'])
    G.add_nodes_from(valid_users['id'].unique())

    lookup = {'email': {}, 'phone': {}, 'address': {}}

    for _, row in valid_users.iterrows():
        uid = row['id']
        props = [
            ('email', str(row['email']).lower().strip() if pd.notna(row['email']) else None),
            ('phone', re.sub(r'\D', '', str(row['phone'])) if pd.notna(row['phone']) else None),
            ('address', str(row['address']).lower().strip()[:15] if pd.notna(row['address']) else None)
        ]
        for key, val in props:
            if val and len(val) > 4:
                lookup[key].setdefault(val, []).append(uid)

    for cat in lookup.values():
        for ids in cat.values():
            if len(ids) > 1:
                nx.add_path(G, ids)
    comps = list(nx.connected_components(G))
    id_map = {uid: list(c)[0] for c in comps for uid in c}
    groups = {list(c)[0]: list(c) for c in comps}
    return len(comps), id_map, groups


def process_dataset(folder_path: Path) -> Optional[Dict]:
    folder_name = folder_path.name
    users = pd.read_csv(folder_path / FILES['users'])
    orders = pd.read_parquet(folder_path / FILES['orders'])
    with open(folder_path / FILES['books'], 'r', encoding='utf-8') as f:
        books = pd.DataFrame(yaml.safe_load(f))

    books.columns = [c.replace(':', '') if isinstance(c, str) else c for c in books.columns]

    cols = {
        'date': next(c for c in orders.columns if 'date' in c.lower() or 'timestamp' in c.lower()),
        'price': next(c for c in orders.columns if 'price' in c.lower()),
        'qty': next(c for c in orders.columns if 'qty' in c.lower() or 'quantity' in c.lower()),
        'item_id': next(c for c in orders.columns if 'item' in c or 'book' in c),
        'user_id': next(c for c in orders.columns if 'user' in c),
        'book_id': next(c for c in books.columns if 'id' in c)
    }

    orders['dt'] = pd.to_datetime(
        orders[cols['date']].apply(clean_date_string),
        errors='coerce',
        utc=True
    )
    orders = orders.dropna(subset=['dt'])
    orders['date_str'] = orders['dt'].dt.strftime('%Y-%m-%d')

    orders['paid'] = orders[cols['qty']].fillna(0) * orders[cols['price']].apply(clean_price)
    daily_revenue = orders.groupby('date_str')['paid'].sum().sort_values(ascending=False)
    uniq_users_count, user_map, user_groups = reconciliate_users(users)
    orders['join_id'] = orders[cols['item_id']].astype(str).str.replace(r'\.0$', '', regex=True)
    books['join_id'] = books[cols['book_id']].astype(str).str.replace(r'\.0$', '', regex=True)
    merged = orders.merge(books, on='join_id', how='left')
    pop_author = "Unknown"
    uniq_authors = 0
    if 'author' in books.columns:
        books['author_clean'] = books['author'].astype(str).str.strip()
        uniq_authors = books['author_clean'].nunique()

        if not merged.empty:
            merged['author_clean'] = merged['author'].astype(str).str.strip()
            top_authors = merged[merged['author_clean'] != 'nan'].groupby('author_clean')[cols['qty']].sum()
            if not top_authors.empty:
                pop_author = top_authors.idxmax()

    orders['real_uid'] = orders[cols['user_id']].map(user_map).fillna(orders[cols['user_id']])
    top_buyer_id = orders.groupby('real_uid')['paid'].sum().idxmax()
    best_buyer_aliases = user_groups.get(top_buyer_id, [top_buyer_id])
    best_buyer_str = ", ".join(map(str, best_buyer_aliases))
    chart_b64 = generate_plot(daily_revenue, f'Revenue Trend: {folder_name}')

    return {
        'folder': folder_name,
        'top_5': daily_revenue.head(5).index.tolist(),
        'uniq_u': uniq_users_count,
        'uniq_a': uniq_authors,
        'pop_a': pop_author,
        'best_ids_str': best_buyer_str,  # Отправляем готовую строку
        'chart': chart_b64
    }


def main():
    root = Path('.')
    results = []
    for folder_name in DATA_DIRS:
        folder_path = root / folder_name
        data = process_dataset(folder_path)
        if data:
            results.append(data)

    rendered_html = Template(HTML_TEMPLATE).render(results=results)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

if __name__ == "__main__":
    main()