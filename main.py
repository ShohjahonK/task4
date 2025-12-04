from fileinput import filename
import pandas as pd
import yaml
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from dateutil import parser


def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).lower() == 'nan':
        return pd.NaT

    s = str(date_str)

    s = re.sub(r'\bM\b', '', s)
    s = re.sub(r'(?i)[o0]c', 'Oct', s)
    s = re.sub(r'(?i)Octtober', 'Oct', s)
    s = re.sub(r'(?i)Octt', 'Oct', s)
    s = re.sub(r'(?i)P\.\.', 'PM', s)
    s = re.sub(r'(?i)A\.\.', 'AM', s)
    s = s.replace(' -', '-')
    s = re.sub(r'[^a-zA-Z0-9:\-\/\.]', ' ', s)

    s = re.sub(r'(?i)\bA\s*M\b', 'AM', s)
    s = re.sub(r'(?i)\bP\s*M\b', 'PM', s)
    s = re.sub(r'\s+', ' ', s).strip()

    time_first_pattern = r'^(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s+(.*)'
    match = re.match(time_first_pattern, s, re.IGNORECASE)
    if match:
        s = f"{match.group(2)} {match.group(1)}"

    try:
        dt = parser.parse(s, dayfirst=False, fuzzy=True)
        return dt.replace(tzinfo=None)
    except:
        return pd.NaT


def clean_currency(price_str):
    if pd.isna(price_str): return 0.0
    s = str(price_str).strip()

    s = s.replace('¢', '.').replace(',', '.')

    num_str = re.sub(r'[^\d\.]', '', s)
    try:
        val = float(num_str)
    except:
        return 0.0

    if '€' in s or 'EUR' in s:
        return val * 1.2
    else:
        return val


def clean_phone(phone_str):
    if pd.isna(phone_str): return ""
    return re.sub(r'\D', '', str(phone_str))


def resolve_user_identities(users_df):
    G = nx.Graph()

    users_df['temp_clean_phone'] = users_df['phone'].apply(clean_phone)

    for _, row in users_df.iterrows():
        uid = row['id']
        phone_node = f"PHONE_{row['temp_clean_phone']}"
        email_node = f"EMAIL_{str(row['email']).lower().strip()}"

        G.add_edge(uid, phone_node)
        G.add_edge(uid, email_node)

    connected_components = list(nx.connected_components(G))
    user_mapping = {}

    for cluster in connected_components:
        real_ids = [n for n in cluster if
                    isinstance(n, int) or (isinstance(n, str) and not n.startswith(('PHONE_', 'EMAIL_')))]

        if real_ids:
            master_id = sorted(real_ids)[0]
            for rid in real_ids:
                user_mapping[rid] = master_id

    users_df.drop(columns=['temp_clean_phone'], inplace=True)

    return user_mapping


def extract(folder_path):
    users = pd.read_csv(os.path.join(folder_path, 'users.csv'))
    orders = pd.read_parquet(os.path.join(folder_path, 'orders.parquet'))

    with open(os.path.join(folder_path, 'books.yaml'), 'r') as f:
        books_data = yaml.safe_load(f)
    books = pd.json_normalize(books_data)

    return users, orders, books


def transform(users, orders, books):
    if 'timestamp' in orders.columns:
        orders['timestamp_clean'] = orders['timestamp'].astype(str).apply(parse_date)
        orders['date'] = orders['timestamp_clean'].dt.date

    orders['unit_price_clean'] = orders['unit_price'].apply(clean_currency)
    orders['paid_price'] = orders['quantity'] * orders['unit_price_clean']

    author_col = ':author' if ':author' in books.columns else 'authors'
    if author_col in books.columns:
        books['author_set'] = books[author_col].apply(
            lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,)
        )

    id_map = resolve_user_identities(users)
    users['real_user_id'] = users['id'].map(id_map)
    users['real_user_id'] = users['real_user_id'].fillna(users['id'])

    users = users.drop_duplicates()
    orders = orders.drop_duplicates()
    books = books.drop_duplicates()

    return users, orders, books


def load_metrics(users, orders, books):
    if 'user_id' in orders.columns:
        user_map = dict(zip(users['id'], users['real_user_id']))
        orders['real_user_id'] = orders['user_id'].map(user_map)

    valid_orders = orders.dropna(subset=['date'])
    daily_revenue = valid_orders.groupby('date')['paid_price'].sum().sort_values(ascending=False)

    unique_real_users = users['real_user_id'].nunique()

    unique_author_sets = books['author_set'].nunique()

    top_author_name = "N/A"
    if 'book_id' in orders.columns and ":id" in books.columns:
        merged_books = orders.merge(books, left_on='book_id', right_on=':id', how='inner')
        author_sales = merged_books.groupby('author_set')['quantity'].sum().sort_values(ascending=False)
        if not author_sales.empty:
            top_author_set = author_sales.index[0]
            top_author_name = ', '.join(top_author_set)

    best_buyer_id = "N/A"
    best_buyer_spent = 0.0
    best_buyer_aliases = []
    if 'real_user_id' in orders.columns:
        top_buyer = orders.groupby('real_user_id')['paid_price'].sum().sort_values(ascending=False)
        if not top_buyer.empty:
            best_buyer_id = top_buyer.index[0]
            best_buyer_spent = top_buyer.iloc[0]
            best_buyer_aliases = users[users['real_user_id'] == best_buyer_id]['id'].tolist()

    return {
        "daily_revenue": daily_revenue,
        "unique_users": unique_real_users,
        "unique_authors": unique_author_sets,
        "top_author": top_author_name,
        "top_buyer_id": best_buyer_id,
        "top_buyer_spent": best_buyer_spent,
        "top_buyer_aliases": best_buyer_aliases
    }


def generate_dashboard(metrics, folder_name):
    daily_revenue = metrics['daily_revenue']
    plt.figure(figsize=(10, 5))
    daily_revenue.sort_index().plot(kind='line', color='#2c3e50', linewidth=2)
    plt.title('Daily Revenue Trend', fontsize=14)
    plt.ylabel('Revenue ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    top_days_html = ""
    for date, val in daily_revenue.head(5).items():
        top_days_html += f"<tr><td>{date}</td><td>${val:,.2f}</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Sales Dashboard - {folder_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; color: #333; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
            .kpi-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
            .kpi-card h3 {{ margin: 0; font-size: 14px; color: #7f8c8d; }}
            .kpi-card p {{ margin: 10px 0 0; font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .alias-box {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; word-break: break-all; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Executive Dashboard ({folder_name})</h1>
            
            <div class="kpi-grid">
                <div class="kpi-card"><h3>Real Unique Users</h3><p>{metrics['unique_users']:,}</p></div>
                <div class="kpi-card"><h3>Unique Author Sets</h3><p>{metrics['unique_authors']:,}</p></div>
                <div class="kpi-card"><h3>Total Revenue</h3><p>${daily_revenue.sum():,.0f}</p></div>
            </div>

            <div class="kpi-grid">
                 <div class="kpi-card" style="grid-column: span 3;">
                    <h3>🏆 Top Customer (With Aliases)</h3>
                    <p>{metrics['top_buyer_id']} <span style="font-size:18px; color:#27ae60">(Spent ${metrics['top_buyer_spent']:,.2f})</span></p>
                    <div class="alias-box">Known Aliases: {metrics['top_buyer_aliases']}</div>
                </div>
            </div>
            
            <div class="kpi-card" style="margin-bottom: 30px;">
                <h3>Most Popular Author</h3>
                <p>{metrics['top_author']}</p>
            </div>

            <h2>📈 Revenue Trend</h2>
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px;">
                <img src="data:image/png;base64,{plot_url}" style="width:100%">
            </div>

            <h2>📅 Top 5 Days by Revenue</h2>
            <table>
                <tr><th>Date</th><th>Revenue</th></tr>
                {top_days_html}
            </table>
        </div>
    </body>
    </html>
    """

    filename = f"dashboard_{folder_name.split('/')[-1]}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__":
    folders = ['Data/DATA3']

    for folder in folders:
        try:
            users_raw, orders_raw, books_raw = extract(folder)

            users_clean, orders_clean, books_clean = transform(users_raw, orders_raw, books_raw)

            metrics = load_metrics(users_clean, orders_clean, books_clean)

            generate_dashboard(metrics, folder)

        except Exception as e:
            print(f"ERROR processing {folder}: {e}")
