import pandas as pd
from faker import Faker
import random
import numpy as np
from datetime import datetime, date, timedelta

faker = Faker()
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
ROWS       = 65000
START_DATE = date.today() - timedelta(days=365)
END_DATE   = date.today()

products = {
    'Electronics': [20,  800],
    'Grocery':     [10,   80],
    'Clothing':    [30,  300],
    'Decorative':  [10,  500]
}

categories = list(products.keys())

base_weights = {
    'Electronics': 0.30,
    'Grocery':     0.35,
    'Clothing':    0.25,
    'Decorative':  0.10
}


def get_seasonal_weight(category, date):
    month = date.month
    w = base_weights[category]

    if category == 'Electronics':
        if month in [11, 12]:   w *= 1.6
        elif month in [1, 2]:   w *= 0.8

    elif category == 'Grocery':
        if month in [11, 12]:   w *= 0.85
        elif month in [6, 7, 8]: w *= 1.1

    elif category == 'Clothing':
        if month in [3, 4]:     w *= 1.4
        elif month in [10, 11]: w *= 1.3
        elif month in [6, 7]:   w *= 0.75

    elif category == 'Decorative':
        if month in [10, 12]:   w *= 1.8
        elif month in [1, 6]:   w *= 0.6

    return w


def pick_category(date):
    weights = [get_seasonal_weight(c, date) for c in categories]
    return random.choices(categories, weights=weights, k=1)[0]


def get_quantity(category, date):
    is_weekend = date.weekday() >= 5
    base_qty = random.randint(1, 5)
    if category == 'Grocery':
        base_qty = random.randint(2, 8)
    if is_weekend:
        base_qty = min(base_qty + random.randint(0, 2), 10)
    return base_qty


def get_status(date):
    days_ago = (date.today() - date).days
    if days_ago < 7:
        return random.choices(["Delivered", "On-way", "Cancelled"],
                               weights=[0.3, 0.6, 0.1])[0]
    elif days_ago < 30:
        return random.choices(["Delivered", "On-way", "Cancelled", "Returned"],
                               weights=[0.6, 0.2, 0.1, 0.1])[0]
    else:
        return random.choices(["Delivered", "Cancelled", "Returned"],
                               weights=[0.75, 0.15, 0.10])[0]


def generate_date():
    while True:
        date = faker.date_between(start_date=START_DATE, end_date=END_DATE)
        month_boost = 1.0
        if date.month in [10, 11, 12]:   month_boost = 1.8
        elif date.month in [6, 7]:        month_boost = 0.7
        weekend_boost = 1.4 if date.weekday() >= 5 else 1.0
        if random.random() < (month_boost * weekend_boost) / (1.8 * 1.4):
            return date


# ─────────────────────────────────────────
# GENERATE DATA
# ─────────────────────────────────────────
print("⏳ Generating data...")

data = []
for i in range(ROWS):
    date     = generate_date()
    category = pick_category(date)
    price    = round(random.uniform(*products[category]), 2)
    quantity = get_quantity(category, date)

    data.append({
        "order-id":         faker.uuid4(),
        "customer-name":    faker.name(),
        "customer-email":   faker.email(),
        "order-date":       date,
        "product-category": category,
        "price":            price,
        "quantity":         quantity,
        "total-value":      round(price * quantity, 2),
        "status":           get_status(date),
        "city":             faker.city()
    })

    if (i + 1) % 10000 == 0:
        print(f"  ✅ {i+1:,} rows generated...")

df = pd.DataFrame(data)
df = df.sort_values("order-date").reset_index(drop=True)
df.to_csv("Sample6_data.csv", index=False)

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n✅ Data saved to Sample4_data.csv")
print(f"\n📊 Category Distribution:")
dist = df['product-category'].value_counts(normalize=True).mul(100).round(1)
for cat, pct in dist.items():
    bar = "█" * int(pct / 2.5)
    print(f"  {cat:<15} {pct}%  {bar}")

print(f"\n📅 Date Range   : {df['order-date'].min()} → {df['order-date'].max()}")
print(f"📦 Total Orders : {len(df):,}")
print(f"💰 Total Revenue: ${df['total-value'].sum():,.2f}")
print(f"📈 Avg Order Val: ${df['total-value'].mean():.2f}")

print(f"\n📋 Status Breakdown:")
print(df['status'].value_counts(normalize=True).mul(100).round(1).to_string())

print(f"\n🔍 Sample rows:")
print(df[["order-date","product-category","price",
          "quantity","total-value","status"]].head(8).to_string(index=False))