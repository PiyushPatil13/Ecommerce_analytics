import pandas as pd
import uuid

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Amazon Sale Report.csv\Amazon Sale Report.csv")




# ── Drop useless columns ──────────────────
df = df.drop(columns=[
    'index', 'Fulfilment', 'Sales Channel ', 'ship-service-level',
    'Style', 'SKU', 'ASIN', 'Courier Status', 'currency',
    'ship-state', 'ship-postal-code', 'ship-country',
    'promotion-ids', 'B2B', 'fulfilled-by', 'Unnamed: 22'
], errors='ignore')

# ── Rename to match your app's schema ────
df = df.rename(columns={
    'Order ID' : 'order-id',
    'Date'     : 'order-date',
    'Status'   : 'status',
    'Category' : 'product-category',
    'Qty'      : 'quantity',
    'Amount'   : 'total-value',
    'ship-city': 'city'
})

# ── Add missing columns ───────────────────
df['customer-name']  = 'Unknown'      # not in this dataset
df['customer-email'] = 'unknown@example.com'
df['price']          = df['total-value'] / df['quantity'].replace(0, 1)

# ── Clean status values to match your app ─
# Your app filters: Cancelled, Returned, Delivered, On-way
status_map = {
    'Shipped'                        : 'Delivered',
    'Shipped - Delivered to Buyer'   : 'Delivered',
    'Shipped - Returned to Seller'   : 'Returned',
    'Shipped - Returning to Seller'  : 'Returned',
    'Cancelled'                      : 'Cancelled',
    'Pending'                        : 'On-way',
    'Pending - Waiting for Pick Up'  : 'On-way',
    'Shipped - Out for Delivery'     : 'On-way',
    'Shipped - Picked Up'            : 'On-way',
}
df['status'] = df['status'].map(status_map).fillna('On-way')

# ── Clean data ────────────────────────────
df['order-date']   = pd.to_datetime(df['order-date'], errors='coerce')
df['quantity']     = pd.to_numeric(df['quantity'],   errors='coerce').fillna(1).astype(int)
df['total-value']  = pd.to_numeric(df['total-value'],errors='coerce').fillna(0)
df['price']        = pd.to_numeric(df['price'],      errors='coerce').fillna(0)
df['city']         = df['city'].fillna('Unknown').str.title()

# Drop rows with missing critical fields
df = df.dropna(subset=['order-date', 'total-value'])
df = df[df['total-value'] > 0]

# ── Reorder columns to match your app ─────
df = df[[
    'order-id', 'customer-name', 'customer-email',
    'order-date', 'product-category', 'price',
    'quantity', 'total-value', 'status', 'city'
]]

print(f"✅ Cleaned dataset: {df.shape}")
print(f"   Date range : {df['order-date'].min().date()} → {df['order-date'].max().date()}")
print(f"   Categories : {df['product-category'].unique()}")
print(f"   Status dist: {df['status'].value_counts().to_dict()}")


# ── Map all to Clothing first ─────────────
category_map = {
    'Set'          : 'Clothing',
    'kurta'        : 'Clothing',
    'Western Dress': 'Clothing',
    'Top'          : 'Clothing',
    'Ethnic Dress' : 'Clothing',
    'Bottom'       : 'Clothing',
    'Saree'        : 'Clothing',
    'Blouse'       : 'Clothing',
    'Dupatta'      : 'Clothing',
}
df['product-category'] = df['product-category'].map(category_map).fillna('Clothing')

# ── Reassign categories based on price ────
def assign_category(row):
    price    = row['total-value']
    quantity = row['quantity']

    if price > 1500:
        return 'Electronics'
    elif price > 800:
        return 'Decorative'
    elif quantity >= 2:
        return 'Grocery'      # bulk orders = grocery behaviour
    else:
        return 'Clothing'

df['product-category'] = df.apply(assign_category, axis=1)

print("📊 Category distribution:")
print(df['product-category'].value_counts(normalize=True).mul(100).round(1))
print(f"\n💰 Price ranges per category:")
print(df.groupby('product-category')['total-value'].agg(['min','mean','max']).round(2))

df.to_csv("Sample7_data.csv", index=False)
print("✅ Saved → Sample4_data.csv")