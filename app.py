import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
import random
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Only create ONE FastAPI instance
app = FastAPI()

# Add CORS middleware configuration - THIS MUST COME BEFORE ANY ROUTES
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Generate rich synthetic data
def generate_synthetic_data():
    # Create 6 months of sales data
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product details
    products = [
        {"id": f"prod-{i}", "name": name, "category": category, "cost_price": round(random.uniform(10, 100), 2)}
        for i, (name, category) in enumerate([
            ("Organic Milk", "Dairy"),
            ("Free-range Eggs", "Dairy"),
            ("Artisan Bread", "Bakery"),
            ("Basmati Rice", "Grains"),
            ("Black Beans", "Grains"),
            ("Handmade Soap", "Toiletries"),
            ("Natural Shampoo", "Toiletries"),
            ("Honeycrisp Apples", "Fruits"),
            ("Organic Bananas", "Fruits"),
            ("Free-range Chicken", "Meat"),
        ])
    ]
    
    # Generate sales data
    sales = []
    for date in date_range:
        # Weekends have more sales
        is_weekend = date.weekday() in [5, 6]
        num_sales = random.randint(20 if is_weekend else 10, 40 if is_weekend else 25)
        
        for _ in range(num_sales):
            product = random.choice(products)
            quantity = random.randint(1, 5)
            unit_price = round(product['cost_price'] * random.uniform(1.3, 2.0), 2)
            total_price = round(unit_price * quantity, 2)
            
            sales.append({
                "sale_id": f"sale-{len(sales)}",
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "item_id": product['id'],
                "item_name": product['name'],
                "category": product['category'],
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": total_price
            })
    
    # Generate inventory data
    inventory = []
    for product in products:
        low_stock_threshold = random.randint(5, 15)
        current_quantity = random.randint(0, 50)
        
        inventory.append({
            "id": product['id'],
            "item_name": product['name'],
            "category": product['category'],
            "quantity": current_quantity,
            "low_stock_threshold": low_stock_threshold,
            "unit_price": product['cost_price']
        })
    
    # Generate expenses data
    expense_categories = ['Rent', 'Utilities', 'Transport', 'Marketing', 'Supplies', 'Salaries', 'Maintenance']
    expenses = []
    for i in range(100):
        date = start_date + timedelta(days=random.randint(0, 180))
        category = random.choice(expense_categories)
        
        # Different expense categories have different typical amounts
        if category == 'Rent':
            amount = round(random.uniform(2000, 5000), 2)
        elif category == 'Salaries':
            amount = round(random.uniform(1500, 4000), 2)
        else:
            amount = round(random.uniform(50, 800), 2)
            
        expenses.append({
            "id": f"exp-{i}",
            "category": category,
            "amount": amount,
            "date": date.strftime("%Y-%m-%d")
        })
    
    return sales, expenses, inventory

# Perform advanced analytics
def perform_analytics(sales, expenses, inventory):
    # Convert to DataFrames
    sales_df = pd.DataFrame(sales)
    expenses_df = pd.DataFrame(expenses)
    inventory_df = pd.DataFrame(inventory)
    
    # Process dates
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])
    
    # Last 30 days analysis
    today = datetime.now()
    last_30_days = today - timedelta(days=30)
    
    # Top selling products
    recent_sales = sales_df[sales_df['date'] >= last_30_days]
    top_selling = recent_sales.groupby('item_name')['quantity'].sum().nlargest(5).reset_index()
    
    # Low stock alerts
    low_stock = inventory_df[inventory_df['quantity'] < inventory_df['low_stock_threshold']]
    
    # Profitability analysis
    merged_df = sales_df.merge(
        inventory_df[['id', 'unit_price']], 
        left_on='item_id', 
        right_on='id', 
        suffixes=('_sale', '_cost')
    )
    merged_df['profit_per_unit'] = merged_df['unit_price_sale'] - merged_df['unit_price_cost']
    merged_df['total_profit'] = merged_df['profit_per_unit'] * merged_df['quantity']
    
    profitable_items = merged_df.groupby('item_name')['total_profit'].sum().nlargest(5).reset_index()
    profitable_items = profitable_items.rename(columns={'total_profit': 'profit'})
    
    # Daily sales trend
    daily_sales = sales_df.set_index('date').resample('D')['total_price'].sum().reset_index()
    daily_sales = daily_sales.rename(columns={'total_price': 'total_amount'})
    
    # Expense breakdown
    expense_breakdown = expenses_df.groupby('category')['amount'].sum().reset_index()
    
    # Financial metrics
    total_sales = recent_sales['total_price'].sum()
    total_expenses = expenses_df[expenses_df['date'] >= last_30_days]['amount'].sum()
    net_profit = total_sales - total_expenses
    profit_margin = (net_profit / total_sales) * 100 if total_sales > 0 else 0
    
    # Slow-moving inventory
    sold_items = recent_sales['item_id'].unique()
    slow_moving = inventory_df[~inventory_df['id'].isin(sold_items)]
    
    # Association rules (Apriori)
    try:
        basket = recent_sales.groupby(['sale_id', 'item_name'])['quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1)
        rules = rules.sort_values('confidence', ascending=False).head(5)
        associations = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    except Exception as e:
        associations = pd.DataFrame()
    
    # Sales forecasting
    try:
        ts_data = daily_sales.set_index('date')['total_amount']
        model = ARIMA(ts_data, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        forecast = [round(x, 2) for x in forecast]
    except Exception as e:
        forecast = []
    
    # Customer trends
    customer_sales = sales_df.groupby('sale_id')['date'].first().reset_index()
    customer_sales['customer_type'] = np.where(
        customer_sales['date'] < last_30_days, 
        'Returning', 
        'New'
    )
    customer_trends = customer_sales['customer_type'].value_counts().reset_index()
    customer_trends.columns = ['type', 'count']
    
    return {
        "top_selling": top_selling.to_dict('records'),
        "low_stock": low_stock.to_dict('records'),
        "profitable_items": profitable_items.to_dict('records'),
        "daily_sales": daily_sales.to_dict('records'),
        "expense_breakdown": expense_breakdown.to_dict('records'),
        "customer_trends": customer_trends.to_dict('records'),
        "financial_metrics": {
            "total_sales": total_sales,
            "total_expenses": total_expenses,
            "net_profit": net_profit,
            "profit_margin": profit_margin
        },
        "slow_moving": slow_moving.to_dict('records'),
        "associations": associations.to_dict('records'),
        "forecast": forecast
    }

@app.get("/api/analytics")
def get_analytics():
    try:
        sales, expenses, inventory = generate_synthetic_data()
        return perform_analytics(sales, expenses, inventory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)