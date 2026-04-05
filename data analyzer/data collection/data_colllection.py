import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# creating object of faker

faker = Faker()

# function to create the data

def create_fake_data(rows=4000):
    data = []
    products = {
        'Electronics': [20,2000], # min price and max price
        'Grocery': [10,80],
        'Clothing' : [30,300],
        'Decorative':[10,1000]
    }

    for _ in range(rows):

        category = random.choice(list(products.keys())) # randomly assignning the key values according to product
        price = round(random.uniform(products[category][0],products[category][1]),2)
        quantity = random.randint(1,5)
         
        data.append({
            "order-id" : faker.uuid4(),
            "customer-name": faker.name(),
            "customer-email" : faker.email(),
            "order-date":faker.date_between(start_date='-1y',end_date='today'),
            "product-category":category,
            "price" : price,
            "quantity": quantity,
            "total-value":price*quantity,
            "status": random.choice(["Delivered","On-way","Cancelled","Returned"]),
            "city":faker.city()

        })

    return pd.DataFrame(data)

df = create_fake_data(60000)
df.to_csv("Train_data.csv",index=False)

