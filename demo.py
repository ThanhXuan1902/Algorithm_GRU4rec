import numpy as np
import pandas as pd

num_users = 1000
num_items = 1000
max_items_per_user = 10

# Tạo ngẫu nhiên dữ liệu cho user_id, item_id và timestamp
user_id = np.random.randint(0, num_users, size=num_users*max_items_per_user)
item_id = np.random.randint(0, num_items, size=num_users*max_items_per_user)
timestamp = np.abs(np.random.normal(size=num_users*max_items_per_user))

# Đưa dữ liệu vào DataFrame
data = pd.DataFrame({'user_id': user_id, 'item_id': item_id, 'timestamp': timestamp})

# Sắp xếp dữ liệu theo thứ tự thời gian
data.sort_values(by='timestamp', inplace=True)

# Lưu dữ liệu vào tệp CSV
data.to_csv('data.csv', index=False)