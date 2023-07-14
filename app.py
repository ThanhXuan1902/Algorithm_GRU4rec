from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.layers import Reshape, GRU
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Đọc dữ liệu
data = pd.read_csv('data.csv')
# Lấy ra các cột cần thiết
data = data[['user_id', 'item_id', 'timestamp']]
# Chuyển đổi user_id và item_id thành các số nguyên
user_encoder = LabelEncoder()
data['user_id'] = user_encoder.fit_transform(data['user_id'])
item_encoder = LabelEncoder()
data['item_id'] = item_encoder.fit_transform(data['item_id'])
# Tạo một dictionary để ánh xạ ngược lại từ số nguyên thành user_id và item_id
user_dict = dict(zip(data['user_id'], data['user_id']))
item_dict = dict(zip(data['item_id'], data['item_id']))
# Số lượng người dùng và số lượng sản phẩm
num_users = len(user_dict)
num_items = len(item_dict)

# Chia dữ liệu thành tập train và tập test
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Tạo một dictionary chứa các mục đã xem của từng người dùng trong tập train
user2item = {}
for user_id, item_id, _ in train_data.values:
    if user_id not in user2item:
        user2item[user_id] = []
    user2item[user_id].append(item_id)

# Hàm sinh batch dữ liệu
def generate_batch(user2item, batch_size, num_items):
    # Tạo một mảng chứa các user_id
    user_ids = list(user2item.keys())
    while True:
        # Lấy ngẫu nhiên một batch user_id
        batch_user_ids = np.random.choice(user_ids, size=batch_size, replace=False)
        # Khởi tạo input_sequence và output_sequence
        input_sequence = []
        output_sequence = []
        for user_id in batch_user_ids:
            # Lấy ra các mục đã xem của user_id
            viewed_items = user2item[user_id]
            # Nếu user_id không có mục nào đã xem thì bỏ qua
            if len(viewed_items) == 0:
                continue
            # Lấy ra mục cuối cùng đã xem của user_id
            last_item = viewed_items[-1]
            # Tạo input_sequence bằng cách thêm từng mục đã xem của user_id vào input_sequence
            input_sequence.append(viewed_items[:-1])
            # Tạo output_sequence bằng cách thêm mục cuối cùng đã xem của user_id vào output_sequence
            output_sequence.append(last_item)
        # Chuyển input_sequence và output_sequence thành các mảng numpy
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=None, padding='pre', truncating='pre', dtype='float32', value=num_items)
        output_sequence = tf.keras.utils.to_categorical(output_sequence, num_items+1)
        # Trả về input_sequence và output_sequence
        yield input_sequence, output_sequence

# Định nghĩa mô hình GRU4Rec
def gru4rec(num_items, hidden_units):
    input_sequence = Input(shape=(None,), dtype='float32') # chuyển đổi dtype thành float32
    # Chuyển đổi input_sequence sang dạng 3D tensor
    reshaped_input_sequence = Reshape((-1, 1))(input_sequence)
    # Khởi tạo GRU layer với số lượng hidden units cho trước
    gru_layer = GRU(hidden_units, activation='tanh', recurrent_activation='sigmoid')(reshaped_input_sequence)
    # Khởi tạo output layer với số lượng units bằng với số lượng sản phẩm
    from keras.layers import Dense
    output_layer = Dense(num_items+1, activation='softmax')(gru_layer)
    # Tạo mô hình với input là input_sequence và output là output_layer
    model = Model(inputs=input_sequence, outputs=output_layer)
    return model

# Thiết lập các thông số cho mô hình
batch_size = 64
hidden_units = 100
learning_rate = 0.001
num_epochs = 10

# Tạo mới hình GRU4Rec
model = gru4rec(num_items=num_items, hidden_units=hidden_units)

# Biên dịch mô hình với optimizer là Adam và loss function là categorical_crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')

# Huấn luyện mô hình
train_generator = generate_batch(user2item, batch_size, num_items)
model.fit(train_generator, steps_per_epoch=len(user2item)//batch_size, epochs=num_epochs)

# Đánh giá mô hình trên tập test
test_user2item = {}
for user_id, item_id, _ in test_data.values:
    if user_id not in test_user2item:
        test_user2item[user_id] = []
    test_user2item[user_id].append(item_id)
test_input_sequence = [test_user2item[user_id][:-1] for user_id in test_user2item.keys()]
test_input_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_input_sequence, maxlen=None, padding='pre', truncating='pre', dtype='float32', value=num_items) # chuyển đổi dtype thành float32
test_output_sequence = [test_user2item[user_id][-1] for user_id in test_user2item.keys()]
test_output_sequence = tf.keras.utils.to_categorical(test_output_sequence, num_items+1)
test_loss = model.evaluate(test_input_sequence, test_output_sequence, verbose=0)
print('Test loss:', test_loss)

# Tạo giao diện web
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = int(request.form['user_id'])
    viewed_items = [int(x) for x in request.form['viewed_items'].split(',')]
    input_sequence = np.array([viewed_items])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=None, padding='pre', truncating='pre', dtype='float32', value=num_items)
    output = model.predict(input_sequence)
    recommended_items = np.argsort(output[0])[::-1][:10] # Lấy ra 10 sản phẩm được đề xuất có xác suất cao nhất
    recommended_items = [item_encoder.inverse_transform([x])[0] for x in recommended_items] # Chuyển đổi lại thành item_id
    return render_template('result.html', user_id=user_encoder.inverse_transform([user_id])[0], recommended_items=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)