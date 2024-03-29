from official.nlp import optimization  # to create AdamW optimizer

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

import common
from common import cat1, mapping, demapping
import time
from keras.callbacks import ModelCheckpoint

tf.get_logger().setLevel('ERROR')

print("*" * 50, demapping)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Define your model
epochs = 1
batch_size=32
num_classes = len(mapping) + 1

# Choose a BERT model to fine-tune
# 
bert_model_name="distilbert"
# bert_model_name='albert_en_base'
# bert_model_name='zh-l-12-h-768-a-12'
# bert_model_name = 'bert_multi_cased_L-12_H-768_A-12'   # bert_model_name = 'bert_en_uncased_L-4_H-512_A-8'


# create my own dataset
def load_jsons_in_folder(folder_path):
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

    df_list = []

    for file in json_files:
        file_path = os.path.join(folder_path, file)
        df=pd.read_json(file_path, lines=True)

        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)  # concatenate all dataframes into one



def preprocess_training_data(num_classes, load_jsons_in_folder, folder_path):
	df = load_jsons_in_folder(folder_path)

	# file_path="/data/fred/searching_category_predict/data/20240305/demo.json"
	# df = pd.read_json(file_path, lines=True)

	# len(cat1) 为默认的数据
	df['label'] = df.cat1.apply(lambda x: mapping.get(x, len(cat1)))
	df['sentence'] = df.apply(lambda row: row['product_name'] + ' ' + row['brand_name'], axis=1)
	print("-" * 50)
	print(df.head(50))

	X = df['sentence']
	y = df['label']
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.01, random_state = 0)
	
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)
	print("y_test check:", y_test)
	return X_train,X_test,y_train,y_test


def check_bert(tfhub_handle_encoder, tfhub_handle_preprocess):
    # Let's try the preprocessing model on some text and see the output:
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
	
    text_test = ['阔腿裤女夏季薄款高腰垂感2023新款冰丝窄版直筒春秋小个子西装裤']
    text_preprocessed = bert_preprocess_model(text_test)

    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    # print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}') # distilbert 会报错

    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    bert_results = bert_model(text_preprocessed)

    print(f'Loaded BERT: {tfhub_handle_encoder}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
    return text_test


def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder, X_train):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess,name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dense(128,activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(51,activation='softmax')(net)

    model = tf.keras.models.Model(inputs = [text_input], outputs = [net])

    steps_per_epoch=len(X_train)/batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy','AUC']
                            )
    return model


def plot_training_history(history):
    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("sample.png")
    
  
def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i]} : top3 labels: {[demapping.get(itm, "unk") for itm in np.argsort(results[i])[::-1][:3]]}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')

  
def check_model_quality(classifier_model, reloaded_model,  examples):
    reloaded_results = reloaded_model(tf.constant(examples))
    original_results = classifier_model(tf.constant(examples))

    idxs_reloaded = np.argsort(reloaded_results)[::-1][:3]
    idxs_original = np.argsort(original_results)[::-1][:3]

    print("saved model:", idxs_reloaded)
    print("model in memory:", idxs_original)

    print('Results from the saved model:')
    print_my_examples(examples, reloaded_results)
    print('Results from the model in memory:')
    print_my_examples(examples, original_results)

	# tf-serving
    serving_results = reloaded_model \
            .signatures['serving_default'](tf.constant(examples))

	# serving_results = tf.sigmoid(serving_results['dense_1'])
    serving_results=tf.nn.softmax(serving_results['dense_1'])
    # print_my_examples(examples, serving_results)
    

weights_path="model_weights"
import os
ws=os.listdir(weights_path)
ws = [itm for itm in ws if itm.startswith("weights") and itm.endswith("keras")]
use_weight=False

if len(ws) > 0:
    use_weight=True
    weights_file_name = max(ws)
    weights_file = f"{weights_path}/{weights_file_name}"
    
if __name__=="__main__":   
    folder_path=sys.argv[1]

    X_train, X_test, y_train, y_test = preprocess_training_data(num_classes, load_jsons_in_folder, folder_path)


    tfhub_handle_encoder = common.map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = common.map_model_to_preprocess[bert_model_name]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

    
    text_test = check_bert(tfhub_handle_encoder, tfhub_handle_preprocess)


    print("list_local_devices",device_lib.list_local_devices())

    classifier_model = build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder, X_train)
        
    
    if use_weight:
        classifier_model.load_weights(weights_file)
        print("load weight success", weights_file)

    # tf.keras.utils.plot_model(classifier_model)
    
    # 检查初始的模型结果
    bert_raw_result = classifier_model(tf.constant(text_test))

    # print("model result", bert_raw_result)
    # todo 使用timestamp 作为名字

    ts = int(time.time())
    checkpoint = ModelCheckpoint(f'{weights_path}/weights_{ts}.keras', monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    history = classifier_model.fit(X_train,
                                y_train,
                                batch_size,
                                validation_data=(X_test, y_test),
                                epochs=epochs,
                                callbacks=[checkpoint])

    # plot_training_history(history)

    examples = [
        '金霸王碱性电池 AAA小干电池1.5V鼠标号8粒+7号12粒',  # this is the same sentence tried earlier
        '好媳妇（okaywife）垃圾桶家用厨房垃圾桶',
        '【售罄不补】杞里香金丝皇菊20g（约50朵）*3罐-XB',
        'OAVE蜜桃臀贴OA33T08',
        '【底价清仓】资生堂肌源焕活精萃水150ml(滋润型)'
    ]
    
    # if use_weight:
    # classifier_model.save_weights(weights_path, overwrite=True, save_format='tf')
    # print("save weight success")

    saved_model_path = '{}/{}'.format("model", int(time.time()))
    classifier_model.save(saved_model_path, include_optimizer=False)

    reloaded_model = tf.saved_model.load(saved_model_path)


    # check_model_quality(classifier_model, reloaded_model,  examples)