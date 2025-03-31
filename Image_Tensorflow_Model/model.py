import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Enable mixed precision for performance optimization
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Use multiple GPUs if available for distributed training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

def preprocess_image(image_path, img_size=(160, 160)):
        clean_path = ''.join(c for c in image_path if c.isprintable())
        img = load_img(clean_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0 # Normalize pixel values to [0,1]
        return img_array

def create_dataset(image_indices, labels, batch_size = 32, img_size=(160, 160), image_dir= 'images_folder', is_training=False):
    def process_path(image_index, label):
        image_index = tf.strings.strip(tf.strings.as_string(image_index))
        image_path = tf.strings.join([image_dir, "/", image_index])

        def load_and_preprocess_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3) # Decode JPEG image
            img = tf.image.resize(img, img_size) # Resize image
            img = img/ 255.0 # Normalize

            if is_training: # Apply data augmentation if training
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.2)
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            return img
        img = load_and_preprocess_image(image_path)
        return img, label
    image_ds = tf.data.Dataset.from_tensor_slices(image_indices)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((image_ds, label_ds))

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
            ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def create_model(input_shape=(160, 160, 3), num_classes = 1):
    """Builds a classification model using DenseNet121 as the backbone."""
    with strategy.scope(): # Use the distributed training strategy
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Convert feature maps into a vector
        x = Dense(256, activation='relu')(x) # Fully connected layer
        predictions = Dense(num_classes, activation='sigmoid', dtype='float32')(x)  # Output layer
        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the first 100 layers, train the rest
        for layer in base_model.layers[:100]:
                layer.trainable = False
        for layer in base_model.layers[100:]:
                layer.trainable = True

        # Define learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='binary_acc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
    
    return model

def predict_on_test_set(model, test_indices, test_labels, image_dir, batch_size=32, img_size=(160,160)):
      test_ds = create_dataset(test_indices, test_labels, batch_size=batch_size, img_size=img_size, image_dir=image_dir, is_training=False)
      predictions = model.predict(test_ds)
      return predictions
        

def main():
    data_path = "Data_Entry_2017_v2020.csv"
    images_folder = 'images'
    batch_size = 32 * strategy.num_replicas_in_sync  # Adjust batch size based on available GPUs
    epochs = 10
    img_size = (160, 160)

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load dataset and preprocess labels    
    df = pd.read_csv(data_path)
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    df['Is_Finding'] = df['Finding Labels'].apply(lambda x: 1 if x != ['No Finding'] else 0)
    
    # Split dataset into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['Image Index'], df['Is_Finding'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Create datasets
    train_ds = create_dataset(X_train.values, y_train.values, batch_size=batch_size, 
                             img_size=img_size, image_dir=images_folder, is_training=True)
    val_ds = create_dataset(X_val.values, y_val.values, batch_size=batch_size, 
                           img_size=img_size, image_dir=images_folder, is_training=False)
    
    model = create_model(input_shape=(*img_size, 3), num_classes=1)
    
    checkpoint = ModelCheckpoint(
        'models/chest_xray_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    os.makedirs('tf_cache', exist_ok=True)
    cached_train = train_ds.cache(f'./tf_cache/train_cache')
    cached_val = val_ds.cache(f'./tf_cache/val_cache')
    
    history = model.fit(
        cached_train,
        epochs=epochs,
        validation_data=cached_val,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('results/training_history.csv', index=False)
    
    y_pred = predict_on_test_set(
        model, 
        X_val.values, 
        y_val.values,
        images_folder, 
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Evaluate model
    y_pred_classes = (y_pred > 0.75).astype(int)
    y_val_array = y_val.values
    
    f1 = f1_score(y_val_array, y_pred_classes)
    print("F1:", f1)
    print(classification_report(y_val_array, y_pred_classes))
    
    results_df = pd.DataFrame({
        'Image_Index': X_val.values,
        'Actual': y_val_array,
        'Predicted_Prob': y_pred.flatten(),
        'Predicted_Class': y_pred_classes.flatten()
    })
    results_df.to_csv('results/validation_predictions.csv', index=False)

if __name__ == '__main__':
        main()