import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from tensorflow.keras.applications import DenseNet121, EfficientNetB3, ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler

#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

def preprocess_image(image_path, img_size=(160, 160)):
    clean_path = ''.join(c for c in image_path if c.isprintable())
    img = load_img(clean_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array


def create_dataset(image_indices, labels, batch_size = 32, img_size=(160, 160), image_dir= 'images_folder', is_training=False):
    def process_path(image_index, label):
        image_index = tf.strings.strip(tf.strings.as_string(image_index))
        image_path = tf.strings.join([image_dir, "/", image_index])

        def load_and_preprocess_image(path):
            try:
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, img_size)
                img = img/ 255.0

                if is_training:
                    img = tf.image.random_flip_left_right(img)
                    img = tf.image.random_flip_up_down(img)
                    img = tf.image.random_brightness(img, max_delta=0.3)
                    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

                    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
                    
                    crop_size = tf.random.uniform(shape=[], 
                                                minval=tf.cast(tf.cast(img_size[0], tf.float32) * 0.8, tf.int32), 
                                                maxval=img_size[0], 
                                                dtype=tf.int32)
                    
                    crop_size = tf.minimum(crop_size, img_size[0])

                    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
                    img = tf.image.resize(img, img_size)
                    
                    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01)
                    img = tf.clip_by_value(img + noise, 0.0, 1.0)

                return img
            except tf.errors.InvalidArgumentError:
                tf.print(f"Error loading image: {path}")
                return tf.zeros(img_size + (3,))
        
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


def create_model(input_shape=(160, 160, 3), num_classes = 1, model_type='densenet'):
    with strategy.scope():
        if model_type == 'efficientnet':
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == 'resnet':
            base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:  # default to densenet
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Add dropout to prevent overfitting
        x = Dropout(0.3)(x)
        
        # Add batch normalization
        x = BatchNormalization()(x)
        
        # Multiple dense layers with dropouts for better feature extraction
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        predictions = Dense(num_classes, activation='sigmoid', dtype='float32')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        total_layers = len(base_model.layers)
        trainable_layers = int(total_layers * 0.3)

        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        
        trainable_layers = int(len(base_model.layers) * 0.3)
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        initial_learning_rate = 1e-3
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-5
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='binary_acc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.F1Score(name='f1_score', threshold=0.5)
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

def create_ensemble(img_size=(224, 224, 3), num_classes=1):
    """
    Create an ensemble of different models
    """
    architectures = [
        ('densenet', DenseNet121),
        ('efficientnet', EfficientNetB3),
        ('resnet', ResNet50V2)
    ]
    
    models = []
    for name, arch_class in architectures:
        model = create_model(input_shape=img_size, num_classes=num_classes, model_type=name)
        models.append(model)
    
    return models

def ensemble_predict(models, test_ds):
    """
    Generate predictions from an ensemble of models
    """
    all_predictions = []
    for model in models:
        preds = model.predict(test_ds)
        all_predictions.append(preds)
    
    # Average predictions from all models
    ensemble_preds = np.mean(all_predictions, axis=0)
    return ensemble_preds

def train_with_kfold(df, image_dir, k=5, img_size=(224, 224), batch_size=32, epochs=10):
    """
    Train models using K-fold cross-validation
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    # Extract features and labels
    X = df['Image Index']
    y = df['Is_Finding']
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n===== Training fold {fold+1}/{k} =====")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_ds = create_dataset(X_train.values, y_train.values.astype('float32'), batch_size=batch_size, 
                                 img_size=img_size, image_dir=image_dir, is_training=True)
        val_ds = create_dataset(X_val.values, y_val.values.astype('float32'), batch_size=batch_size, 
                               img_size=img_size, image_dir=image_dir, is_training=False)
        
        # Create model
        model = create_model(input_shape=(*img_size, 3), num_classes=1)
        
        # Calculate class weights
        class_counts = y_train.value_counts()
        total = class_counts.sum()
        class_weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            f'models/chest_xray_fold_{fold+1}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weights
        )
        
        # Evaluate on validation set
        y_pred = predict_on_test_set(model, X_val.values, y_val.values, image_dir, batch_size, img_size)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_val.values, y_pred_classes)
        auc = roc_auc_score(y_val.values, y_pred)
        
        print(f"Fold {fold+1} - F1: {f1:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_val.values, y_pred_classes))
        
        fold_results.append({
            'fold': fold+1,
            'f1': f1,
            'auc': auc,
            'model_path': f'models/chest_xray_fold_{fold+1}.h5'
        })
        
    # Save fold results
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv('results/kfold_results.csv', index=False)
    
    # Calculate average metrics
    avg_f1 = fold_df['f1'].mean()
    avg_auc = fold_df['auc'].mean()
    
    print(f"\n===== Cross-validation results =====")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    
    return fold_df

def main():
    """
    Main execution function
    """
    data_path = "Data_Entry_2017_v2020.csv"
    images_folder = 'final_images/images'
    batch_size = 32 * strategy.num_replicas_in_sync
    epochs = 15
    img_size = (224, 224)  # Standard size for many models
    use_kfold = True       # Whether to use k-fold cross-validation
    use_ensemble = True    # Whether to train an ensemble of models

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    df['Is_Finding'] = df['Finding Labels'].apply(lambda x: 1 if x != ['No Finding'] else 0)
    
    if use_kfold:
        # Use k-fold cross-validation
        print("Starting k-fold cross-validation training...")
        fold_results = train_with_kfold(
            df, 
            image_dir=images_folder, 
            k=5, 
            img_size=img_size, 
            batch_size=batch_size, 
            epochs=epochs
        )
    else:
        # Standard train/validation/test split
        print("Splitting data into train/validation/test sets...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['Image Index'], df['Is_Finding'], test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create datasets
        print("Creating TensorFlow datasets...")
        train_ds = create_dataset(X_train.values, y_train.values.astype('float32'), batch_size=batch_size, 
                                 img_size=img_size, image_dir=images_folder, is_training=True)
        val_ds = create_dataset(X_val.values, y_val.values.astype('float32'), batch_size=batch_size, 
                               img_size=img_size, image_dir=images_folder, is_training=False)
        test_ds = create_dataset(X_test.values, y_test.values.astype('float32'), batch_size=batch_size, 
                                img_size=img_size, image_dir=images_folder, is_training=False)
        
        # Calculate class weights to handle imbalance
        class_counts = y_train.value_counts()
        total = class_counts.sum()
        class_weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        print(f"Class weights: {class_weights}")
        
        if use_ensemble:
            # Train ensemble of models
            print("Training ensemble of models...")
            models = create_ensemble(img_size=(*img_size, 3), num_classes=1)
            
            for i, model in enumerate(models):
                print(f"\n===== Training model {i+1}/{len(models)} =====")
                model_name = f"model_{i+1}"
                
                # Callbacks
                checkpoint = ModelCheckpoint(
                    f'models/chest_xray_{model_name}.h5',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
                
                early_stopping = EarlyStopping(
                    monitor='val_auc',
                    patience=5,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_auc',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
                
                tensorboard = TensorBoard(
                    log_dir=f'logs/{model_name}',
                    histogram_freq=1,
                    write_graph=True
                )
                
                # Train model
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    validation_data=val_ds,
                    callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
                    class_weight=class_weights
                )
                
                # Save training history
                hist_df = pd.DataFrame(history.history)
                hist_df.to_csv(f'results/training_history_{model_name}.csv', index=False)
            
            # Generate ensemble predictions
            print("Generating ensemble predictions...")
            ensemble_preds = ensemble_predict(models, test_ds)
            ensemble_preds_classes = (ensemble_preds > 0.5).astype(int)
            
            # Evaluate ensemble
            f1 = f1_score(y_test.values, ensemble_preds_classes)
            auc = roc_auc_score(y_test.values, ensemble_preds)
            
            print(f"Ensemble - F1: {f1:.4f}, AUC: {auc:.4f}")
            print(classification_report(y_test.values, ensemble_preds_classes))
            
            # Save results
            results_df = pd.DataFrame({
                'Image_Index': X_test.values,
                'Actual': y_test.values,
                'Predicted_Prob': ensemble_preds.flatten(),
                'Predicted_Class': ensemble_preds_classes.flatten()
            })
            results_df.to_csv('results/ensemble_predictions.csv', index=False)
            
        else:
            # Train a single model
            print("Training single model...")
            model = create_model(input_shape=(*img_size, 3), num_classes=1)
            
            # Callbacks
            checkpoint = ModelCheckpoint(
                'models/chest_xray_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            
            early_stopping = EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            
            tensorboard = TensorBoard(
                log_dir='logs/single_model',
                histogram_freq=1,
                write_graph=True
            )
            
            # Cache datasets for faster training
            os.makedirs('tf_cache', exist_ok=True)
            cached_train = train_ds.cache(f'./tf_cache/train_cache')
            cached_val = val_ds.cache(f'./tf_cache/val_cache')
            
            # Train model
            history = model.fit(
                cached_train,
                epochs=epochs,
                validation_data=cached_val,
                callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
                class_weight=class_weights
            )
            
            # Save training history
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv('results/training_history.csv', index=False)
            
            # Evaluate on test set
            y_pred = predict_on_test_set(
                model, 
                X_test.values, 
                y_test.values,
                images_folder, 
                batch_size=batch_size,
                img_size=img_size
            )
            
            y_pred_classes = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            f1 = f1_score(y_test.values, y_pred_classes)
            auc = roc_auc_score(y_test.values, y_pred)
            
            print(f"Test - F1: {f1:.4f}, AUC: {auc:.4f}")
            print(classification_report(y_test.values, y_pred_classes))
            
            # Save results
            results_df = pd.DataFrame({
                'Image_Index': X_test.values,
                'Actual': y_test.values,
                'Predicted_Prob': y_pred.flatten(),
                'Predicted_Class': y_pred_classes.flatten()
            })
            results_df.to_csv('results/test_predictions.csv', index=False)
            
            # Also evaluate on validation set for comparison
            y_val_pred = predict_on_test_set(
                model, 
                X_val.values, 
                y_val.values,
                images_folder, 
                batch_size=batch_size,
                img_size=img_size
            )
            
            y_val_pred_classes = (y_val_pred > 0.5).astype(int)
            val_f1 = f1_score(y_val.values, y_val_pred_classes)
            
            print(f"Validation - F1: {val_f1:.4f}")
            print(classification_report(y_val.values, y_val_pred_classes))
            
            val_results_df = pd.DataFrame({
                'Image_Index': X_val.values,
                'Actual': y_val.values,
                'Predicted_Prob': y_val_pred.flatten(),
                'Predicted_Class': y_val_pred_classes.flatten()
            })
            val_results_df.to_csv('results/validation_predictions.csv', index=False)
    
    print("Training complete!")

if __name__ == '__main__':

        main()
