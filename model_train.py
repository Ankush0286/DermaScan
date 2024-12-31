import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential


#===========================================================================================
# Paths
train_dir = r".\output\train"
test_dir = r".\output\test"
val_dir = r".\output\val"

#===========================================================================================

# Load data
train_data = image_dataset_from_directory(
    train_dir, batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=True, seed=42
)
test_data = image_dataset_from_directory(
    test_dir, batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=False, seed=42
)
val_data = image_dataset_from_directory(
    val_dir, batch_size=32, image_size=(224, 224), label_mode='categorical', shuffle=False, seed=42
)

#===========================================================================================

# Get class names and class count
class_names = train_data.class_names
class_count = len(class_names)
print(f"Number of classes: {class_count}")
print(f"Class names: {class_names}")

# Prefetch data for optimization
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

#===========================================================================================

# Data augmentation to improve model generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

#===========================================================================================

# Base model - EfficientNetB2
base_model = tf.keras.applications.EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling='max'
)

# Add custom layers on top
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(0.016), 
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dense(128, kernel_regularizer=regularizers.l2(0.016), 
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=0.45, seed=42)(x)
output = Dense(class_count, activation='softmax')(x)


# Build the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

#===========================================================================================

# Model summary
model.summary()

#===========================================================================================

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# TensorBoard for real-time metrics visualization
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# Train the model
history = model.fit(
    train_data, 
    epochs=10, 
    validation_data=val_data, 
    callbacks=[early_stopping, tensorboard_callback]
)


#===========================================================================================

# Save the model in .keras format
model.save_weights("model_weights.h5")