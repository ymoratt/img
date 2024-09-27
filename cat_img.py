
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

 

def create_model():

    # Load the pre-trained MobileNetV2 model without the top layers

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

   

    # Add new layers

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(2, activation='softmax')(x)  # 2 classes: your cat and not your cat

   

    model = Model(inputs=base_model.input, outputs=predictions)

   

    # Freeze the base model layers

    for layer in base_model.layers:

        layer.trainable = False

   

    return model

 

def train_model(train_dir, validation_dir, epochs=10):

    model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
 

    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
 

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

 

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32)

    return model

 

# Example usage

train_dir = 'path/to/train/directory'
validation_dir = 'path/to/validation/directory'
model = train_model(train_dir, validation_dir)

 

# Save the model
model.save('my_cat_model.h5')

 

# To use the model for prediction later

def predict_cat(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.
 
    prediction = model.predict(img_array)

    return "It's your cat!" if prediction[0][0] > 0.5 else "It's not your cat."

 

# Example prediction

result = predict_cat('my_cat_model.h5', 'path/to/test/image.jpg')
print(result)