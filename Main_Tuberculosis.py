import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def unet_branch(input_tensor):
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    bn = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)

    u1 = UpSampling2D((2, 2))(bn)
    u1 = concatenate([u1, c3])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c2])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2))(c5)
    u3 = concatenate([u3, c1])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u3)

    return GlobalAveragePooling2D()(c6)
def build_ensemble_model(input_shape=(128, 128, 3)):
    input_tensor = Input(shape=input_shape)

    # U-Net Branch
    unet_features = unet_branch(input_tensor)

    # EfficientNetB0 Branch
    eff_base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
    eff_features = GlobalAveragePooling2D()(eff_base.output)

    # Xception Branch
    xcp_base = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
    xcp_features = GlobalAveragePooling2D()(xcp_base.output)

    # Combine features
    merged = concatenate([unet_features, eff_features, xcp_features])
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=input_tensor, outputs=output, name='TB_Ensemble_Model')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
data_path = '/content/drive/My Drive/Dataset/TB_Chest_Radiography_Database'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
model = build_ensemble_model()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)
