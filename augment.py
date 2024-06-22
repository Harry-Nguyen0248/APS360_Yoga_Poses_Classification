import numpy as np
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# SOURCE: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
import os

def augment_image(image_path, save_path, num_images, filename):
    try:
        img = load_img(image_path)  # Load the image
        x = img_to_array(img)  # Convert the image to a numpy array
        x = np.expand_dims(x, axis=0)  # Add a dimension to the array

        # Create the datagen object with desired augmentations
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1./255, #dont remove this because it keeps all images same scale
        )

        # Generate and save 20 augmented images in a folder for the specific pose and specific image
        # Example: /augmented_images/pose1/image1/image1_aug_1.png
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            filename_without_extension = filename.split('.png')[0]
            augmented_images_save_path = save_path + '/{s}'.format(s=filename_without_extension)

            # if folder doesnt exist, create it
            if not os.path.exists(augmented_images_save_path):
                os.makedirs(augmented_images_save_path)

            augmented_image = array_to_img(batch[0], scale=True)

            # convert every other augmented image to grayscale
            # if (i % 2 == 0):
            #     augmented_image = augmented_image.convert('L')

            augmented_image.save(os.path.join(augmented_images_save_path, '{s}_aug_{i}.png'.format(s=filename_without_extension, i=i)))
            
            i += 1
            if i >= num_images:
                break

    except Exception as e:
        print(f"Error augmenting image {image_path}: {e}")


    
def augment_images_in_folder(image_folder_path, save_path, num_images):
    # Get the images from each subfolder in the folder and call augment_image on each
    for file in os.listdir(image_folder_path):
        filename = os.fsdecode(file)
        current_image_path = os.path.join(image_folder_path, filename)
        augment_image(current_image_path, save_path, num_images, filename)
        

def augment_images(dataset_path, save_path, num_images):
    # iterate each subfolder in the dataset folder
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            augment_images_in_folder(folder_path, save_path + "/" + folder, num_images)

if __name__ == "__main__":
    #add absolute paths for dataset and save location
    dataset_path = '/'
    save_path = '/'
    num_images = 3
    print("Augmentation started")
    augment_images(dataset_path, save_path, num_images)
    print("Augmentation complete")