import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime
import streamlit as st
from io import BytesIO
import os

def main():
    # Use Streamlit's file uploader to allow image selection
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file into a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Extract the original file name without the extension
        original_name = os.path.splitext(uploaded_file.name)[0]

        heightx, widthx, _ = image.shape

        grid_size = st.sidebar.slider("Grid Size", min_value=1, max_value=20, value=1)
        brightness_factor = st.sidebar.slider("Brightness Factor", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
        num_colors_override = st.sidebar.slider("Number of Colors", min_value=1, max_value=200, value=5)
        hue_adjustment = st.sidebar.slider("Hue Adjustment", min_value=0, max_value=360, value=0)
        target_width = st.sidebar.slider("Target Width", min_value=int(0.2 * widthx), max_value=int(2 * widthx),
                                         value=widthx)
        target_height = st.sidebar.slider("Target Height", min_value=int(0.2 * heightx), max_value=int(2 * heightx),
                                          value=heightx)
        color_override = st.sidebar.slider("Color Override", min_value=0, max_value=1, value=0, step=1)

        process_images(image, grid_size, brightness_factor, num_colors_override, hue_adjustment, target_width,
                       target_height, color_override, original_name)
    else:
        st.info("Please upload an image to proceed.")

def process_images(image, grid_size, brightness_factor, num_colors_override, hue_adjustment, target_width,
                   target_height, color_override, original_name):
    target_shape = (target_width, target_height)
    images = []

    for row in range(grid_size):
        for col in range(grid_size):
            image_grid = get_image_grid(image, row, col, grid_size, target_shape)
            output_image = convert_to_color_by_number(image_grid, num_colors_override, brightness_factor,
                                                      hue_adjustment, color_override)
            output_image_with_text = write_grid_number(output_image, row, col, grid_size)
            images.append(output_image_with_text)

    final_image = imagegrid(images, grid_size)
    st.image(final_image, caption="Final Image", use_column_width=True, output_format="JPEG")  # Enable zooming
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_image_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    # Save option with download buttons
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_rgb = f"{original_name}_updated_{timestamp}.jpg"
    output_filename_gray = f"{original_name}_updated_gray_{timestamp}.jpg"

    # Convert RGB image to in-memory buffer for download
    is_success_rgb, buffer_rgb = cv2.imencode(".jpg", final_image_rgb)
    if is_success_rgb:
        st.download_button(
            label="Download Final RGB Image",
            data=BytesIO(buffer_rgb),
            file_name=output_filename_rgb,
            mime="image/jpeg"
        )

    # Convert Grayscale image to in-memory buffer for download
    is_success_gray, buffer_gray = cv2.imencode(".jpg", final_image_gray)
    if is_success_gray:
        st.download_button(
            label="Download Final Grayscale Image",
            data=BytesIO(buffer_gray),
            file_name=output_filename_gray,
            mime="image/jpeg"
        )

def get_image_grid(image, row, col, grid_size, target_shape):
    height, width, _ = image.shape
    start_row = int(row * height / grid_size)
    end_row = int((row + 1) * height / grid_size)
    start_col = int(col * width / grid_size)
    end_col = int((col + 1) * width / grid_size)
    cropped_image = image[start_row:end_row, start_col:end_col]
    resized_image = cv2.resize(cropped_image, target_shape)  # Resize the image to a common shape
    return resized_image

def imagegrid(images, grid_size):
    image_shape = images[0].shape
    grid_height = grid_size * image_shape[0]
    grid_width = grid_size * image_shape[1]
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            image_index = i * grid_size + j
            start_row = i * image_shape[0]
            end_row = (i + 1) * image_shape[0]
            start_col = j * image_shape[1]
            end_col = (j + 1) * image_shape[1]
            grid[start_row:end_row, start_col:end_col, :] = images[image_index]

    return grid

def convert_to_color_by_number(image, num_colors, brightness_factor, hue_adjustment, color_override):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flattened_image = image_rgb.reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(flattened_image)
    colors = kmeans.cluster_centers_.astype(int)

    if color_override == 1:
        middle_color = np.array([255, 0, 255])
        colors = []
        for i in range(num_colors):
            if i < num_colors // 2:
                j = num_colors // 2 - i
                hue = int((j / (num_colors // 2)) * 255)
                colors.append(np.array([255, 0, 255 - hue]))
            elif i > num_colors // 2:
                hue = int(((i - num_colors // 2) / (num_colors // 2)) * 255)
                colors.append(np.array([255 - hue, 0, 255]))
            else:
                colors.append(middle_color)

    colors = [np.array(color) for color in colors]
    adjusted_colors = np.clip((np.array(colors) * brightness_factor), 0, 255).astype(int)
    adjusted_colors_hsv = cv2.cvtColor(adjusted_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
    adjusted_colors_hsv[..., 0] = (adjusted_colors_hsv[..., 0] + hue_adjustment) % 180
    adjusted_colors_rgb = cv2.cvtColor(adjusted_colors_hsv, cv2.COLOR_HSV2RGB)

    color_by_number = np.squeeze(adjusted_colors_rgb, axis=0)[kmeans.labels_].reshape(image_rgb.shape)
    return color_by_number

def write_grid_number(image, row, col, grid_size):
    return image  # Optionally, add grid numbering logic here

if __name__ == "__main__":
    main()
