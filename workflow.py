import os
import cv2
import numpy as np
import tensorflow as tf
import pyvista as pv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tensorflow.keras.preprocessing.image import img_to_array

# Load the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# GPU setup for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the VGG16 model for classification
model = tf.keras.models.load_model('vgg16_tree_classifier_augmented.h5')

# Helper Functions

def remove_small_masks(masks, min_area=10000):
    return [mask for mask in masks if np.sum(mask['segmentation']) >= min_area]

def resolve_overlaps(masks):
    keep_masks = np.ones(len(masks), dtype=bool)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(np.logical_and(masks[i]['segmentation'], masks[j]['segmentation'])):
                if np.sum(masks[i]['segmentation']) >= np.sum(masks[j]['segmentation']):
                    keep_masks[j] = False
                else:
                    keep_masks[i] = False
    return [masks[i] for i in range(len(masks)) if keep_masks[i]]

def classify_masked_regions(masked_image, model):
    masked_image_resized = cv2.resize(masked_image, (224, 224))
    masked_image_array = img_to_array(masked_image_resized)
    masked_image_array = np.expand_dims(masked_image_array, axis=0) / 255.0
    prediction = model.predict(masked_image_array)
    return prediction[0][0] > 0.5  # Returns True if classified as a tree

def convert_mask_to_depth_image(mask, scale=50):
    """Convert a binary or grayscale mask to a depth image by scaling pixel values."""
    return (mask.astype(float) / 255.0) * scale

def create_depth_colored_point_cloud(depth_image, bottom_layer_percentage=25):
    """Create a 3D point cloud and calculate red area percentage based on bottom layers."""
    if len(depth_image.shape) != 2:
        raise ValueError("Depth image must be a 2D grayscale array.")

    points, colors = [], []
    h, w = depth_image.shape
    z_values = depth_image
    unique_z_values = np.unique(z_values[z_values > 0])

    if len(unique_z_values) > 1:
        max_z, min_z = unique_z_values[-1], unique_z_values[0]
    else:
        print("No valid points in mask.")
        return np.array([]), np.array([]), 0

    # Calculate z-threshold for bottom layers
    z_threshold = min_z + (bottom_layer_percentage / 100.0) * (max_z - min_z)
    red_point_count = 0

    # Generate points and colors based on z-threshold
    for y in range(h):
        for x in range(w):
            z = z_values[y, x]
            if z > 0:
                points.append([x, y, z])
                if z <= z_threshold:
                    colors.append([1, 0, 0])  # Red for bottom layers
                    red_point_count += 1
                else:
                    colors.append([0, 1, 0])  # Green for other layers

    total_point_count = len(points)
    red_area_percentage_3d = (red_point_count / total_point_count) * 100 if total_point_count > 0 else 0

    return np.array(points), np.array(colors, dtype=np.float32), red_area_percentage_3d

def save_point_cloud(points, colors, save_path):
    """Save the 3D point cloud with color information to a VTP file."""
    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = colors
    point_cloud.save(save_path)

def visualize_and_save_point_cloud(points, colors, red_area_percentage_3d, screenshot_path, point_size=4):
    """Visualize the tree point cloud with 3D red area percentage displayed and save the screenshot."""
    if points.size == 0:
        print("No points to display.")
        return

    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = colors

    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=point_size, render_points_as_spheres=True)

    # Add text annotations for the 3D red area percentage
    plotter.add_text(f"3D Red Area: {red_area_percentage_3d:.2f}%", position="upper_left", font_size=14, color="white", shadow=True)
    plotter.view_isometric()

    # Save the screenshot with the 3D annotation
    plotter.screenshot(screenshot_path)
    plotter.close()

def calculate_red_area_from_mask(mask, output_path):
    """Calculate red area within tree mask and save annotated image."""
    # Convert to RGB if it's a grayscale image
    if len(mask.shape) == 2:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = mask

    # Step 1: Convert to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 2: Edge detection and dilation to close gaps
    edges = cv2.Canny(blurred_image, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Step 3: Find contours and assume the largest contour is the tree
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the tree based on the largest contour
        tree_mask = np.zeros_like(gray_image)
        cv2.drawContours(tree_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        tree_only = cv2.bitwise_and(mask_rgb, mask_rgb, mask=tree_mask)

        # Step 4: Find inner black areas (openings) within the tree
        gray_image_tree_only = cv2.cvtColor(tree_only, cv2.COLOR_RGB2GRAY)
        blurred_tree_only = cv2.GaussianBlur(gray_image_tree_only, (5, 5), 0)
        edges_tree_only = cv2.Canny(blurred_tree_only, 50, 150)
        contours_tree_only, _ = cv2.findContours(edges_tree_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_tree_only = np.zeros_like(gray_image_tree_only)

        # Fill openings within the tree area
        for contour in contours_tree_only:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask_tree_only, (x, y), (x + w, y + h), (255), -1)

        # Step 5: Highlight openings within the tree mask in red
        inverted_mask_tree_only = 255 - mask_tree_only
        image_with_red_areas = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        image_with_red_areas[(mask_tree_only == 0) & (tree_mask == 255)] = [0, 0, 255]

        # Step 6: Calculate areas
        red_area_inside_tree = np.sum((mask_tree_only == 0) & (tree_mask == 255))  # Red areas within tree
        tree_area = np.sum(tree_mask == 255)  # Total tree area

        # Debug statements to print areas
        print("Debug Information:")
        print(f"Total Tree Area (tree_area): {tree_area} pixels")
        print(f"Red Area Inside Tree (red_area_inside_tree): {red_area_inside_tree} pixels")
        image_area = mask.shape[0] * mask.shape[1]
        print(f"Total Image Area: {image_area} pixels")
        red_area_percentage = (red_area_inside_tree / tree_area) * 100 if tree_area > 0 else 0
        print(f"Calculated Red Area Percentage (red_area_percentage): {red_area_percentage:.2f}%")

        # Annotate the red area percentage on the image
        annotated_image = image_with_red_areas.copy()
        cv2.putText(annotated_image, f"Red Area: {red_area_percentage:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Save the annotated image
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Return the original mask, red area mask, annotated image, and red area percentage
        return mask_rgb, image_with_red_areas, annotated_image, red_area_percentage

    else:
        print("No contours found capable of isolating the tree.")
        return None
    
def ensure_rgb_image(image):
    """Ensure the image is in RGB format without applying any color transformations."""
    if len(image.shape) == 2:  # If it's a grayscale image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # Single-channel grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return image  # Already in RGB

def combine_images(original,annotated, screenshot_path, output_path):
    """
    Combine the original mask (in its original color), red area mask, annotated image, and 3D screenshot into one image.
    """
    # Load the 3D screenshot image
    screenshot = cv2.imread(screenshot_path)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

    # Ensure all images are in RGB format
    original_rgb = ensure_rgb_image(original)  # Display original mask without additional color mapping
    # red_area_rgb = ensure_rgb_image(red_area)  # Ensure red area mask is in RGB
    annotated_rgb = ensure_rgb_image(annotated)  # Ensure annotated mask is in RGB

    # Resize the screenshot to match the height of the other images if necessary
    target_height = original_rgb.shape[0]
    if screenshot.shape[0] != target_height:
        screenshot = cv2.resize(screenshot, (int(screenshot.shape[1] * target_height / screenshot.shape[0]), target_height))

    # Combine the images horizontally
    combined_image = np.hstack((original_rgb, annotated_rgb, screenshot))

    # Save the combined image
    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved to {output_path}")

def process_image_and_save(image_path, sam, model, min_mask_area, save_masks_folder):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=min_mask_area)
    masks = mask_generator.generate(image)

    filtered_masks = resolve_overlaps(remove_small_masks(masks, min_area=min_mask_area))

    for idx, mask in enumerate(filtered_masks):
        mask_bool = mask['segmentation'].astype(bool)
        masked_image = np.zeros_like(image)
        for c in range(3):
            masked_image[:, :, c] = image[:, :, c] * mask_bool

        is_tree = classify_masked_regions(masked_image, model)

        if is_tree:
            mask_grayscale = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

            # Calculate the 2D red area percentage and save the annotated mask
            red_area_image_path = os.path.join(save_masks_folder, f"{image_filename}_red_area_2d_{idx}.png")
            original_mask, red_area_image, annotated_image, red_area_percentage_2d = calculate_red_area_from_mask(mask_grayscale, red_area_image_path)

            # Convert mask to depth image and generate 3D point cloud with red area calculation
            depth_image = convert_mask_to_depth_image(mask_grayscale)
            points, colors, red_area_percentage_3d = create_depth_colored_point_cloud(depth_image)

            # Save the 3D point cloud to a file
            point_cloud_path = os.path.join(save_masks_folder, f"{image_filename}_point_cloud_{idx}.vtp")
            save_point_cloud(points, colors, point_cloud_path)

            # Save annotated visualization as a screenshot with the 3D red area percentage
            screenshot_path = os.path.join(save_masks_folder, f"{image_filename}_point_cloud_{idx}_annotated.png")
            visualize_and_save_point_cloud(points, colors, red_area_percentage_3d, screenshot_path)

            # Combine all images into one
            combined_image_path = os.path.join(save_masks_folder, f"{image_filename}_combined_{idx}.png")
            combine_images(masked_image, annotated_image, screenshot_path, combined_image_path)

            print(f"Processed mask {idx} for {image_filename}.")
            print(f"2D Red Area (Mask): {red_area_percentage_2d:.2f}%")
            print(f"3D Red Area (Point Cloud): {red_area_percentage_3d:.2f}%")
            print(f"Saved combined image at {combined_image_path}")

def main():
    image_folder = '100FPLAN'
    save_masks_folder = 'tree_masks_output_final'
    min_mask_area = 10000

    os.makedirs(save_masks_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            process_image_and_save(image_path, sam, model, min_mask_area, save_masks_folder)

if __name__ == "__main__":
    main()
