import sys
import os
import struct

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from PIL import Image
from skimage.feature import peak_local_max
from matplotlib import patches
from matplotlib.ticker import MultipleLocator

def read_pma_f0(pma_file_path):
    try:
        with open(pma_file_path, "rb") as f:
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            f.seek(0, 2)
            f.seek(0, 4) 
            frame_data0 = f.read(X_pixels * Y_pixels)
            image_data = np.frombuffer(frame_data0, dtype=np.uint8).reshape((Y_pixels, X_pixels))
            return image_data

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None

def read_pma(pma_file_path):
    try:
        with open(pma_file_path, "rb") as f:
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            f.seek(0, 2) 
            filesize = f.tell()  
            Nframes = (filesize - 4) // (X_pixels * Y_pixels)
            f.seek(4)  
            return [np.frombuffer(f.read(X_pixels*Y_pixels), dtype=np.uint8).reshape((Y_pixels, X_pixels)) for frame_idx in range(Nframes)]

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None
    
def generate_images(pma_file_path):
    try:
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        if not os.path.exists(f"{output_name}_Files"):
            os.makedirs(f"{output_name}_Files")
        else:
            print(f"Directory already exists: {output_name}_Files")
            return None
        
        Frames_data = read_pma(pma_file_path)
        for frame_idx, frame_data in enumerate(Frames_data):
            plt.imsave(f"{output_name}_Files/{output_name}frame_{frame_idx}.png", frame_data, cmap='gray')

    except Exception as e:
        print(f"Error generating images or creating directory: {e}")
        return None
    

def generate_mp4(images_path, fps=100):
    try:
        pma_name = f"{images_path.split('_')[-2]}"
        video_name = f"{pma_name}.mp4"
        video_file= os.path.join(images_path, f"{pma_name}_Video")
        
        if not os.path.exists(video_file):
            os.makedirs(video_file)
        else:
            print(f"Directory already exists: {video_file}")
            return None
            
        images = [img for img in os.listdir(images_path) if img.endswith(".png")]
        images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width,_ = frame.shape
        video = cv2.VideoWriter(os.path.join(video_file, video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        video.release()
        cv2.destroyAllWindows()

        print(f"Video sucessfully generated and saved as: {video_name}")
        print(f"Images: {(images)}")
    
    except Exception as e:
        print(f"Error generating video: {e}")
        #delete the directory
        if os.path.exists(video_file):
            os.rmdir(video_file)
        else:
            print(f"Directory does not exist: {video_file}")
        return None
    
def avg_frame_arr(pma_file_path):
    try:

        Frames_data = read_pma(pma_file_path)
        avg_frame_data = np.mean(Frames_data, axis=0).astype(np.uint8)
        print(f"Sucessfully generated average frame")
        return avg_frame_data

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None


def avg_frame_png(pma_file_path):
    try:
        avg_frame_data = avg_frame_arr(pma_file_path)
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        image_file_name = f'{output_name}_Avg_Frame.png'
        if not os.path.exists(f"{output_name}_Avg_Frame"):
            os.makedirs(f"{output_name}_Avg_Frame")
        else:
            pass
        image = Image.fromarray(avg_frame_data)
        image.save(f"{output_name}_Avg_Frame/{image_file_name}")
        print(f"Average frame saved as: {image_file_name}")

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None

def dim_to_3(image):
    return np.stack((image,) * 3, axis=-1)

def find_peaks_scipy_IDL(image_path, sigma=3, block_size=16, scaler_percent=32):
    std = 4*sigma
    # Load image (assumes grayscale uint8 image)
    image = io.imread(image_path, as_gray=True).astype(np.uint8)
    height, width = image.shape
    image_1 = image.copy()
    min_intensity = np.min(image_1)
    max_intensity = np.max(image_1)
    threshold = min_intensity + (scaler_percent / 100.0) * (max_intensity - min_intensity)
        
    background = np.zeros((height, width), dtype=np.float32)

    for i in range(8, height, block_size):
        for j in range(8, width, block_size):
            background[(i-8)//block_size, (j-8)//block_size] = np.min(image_1[i-8:i+8, j-8:j+8])
        

    background = np.clip(background.astype(np.uint8) - 10, 0, 255)
    image_1 = image - background
    image_2 = image_1.copy()
    med = np.median(image_1)
    image_2[image_2 < (med + 3*std)] = 0
    peak_coords = peak_local_max(image_2, min_distance=int(sigma), threshold_abs=threshold)
    
    return peak_coords, image_2


def good_peak_finder(image_path, sigma=3, block_size=16, scaler_percent=32, boarder=10, max_rad=3):
    peaks_coords_IDL, image_2 = find_peaks_scipy_IDL(image_path, sigma, block_size, scaler_percent)
    large_peaks = []
    correct_size_peaks = []
    height, width = io.imread(image_path).shape

    for peak in peaks_coords_IDL:
        y, x = peak
        # Extract the peak region, if pixels outside of 5x5 region are non-zero, then append peak to large_peaks
        if y < boarder or y > height - boarder or x < boarder or x > width - boarder:
            large_peaks.append(peak)
        elif image_2[y, x + max_rad+1] > 0 or image_2[y, x - max_rad] > 0 or image_2[y+max_rad+1, x ] > 0 or image_2[y-max_rad, x] > 0 or peak[0] < boarder or peak[0] > height - boarder or peak[1] < boarder or peak[1] > width - boarder:
            large_peaks.append(peak)
        else:
            correct_size_peaks.append(peak)

    correct_size_peaks = np.array(correct_size_peaks)
    large_peaks = np.array(large_peaks)
    
    return correct_size_peaks, large_peaks

def shift_peaks(peaks, shift=[0, 256]):
    return np.add(peaks, shift)

#change the arrow colour to white
def init_annot(ax, text="", xy=(0, 0), xytext=(0, 10),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->", color="w")):
    global annot
    annot = ax.annotate(text, xy=xy, xytext=xytext, textcoords=textcoords, bbox=bbox, arrowprops=arrowprops)
    annot.set_visible(False)
    return annot

# Function to update annotation text and position
def update_annot(ind, scatter, peaks, label):
    """ Updates the annotation position and text """
    idx = ind["ind"][0]
    y, x = peaks[idx]
    annot.xy = (scatter.get_offsets()[idx][0], scatter.get_offsets()[idx][1])
    annot.set_text(f"{label} Peak {idx}: (y, x) = ({y}, {x})")
    annot.set_visible(True)

def print_coords_trigger(event, fig, scatter_data):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            if event.name == "button_press_event":
        
                print(f"{label}_Peak{ind['ind'][0]} (y, x):({peaks[ind['ind'][0]][0]},{peaks[ind['ind'][0]][1]})")
            break

    annot.set_visible(visible)
    fig.canvas.draw_idle()

def find_pairs(peaks_1, peaks_2, tolerance=1, Channel_count=2, shift=[0,0]):
    # peaks_2 coordinates goes from [0, 512] to [256,512]
    gp1_list = [tuple(peak) for peak in peaks_1]
    gp2_list = [tuple(peak) for peak in peaks_2]
    gp2_set = set(gp2_list)
    pair_arr_CH1 = []
    pair_arr_CH2 = []
    try: 
        if Channel_count == 2:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (256-tolerance <= abs(coord[1] - c[1]) <= 256+tolerance) and c not in pair_arr_CH2:
                            pair_arr_CH1.append(coord)
                            pair_arr_CH2.append(c)
                            break
        elif Channel_count == 1:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (abs(coord[1] - c[1]) <= tolerance) and c not in pair_arr_CH2:
                            pair_arr_CH1.append(coord)
                            pair_arr_CH2.append(c)
                            break
        else:
            print("Invalid Channel Count, please choose 1 or 2")
            return None

        if shift == [0, 0]:
            out_pair_arr_CH1 = np.array(pair_arr_CH1)
            out_pair_arr_CH2 = np.array(pair_arr_CH2)

        else:

            pair_arr_CH1 = np.array(pair_arr_CH1)
            pair_arr_CH2 = shift_peaks(np.array(pair_arr_CH2), shift = [-shift[0], -shift[1]])
            out_pair_arr_CH2 = pair_arr_CH2[(pair_arr_CH2[:,1] <= 502) & (pair_arr_CH2[:,1] >= 266) & (pair_arr_CH2[:, 0] <= 502) & (pair_arr_CH2[:, 0] >= 10)]
            out_pair_arr_CH1 = pair_arr_CH1[(pair_arr_CH2[:,1] <= 502) & (pair_arr_CH2[:,1] >= 266) & (pair_arr_CH2[:, 0] <= 502) & (pair_arr_CH2[:, 0] >= 10)]

    except Exception as e:
        print(f"Error finding pairs: {e}")
        return None

    return len(out_pair_arr_CH1), out_pair_arr_CH1, out_pair_arr_CH2


def find_polyfit_params(peaks_1, peaks_2, degree=2):
    y1, x1 = peaks_1[:, 0], peaks_1[:, 1] 
    y2, x2 = peaks_2[:, 0], peaks_2[:, 1] 

    # Fit polynomials for x and y separately
    params_x = np.polyfit(x1, x2, degree)  # Fit x transformation
    params_y = np.polyfit(y1, y2, degree)  # Fit y transformation

    return params_x, params_y  # Returns polynomial coefficients

def apply_polyfit_params(CH1_peaks, params_x, params_y):
    y1, x1 = CH1_peaks[:, 0], CH1_peaks[:, 1]
    x_mapped = np.polyval(params_x, x1)  # Apply X transformation
    y_mapped = np.polyval(params_y, y1)  # Apply Y transformation
    return np.column_stack((y_mapped, x_mapped))  # Return transformed points


# Midpoint circle algorithm 
def draw_circle(radius, y_centre, x_centre, background_dim, colour = [255, 255, 0]):
    circle_array = np.zeros((background_dim, background_dim, 3), dtype=np.uint8)
    # Midpoint circle algorithm
    y = radius
    x = 0
    p = 1 - radius
    
    while y >= x:
        circle_array[y_centre + y, x_centre + x] = colour
        circle_array[y_centre - y, x_centre + x] = colour
        circle_array[y_centre + y, x_centre - x] = colour
        circle_array[y_centre - y, x_centre - x] = colour
        circle_array[y_centre + x, x_centre + y] = colour
        circle_array[y_centre - x, x_centre + y] = colour
        circle_array[y_centre + x, x_centre - y] = colour
        circle_array[y_centre - x, x_centre - y] = colour
         
        x += 1
        if p <= 0:
            p = p + 2 * x + 1
        else:
            y -= 1
            p = p + 2 * x - 2 * y + 1
    
    return circle_array

#changed the arguments, please edit in jupyter scripts!
def plot_circle(image, y_centre, x_centre, radius=4, colour = [255, 255, 0]):
    circle_array = draw_circle(radius, y_centre, x_centre, image.shape[0])
    mask = (circle_array == [255, 255, 0]).all(axis=-1)
    try:
        if image.ndim == 2:
            image_3d = np.repeat(image[..., np.newaxis], 3, -1)
        elif image.ndim==3 and image.shape[2]==3:
            image_3d = image
    except Exception as e:
        print(f"Error plotting circle: {e}")
        return None
    
    # Set the pixels in the mask to be yellow
    image_3d[mask] = colour
    # Display the modified image

    plt.imshow(image_3d)
    plt.show()

def count_circle(radius, y_centre=12, x_centre=12):
    total = 0
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total

def SG_background_subtraction(pma_file_path, input_array, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    frames_data = read_pma(pma_file_path) 
    height, width, _ = input_array.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    filled_circle_mask = np.zeros((height, width), dtype=bool)

    if not CH_consideration:
        all_peaks_intensity = 0
        total_intensity = np.sum(input_array[:, :,2])
        num_of_peaks = len(y_centre_arr)
        num_of_frame_pixels = height * width

        num_of_peak_pixels = count_circle(radius) * num_of_peaks #each channel
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask
        
        all_peaks_intensity += np.sum(input_array[filled_circle_mask, 2])
        intensity_to_remove = ((total_intensity-all_peaks_intensity) // (num_of_frame_pixels-num_of_peak_pixels))
        for frame in frames_data: #frame is 1D
            frame = frame.astype(np.int16)
            frame = np.clip(frame - intensity_to_remove, 0, 255).astype(np.uint8)
            corrected_frames_data.append(frame)

    else: 
        all_peaks_intensity_CH1 = 0
        all_peaks_intensity_CH2 = 0

        total_intensity_CH1 = np.sum(input_array[:, :width//2,2])
        total_intensity_CH2 = np.sum(input_array[:, width//2:,2])
        num_of_peaks = len(y_centre_arr)//2 #this is in each channel
        num_of_frame_pixels = height*width//2

        num_of_peak_pixels = count_circle(radius) * num_of_peaks #each channel
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask

        # split the filled_circle_mask into two channels
        filled_circle_mask_CH1 = filled_circle_mask[:, :width//2]
        filled_circle_mask_CH2 = filled_circle_mask[:, width//2:]
            
        all_peaks_intensity_CH1 += np.sum(input_array[:, :width//2, 2][filled_circle_mask_CH1])
        all_peaks_intensity_CH2 += np.sum(input_array[:, width//2:, 2][filled_circle_mask_CH2])

        intensity_to_remove_CH1 = ((total_intensity_CH1-all_peaks_intensity_CH1) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
        intensity_to_remove_CH2 = ((total_intensity_CH2-all_peaks_intensity_CH2) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
    
        for frame in frames_data: #frame is 1D
            frame = frame.astype(np.int16)
            frame_CH1 = np.clip(frame[:,:width//2] - intensity_to_remove_CH1, 0, 255).astype(np.uint8)
            frame_CH2 = np.clip(frame[:,width//2:] - intensity_to_remove_CH2, 0, 255).astype(np.uint8)
            frame = np.concatenate((frame_CH1, frame_CH2), axis=1)
            corrected_frames_data.append(frame)

    return corrected_frames_data

def DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    
    frames_data = read_pma(pma_file_path)
    height, width = frames_data[0].shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    filled_circle_mask = np.zeros((height, width), dtype=bool)
    corrected_frames_data = []

    if not CH_consideration:
        num_of_peaks = len(y_centre_arr)
        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        num_of_frame_pixels = height * width

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask |= mask
        
        for frame in frames_data:  
            all_peaks_intensity = np.sum(frame[filled_circle_mask])
            total_intensity = np.sum(frame)
            intensity_to_remove = np.int16((total_intensity - all_peaks_intensity) // (num_of_frame_pixels - num_of_peak_pixels))
            frame = frame.astype(np.int16) 
            frame -= intensity_to_remove 
            frame = np.clip(frame, 0, 255).astype(np.uint8)  # Clip and convert back
            corrected_frames_data.append(frame)

    else: 
        num_of_peaks = len(y_centre_arr)//2 #this is in each channel
        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        num_of_frame_pixels = frames_data[0].shape[0] * frames_data[0].shape[1]//2

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask |= mask
        
        filled_circle_mask_CH1 = filled_circle_mask[:, :width//2]
        filled_circle_mask_CH2 = filled_circle_mask[:, width//2:]
        
        for frame in frames_data:
            all_peaks_intensity_CH1= np.sum(frame[:, :width//2][filled_circle_mask_CH1])
            all_peaks_intensity_CH2= np.sum(frame[:, width//2:][filled_circle_mask_CH2])
            total_intensity_CH1 = np.sum(frame[:, : width//2])
            total_intensity_CH2 = np.sum(frame[:, width//2:])
            intensity_to_remove_CH1 = np.int16((total_intensity_CH1 - all_peaks_intensity_CH1) // (num_of_frame_pixels - num_of_peak_pixels)).astype(np.int16)
            intensity_to_remove_CH2 = np.int16((total_intensity_CH2 - all_peaks_intensity_CH2) // (num_of_frame_pixels - num_of_peak_pixels)).astype(np.int16)
            frame.astype(np.int16)
            frame_CH1 = np.clip(frame[:,:width//2] - intensity_to_remove_CH1, 0, 255).astype(np.uint8)
            frame_CH2 = np.clip(frame[:,width//2:] - intensity_to_remove_CH2, 0, 255).astype(np.uint8)
            frame = np.concatenate((frame_CH1, frame_CH2), axis=1)
            corrected_frames_data.append(frame)
    return corrected_frames_data


def on_hover(event, fig, ax, scatter_data, image_3d, image_orig, zoom_size=6, CH1_zoom_axes=[0.75, 0.6, 0.2, 0.2], CH2_zoom_axes=[0.75, 0.3, 0.2, 0.2]):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:  # Keep the main axis
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]

                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title("")
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH2})")
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                ax_zoom_CH2.clear()
            
                
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)
                

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def on_hover_intensity(event, pma_file_path, fig, ax, scatter_data, y_centre_arr, x_centre_arr, image_3d, image_orig, mask, radius=4, tpf=1/100, R_0=5.6, time_interval=1, background_treatment = "None", CH_consideration=False, Intense_axes_CH1=[0.48, 0.81, 0.5, 0.15], Intense_axes_CH2=[0.48, 0.56, 0.5, 0.15], FRET_axes=[0.48, 0.31, 0.5, 0.15], dist_axes=[0.48, 0.06, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.22, 0.06, 0.15, 0.15]):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    zoom_size=6
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                if background_treatment == "None":
                    Frames_data = read_pma(pma_file_path)
                elif background_treatment == "SG":
                    Frames_data = SG_background_subtraction(pma_file_path, image_3d, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                elif background_treatment == "DG":
                    Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)


                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:  # Keep the main axis
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                ax_intensity_CH1= fig.add_axes(Intense_axes_CH1)
                ax_intensity_CH2= fig.add_axes(Intense_axes_CH2)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)):

                    # transforms from 2D to 3D
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                
                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))

                ax_intensity_CH1.clear()
                ax_intensity_CH1.plot(time, tot_intensity_all_frames_CH1, color='g', label='CH2')
                ax_intensity_CH1.set_title(f"Intensity v Time in Donor Peak {idx}")
                ax_intensity_CH1.set_xlabel('Time (s)')
                ax_intensity_CH1.set_ylabel('Intensity')
                ax_intensity_CH1.set_ylim(-255, max(tot_intensity_all_frames_CH1)+255)
                ax_intensity_CH1.set_xlim(0, time[-1])
                ax_intensity_CH1.grid()
                ax_intensity_CH1.xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis
                ax_intensity_CH1.yaxis.set_major_locator(MultipleLocator(500))  # 500-unit intervals for y-axis

                ax_intensity_CH2.clear()
                ax_intensity_CH2.plot(time, tot_intensity_all_frames_CH2, color='b', label='CH2')
                ax_intensity_CH2.set_title(f"Intensity v Time in Acceptor Peak {idx}")
                ax_intensity_CH2.set_xlabel('Time (s)')
                ax_intensity_CH2.set_ylabel('Intensity')
                ax_intensity_CH2.set_ylim(-255, max(tot_intensity_all_frames_CH2)+255)
                ax_intensity_CH2.set_xlim(0, time[-1])
                ax_intensity_CH2.grid()
                ax_intensity_CH2.xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis
                ax_intensity_CH2.yaxis.set_major_locator(MultipleLocator(500))  # 500-unit intervals for y-axis

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color='r')
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel('Time (s)')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()
                ax_FRET.xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis


                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color='y')
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel('Time (s)')
                ax_dist.set_ylabel('Distance (nm)')
                ax_dist.set_xlim(0, time[-1])
                ax_dist.grid()
                ax_dist.xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis


                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def on_hover_intensity_merged(event, pma_file_path, fig, ax, scatter_data, y_centre_arr, x_centre_arr, image_3d, image_orig, mask, radius=4, tpf=1/100, R_0=5.6, time_interval=1, background_treatment = "None", CH_consideration=False, Intense_axes=[0.48, 0.6, 0.5, 0.3], FRET_axes=[0.48, 0.35, 0.5, 0.15], dist_axes=[0.48, 0.1, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.23, 0.06, 0.15, 0.15]):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    zoom_size=6
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                if background_treatment == "None":
                    Frames_data = read_pma(pma_file_path)
                elif background_treatment == "SG":
                    Frames_data = SG_background_subtraction(pma_file_path, image_3d, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                elif background_treatment == "DG":
                    Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                else:
                    Frames_data = read_pma(pma_file_path)

                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:  # Keep the main axis
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                ax_intensity= fig.add_axes(Intense_axes)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)): #for i in range(795): i= 0, 1, 2,..., 794

                    # transforms from 2D to 3D
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)

                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))
                ax_intensity.clear()
                ax_intensity.plot(time, tot_intensity_all_frames_CH1, color='g', label='CH1')
                ax_intensity.plot(time, tot_intensity_all_frames_CH2, color='b', label='CH2')
                ax_intensity.set_title(f"Intensity v Time in Peak {idx}")
                ax_intensity.set_xlabel('Time (s)')
                ax_intensity.set_ylabel('Intensity')
                ax_intensity.set_ylim(-255, max(max(tot_intensity_all_frames_CH1), max(tot_intensity_all_frames_CH2))+255)
                ax_intensity.legend(bbox_to_anchor=(1.0, 1.22), loc='upper right')
                ax_intensity.grid()
                ax_intensity.set_xlim(0, time[-1])
                ax_intensity.xaxis.set_major_locator(MultipleLocator(time_interval)) 
                ax_intensity.yaxis.set_major_locator(MultipleLocator(500))  

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color='r')
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel('Time (s)')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()
                ax_FRET.xaxis.set_major_locator(MultipleLocator(time_interval)) 


                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color='y')
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel('Time (s)')
                ax_dist.set_ylabel('Distance (nm)')
                ax_dist.set_xlim(0, time[-1])
                ax_dist.grid()
                ax_dist.xaxis.set_major_locator(MultipleLocator(time_interval))
            
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()

def count_circle(radius, y_centre=12, x_centre=12):
    total = 0
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total

def sgl_frame_intense_arr(input_array, radius, y_centre_arr, x_centre_arr):

    intensity_arr_all_peaks = []
    total_arr = []

    #filling in the circle
    for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
        total = 0
        for i in range(x_centre - radius, x_centre+ radius + 1):
            for j in range(y_centre - radius, y_centre + radius + 1):
                if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                    intensity_arr_all_peaks.append(input_array[j][i][2])
                    total += int(input_array[j][i][2])
        total_arr.append(total)

    return intensity_arr_all_peaks, total_arr

def intensity_in_circle(input_array, radius, y_centre, x_centre):
    total_intensity = 0
    intensity_arr = []
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                intensity_arr.append(int(input_array[j][i][2]))
                total_intensity += int(input_array[j][i][2])

    return total_intensity, intensity_arr

def calc_FRET(I_D_list, I_A_list):
    I_D, I_A = np.array(I_D_list), np.array(I_A_list)
    FRET_arr = I_A/(I_D + I_A)
    return FRET_arr.tolist()

def calc_distance(FRET_list, R_0):
    d = R_0 * ((1/np.array(FRET_list)) - 1)**(1/6)
    return d.tolist()

def display_time_series(pma_file_path, avg_image, peak_idx, CH1_arr, CH2_arr, tpf = 1/100, R_0=5.6, radius=4, time_interval=1, background_treatment = "None", CH_consideration=False, CH1_intensity_interval=500, CH2_intensity_interval=500, figsize=(15, 8)):
    y_CH1, x_CH1 = CH1_arr[peak_idx]
    y_CH2, x_CH2 = CH2_arr[peak_idx]
    tot_intensity_all_frames_CH1 = []
    tot_intensity_all_frames_CH2 = []
    y_centre_arr = np.concatenate((CH1_arr[:, 0], CH2_arr[:, 0]))
    x_centre_arr = np.concatenate((CH1_arr[:, 1], CH2_arr[:, 1]))
    if background_treatment == "None":
        Frames_data = read_pma(pma_file_path)
    elif background_treatment == "SG":
        Frames_data = SG_background_subtraction(pma_file_path, avg_image, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
    elif background_treatment == "DG":
        Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
    else:
        Frames_data = read_pma(pma_file_path)


    circle_array_new = draw_circle(4, y_centre_arr, x_centre_arr, avg_image.shape[0])
    mask = (circle_array_new == [255, 255, 0]).all(axis=-1)


    for i in range(len(Frames_data)): 
        if Frames_data[i].ndim == 2:
            frame_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
        elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
            frame_3d = Frames_data[i]
        frame_3d[mask] = [255, 255, 0]

        total_intensity_CH1,_ = intensity_in_circle(frame_3d, radius, y_CH1, x_CH1)
        total_intensity_CH2,_ = intensity_in_circle(frame_3d, radius, y_CH2, x_CH2)
        tot_intensity_all_frames_CH1.append(total_intensity_CH1)
        tot_intensity_all_frames_CH2.append(total_intensity_CH2)

    time = np.linspace(0, (len(tot_intensity_all_frames_CH1)-1)*tpf, len(tot_intensity_all_frames_CH1))
    
    
    fig, ax = plt.subplots(4, 1, figsize=figsize)
    fig.subplots_adjust(hspace=1)
    ax[0].plot(time, tot_intensity_all_frames_CH1, color='b')
    ax[0].set_title(f'Intensity v Time in Donor Peak {peak_idx}')
    ax[0].set_ylabel('Intensity')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylim(-255, max(tot_intensity_all_frames_CH1)+255)
    ax[0].set_xlim(0, time[-1])
    ax[0].grid()
    ax[0].xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis
    ax[0].yaxis.set_major_locator(MultipleLocator(CH1_intensity_interval))  # 500-unit intervals for y-axis


    ax[1].plot(time, tot_intensity_all_frames_CH2, color='g')
    ax[1].set_title(f'Intensity v Time in Acceptor Peak {peak_idx}')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylim(-255, max(tot_intensity_all_frames_CH2)+255)
    ax[1].grid()
    ax[1].set_xlim(0, time[-1])
    ax[1].xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis
    ax[1].yaxis.set_major_locator(MultipleLocator(CH2_intensity_interval))  # 500-unit intervals for y-axis

    FRET_values = calc_FRET(np.array(tot_intensity_all_frames_CH1), np.array(tot_intensity_all_frames_CH2))
    ax[2].plot(time, FRET_values, color='r')
    ax[2].set_title(f'FRET v Time (Peak {peak_idx})')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('FRET Efficiency')
    ax[2].grid()
    ax[2].set_xlim(0, time[-1])
    ax[2].xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis


    distance = calc_distance(FRET_values, R_0)
    ax[3].plot(time, distance, color='y')
    ax[3].set_title(f'Distance v Time (Peak {peak_idx})')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Distance (nm)')
    ax[3].grid()
    ax[3].set_xlim(0, time[-1])
    ax[3].xaxis.set_major_locator(MultipleLocator(time_interval))  # 1-second intervals for x-axis
    
    plt.show()


def find_polyfit_params_3CH(peaks_1, peaks_2, peaks_3, degree=2):
    y1, x1 = peaks_1[:, 0], peaks_1[:, 1] 
    y2, x2 = peaks_2[:, 0], peaks_2[:, 1] 
    y3, x3 = peaks_3[:, 0], peaks_3[:, 1]

    # Fit polynomials for x and y separately
    params_x_12 = np.polyfit(x1, x2, degree)  # Fit x transformation
    params_y_12 = np.polyfit(y1, y2, degree)  # Fit y transformation
    params_x_13 = np.polyfit(x1, x3, degree)  # Fit x transformation
    params_y_13 = np.polyfit(y1, y3, degree)  # Fit y transformation


    return params_x_12, params_y_12, params_x_13, params_y_13 # Returns polynomial coefficients

def find_trip(peaks_1, mapped_CH2, mapped_CH3, tolerance=4, shift_CH2=[0,0], shift_CH3=[0,0]):

    matched_CH1 = []
    matched_CH2 = []
    matched_CH3 = []

    gp1_list = [tuple(peak) for peak in peaks_1]
    mapped_CH2_set = set([tuple(peak) for peak in mapped_CH2])
    mapped_CH3_set = set([tuple(peak) for peak in mapped_CH3])

    for ch1_peak in gp1_list:
        y1, x1 = ch1_peak

        ch2_match = None
        for ch2_peak in mapped_CH2_set:
            y2, x2 = ch2_peak
            if abs(y1 - y2) <= tolerance and abs(x2 - x1 - 171) <= tolerance:
                ch2_match = ch2_peak
                break

        ch3_match = None
        for ch3_peak in mapped_CH3_set:
            y3, x3 = ch3_peak
            if abs(y1 - y3) <= tolerance and abs(x3 - x1 - 342) <= tolerance:
                ch3_match = ch3_peak
                break

        # If both matches exist, store the triplet
        if ch2_match is not None and ch3_match is not None:
            matched_CH1.append(ch1_peak)
            matched_CH2.append(ch2_match)
            matched_CH3.append(ch3_match)
    
    matched_CH1 = np.array(matched_CH1)
    matched_CH2 = shift_peaks(np.array(matched_CH2), shift=[-shift_CH2[0], -shift_CH2[1]])
    matched_CH3 = shift_peaks(np.array(matched_CH3), shift=[-shift_CH3[0], -shift_CH3[1]])
    out_pair_arr_CH2 = matched_CH2[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]
    out_pair_arr_CH1 = matched_CH1[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]
    out_pair_arr_CH3 = matched_CH3[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]

    return len(out_pair_arr_CH1), out_pair_arr_CH1, out_pair_arr_CH2, out_pair_arr_CH3

