import struct
import numpy as np
from skimage.feature import peak_local_max

def read_pma(pma_file_path):
    """
    Reads a .pma file and returns a list of frames as numpy arrays containing elements of type numpy.uint8.
    """
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

def avg_frame_arr(pma_file_path):
    """
    Reads a .pma file and returns the average frame as a numpy array.
    """
    try:

        Frames_data = read_pma(pma_file_path)
        avg_frame_data = np.mean(Frames_data, axis=0).astype(np.uint8)
        print(f"Sucessfully generated average frame")
        return avg_frame_data

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None

def find_peaks(image_path, sigma=3, block_size=16, scaler_percent=32):
    """
    From a grayscale image, finds all peaks bright spots in the image using a local maximum filter.
    """
    std = 4*sigma

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

def shift_peaks(peaks, shift=[0, 256]):
    """
    Shifts the peaks coordinates by a specified amount.
    """
    return np.add(peaks, shift)


def draw_circle(radius, y_centre, x_centre, background_dim, colour = [255, 255, 0]):
    """"
    Creates an array of the same size as the background_dim and draws cicles of a specified radius centred on coordinates (y_centre, x_centre).
    """
    circle_array = np.zeros((background_dim, background_dim, 3), dtype=np.uint8)

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

def count_circle(radius, y_centre=12, x_centre=12):
    """
    Counts the number of pixels in a circle of a specified radius centred on coordinates (y_centre, x_centre).
    """
    total = 0
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total


def SG_background_subtraction(pma_file_path, avg_frame_array, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    """
    Performs background subtraction using a static global calculation. Returns an array of corrected frames as numpy arrays with the same shape as the input array.
    """
    frames_data = read_pma(pma_file_path) 
    height, width, _ = avg_frame_array.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    filled_circle_mask = np.zeros((height, width), dtype=bool)

    if not CH_consideration:
        all_peaks_intensity = 0
        total_intensity = np.sum(avg_frame_array[:, :,2])
        num_of_peaks = len(y_centre_arr)
        num_of_frame_pixels = height * width

        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask
        
        all_peaks_intensity += np.sum(avg_frame_array[filled_circle_mask, 2])
        intensity_to_remove = ((total_intensity-all_peaks_intensity) // (num_of_frame_pixels-num_of_peak_pixels))
        for frame in frames_data: 
            frame = frame.astype(np.int16)
            frame = np.clip(frame - intensity_to_remove, 0, 255).astype(np.uint8)
            corrected_frames_data.append(frame)

    else: 
        all_peaks_intensity_CH1 = 0
        all_peaks_intensity_CH2 = 0

        total_intensity_CH1 = np.sum(avg_frame_array[:, :width//2,2])
        total_intensity_CH2 = np.sum(avg_frame_array[:, width//2:,2])
        num_of_peaks = len(y_centre_arr)//2 
        num_of_frame_pixels = height*width//2

        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask

  
        filled_circle_mask_CH1 = filled_circle_mask[:, :width//2]
        filled_circle_mask_CH2 = filled_circle_mask[:, width//2:]
            
        all_peaks_intensity_CH1 += np.sum(avg_frame_array[:, :width//2, 2][filled_circle_mask_CH1])
        all_peaks_intensity_CH2 += np.sum(avg_frame_array[:, width//2:, 2][filled_circle_mask_CH2])

        intensity_to_remove_CH1 = ((total_intensity_CH1-all_peaks_intensity_CH1) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
        intensity_to_remove_CH2 = ((total_intensity_CH2-all_peaks_intensity_CH2) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
    
        for frame in frames_data: 
            frame = frame.astype(np.int16)
            frame_CH1 = np.clip(frame[:,:width//2] - intensity_to_remove_CH1, 0, 255).astype(np.uint8)
            frame_CH2 = np.clip(frame[:,width//2:] - intensity_to_remove_CH2, 0, 255).astype(np.uint8)
            frame = np.concatenate((frame_CH1, frame_CH2), axis=1)
            corrected_frames_data.append(frame)

    return corrected_frames_data

def DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    """
    Performs background subtraction using a dynamic global calculation. Returns an array of corrected frames as numpy arrays with the same shape as the input array.
    """
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
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            corrected_frames_data.append(frame)

    else: 
        num_of_peaks = len(y_centre_arr)//2 
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

def update_annot(ind, scatter, peaks, label):
    """ 
    Updates the annotation position and text. 
    """
    idx = ind["ind"][0]
    y, x = peaks[idx]
    annot.xy = (scatter.get_offsets()[idx][0], scatter.get_offsets()[idx][1])
    annot.set_text(f"{label} Peak {idx}: (y, x) = ({y}, {x})")
    annot.set_visible(True)
    

def intensity_in_circle(input_array, radius, y_centre, x_centre):
    """
    From a 3D array, calculates the total intensity in a circle of a specified radius centred on coordinates (y_centre, x_centre).
    """
    total_intensity = 0
    intensity_arr = []

    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                intensity_arr.append(int(input_array[j][i][2]))
                total_intensity += int(input_array[j][i][2])

    return total_intensity, intensity_arr


def calc_FRET(I_D_list, I_A_list):
    """
    Calculates the FRET efficiency from the donor and acceptor intensities.
    """
    I_D, I_A = np.array(I_D_list), np.array(I_A_list)
    FRET_arr = I_A/(I_D + I_A)
    return FRET_arr.tolist()

def calc_distance(FRET_list, R_0):
    """
    Calculates the distance from the FRET efficiency using the Forster equation.
    """
    d = R_0 * ((1/np.array(FRET_list)) - 1)**(1/6)
    return d.tolist()

