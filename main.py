import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to compute SSIM between two frames
def compute_ssim(frameA, frameB):/Users/bingqian/PycharmProjects/Similarity Detector/venv/main.py
    try:
        # Convert frames to grayscale
        grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two frames
        score, _ = ssim(grayA, grayB, full=True)
        return score
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        return None


# Function to compute MSE between two frames
def compute_mse(frameA, frameB):
    err = np.sum((frameA.astype("float") - frameB.astype("float")) ** 2)
    err /= float(frameA.shape[0] * frameA.shape[1])
    return err


# Function to compute histogram similarity between two frames
def compute_histogram_similarity(frameA, frameB):
    histA = cv2.calcHist([frameA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([frameB], [0], None, [256], [0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    # Compare histograms using correlation
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)dasd


# Main function to process the videos
def process_videos(video_path1, video_path2):
    # Load the two videos
    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)

    total_ssim_score = 0
    total_mse_score = 0
    total_histogram_score = 0
    frame_count = 0
    true_frames = 0  # Counter for frames with similarity > 70%

    # Read frames from both videos
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # If either video has ended, stop processing
        if not ret1 or not ret2:
            break

        # Ensure both frames are the same size (resize if necessary)
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Compute SSIM
        ssim_score = compute_ssim(frame1, frame2)
        if ssim_score is not None:
            total_ssim_score += ssim_score

        # Compute MSE
        mse_score = compute_mse(frame1, frame2)
        total_mse_score += mse_score

        # Compute histogram similarity
        hist_score = compute_histogram_similarity(frame1, frame2)
        total_histogram_score += hist_score

        frame_count += 1

    if frame_count == 0:
        print("No frames were processed.")
        return

    # Calculate average similarity metrics across frames
    avg_ssim = total_ssim_score / frame_count
    avg_mse = total_mse_score / frame_count
    avg_histogram = total_histogram_score / frame_count

    # Display the results
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average MSE: {avg_mse}")
    print(f"Average Histogram Similarity: {avg_histogram}")

    if avg_ssim > 0.9:
        print("Same videos.")
    else:
        print("Different videos")

    # Release video objects
    video1.release()
    video2.release()


# Entry point
if __name__ == "__main__":
    # Specify the paths to the two videos
    video_path1 = 'video1.mp4'
    video_path2 = 'video2.mp4'

    # Process and compare the two videos
    process_videos(video_path1, video_path2)