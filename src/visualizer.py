import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def visualize(name, pred, gt):
    # Convert name to string if it's a Tensor
    if hasattr(name, 'cpu'):
        name = name.item() if name.dim() == 0 else name[0] if isinstance(name, list) else str(name)
    name = str(name)
    
    # pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    video_frames = 'dataset5/frames/episode_' + name
    frames = sorted([f for f in os.listdir(video_frames) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    # print(frames)
    # print(f'Visualizing video {video_frames} with {len(frames)} frames, pred length: {len(pred)}, gt length: {len(gt)}')
    # print(f'len of pred: {len(pred)}, len of gt: {len(gt)}')
    # print(np.repeat(pred, 16))

    # create a video adding the frame and constructing iteratively a graphic with the gt and pred
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width = 480, 640
    combined_width = frame_width * 2
    out = cv2.VideoWriter('visualization/episode_' + name + '.mp4', fourcc, 10.0, (combined_width, frame_height))

    for i, frame in enumerate(frames):
        img = cv2.imread(os.path.join(video_frames, frame))
        img = cv2.resize(img, (frame_width, frame_height))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(pred[:i], label='Predicted Anomaly Score', color='red')
        plt.plot(gt[:i], label='Ground Truth', color='blue')
        plt.xlabel('Frame Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Score vs Ground Truth')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()

        # save the plot to a temporary file
        plot_path = 'temp_plot.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # read the plot and combine it with the frame
        plot_img = cv2.imread(plot_path)
        plot_img = cv2.resize(plot_img, (frame_width, frame_height))
        combined_img = np.hstack((img, plot_img))

        out.write(combined_img)

    out.release()
    os.remove(plot_path)
    print(f"Visualization saved as visualization/{name}.mp4")
    